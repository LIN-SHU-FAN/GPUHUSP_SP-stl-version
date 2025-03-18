# GPUHUSP_SP-stl-version
/******************************************************************************
 * 自行實作 Prefix Sum + Scatter CUDA 範例 (自動挑選 block size)
 * 
 * 說明：
 *   - A: 長度 n，只含 0/1
 *   - 要將「A 中值為 1 的索引」寫到 B，最後 B 的大小為 (A 中 1 的總數)。
 *   - 實作步驟：
 *       1. 平行掃描 A，得到 scanA (inclusive)。
 *       2. 再以 scatter kernel，若 A[i] == 1，則 B[scanA[i] - 1] = i。
 *   - 這段程式碼示範多 block 的平行掃描，可處理 n > 一個 block 容量。
 *   - 另外示範如何根據 n 與 GPU 最大限制，自動選擇 blockSize。
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 檢查 CUDA 錯誤的輔助巨集
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), (int)err); \
        exit(-1); \
    }

/******************************************************************************
 * 根據輸入大小 n 與 GPU 屬性，選擇一個「不超過 maxThreadsPerBlock」且
 * 對齊 warpSize(32 的倍數) 的 blockSize。若 n 過小，也不需要太大的 blockSize。
 * 
 * 注意：這只是簡單參考做法。實務中還需考慮 shared memory、register usage 等因素，
 *       並可能做多次 benchmark 尋找最優。
 *****************************************************************************/
int pickBlockSize(int n)
{
    // 查詢 GPU 屬性
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int maxThreads     = prop.maxThreadsPerBlock; // 常見為 1024
    int warpSize       = prop.warpSize;           // 常見為 32
    if (warpSize == 0) warpSize = 32;             // 保底

    // 先取不超過 n 的值，若 n < maxThreads，沒必要開超過 n。
    int candidate = (n < maxThreads) ? n : maxThreads;

    // 再把 candidate 對齊到 warpSize 的倍數
    // 例如 32, 64, 96, 128, 160, 192, ..., <= candidate
    int remainder = candidate % warpSize;
    if (remainder != 0) {
        candidate -= remainder;  // 對齊到 32 倍數
        if (candidate < warpSize) {
            // 避免 candidate 掉到 0，至少是 32
            candidate = warpSize;
        }
    }

    // 若還要考慮對齊到 2 的次方，可在這邊進一步微調。但對於一般 kernel，
    // warpSize 對齊即可，大多情況足夠。
    
    // 為了避免「2 × blockSize」過大超過 shared memory，
    // 我們可以再加個限制: 2×candidate × sizeof(int) <= prop.sharedMemPerBlock
    // 這裡略做示範 (可能還有其他空間被占用)
    size_t neededSharedMem = 2ULL * candidate * sizeof(int);
    if (neededSharedMem > prop.sharedMemPerBlock) {
        // 若需要的 shared memory 超過限制，就縮小 blockSize (這裡簡化做法只砍半)
        while (neededSharedMem > prop.sharedMemPerBlock && candidate >= 32) {
            candidate >>= 1; // 砍半
            // 再對齊 32 倍數
            candidate = (candidate / warpSize) * warpSize;
            neededSharedMem = 2ULL * candidate * sizeof(int);
        }
    }

    // 最終保底至少 32
    if (candidate < 32) candidate = 32;

    return candidate;
}

/******************************************************************************
 * Kernel 1: 每個 block 負責對「部分區段」做掃描 (Blelloch-like)
 *  - d_in: 輸入 A
 *  - d_out: 輸出 scanA (只是部份)
 *  - d_blockSums: 每個 block 處理完後的區段總和 (最後元素)
 *****************************************************************************/
__global__
void blockScanKernel(const int* d_in,  // 輸入 A
                     int* d_out,      // 輸出 scanA (只是部份)
                     int* d_blockSums, // 每個 block 記錄該區段掃描後的最後值
                     int n,
                     int blockSize)    // 動態 blockSize
{
    // block 索引
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // 每個 block 預期處理 2 * blockSize 個元素
    int start = bx * (blockSize * 2);

    extern __shared__ int s_data[]; // 動態 shared memory

    // 載入資料到 shared memory
    int i = start + tx;
    if (i < n) {
        s_data[tx] = d_in[i];
    } else {
        s_data[tx] = 0; // 超過 n 範圍的填 0
    }

    int i2 = start + blockSize + tx;
    if ((tx + blockSize) < (2 * blockSize)) {
        if (i2 < n) {
            s_data[tx + blockSize] = d_in[i2];
        } else {
            s_data[tx + blockSize] = 0;
        }
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Blelloch Scan: 上掃 (reduce phase)
    // ------------------------------------------------------------------------
    for (int step = 1; step < 2 * blockSize; step <<= 1) {
        int idx = (tx + 1) * step * 2 - 1;
        if (idx < 2 * blockSize) {
            s_data[idx] += s_data[idx - step];
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Blelloch Scan: 下掃 (distribution phase)
    // ------------------------------------------------------------------------
    for (int step = (2 * blockSize) >> 1; step > 0; step >>= 1) {
        int idx = (tx + 1) * step * 2 - 1 + step;
        if (idx < 2 * blockSize) {
            s_data[idx] += s_data[idx - step];
        }
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // 把結果寫回 global memory
    // d_out[i], d_out[i2] 分別對應 s_data[tx], s_data[tx+blockSize]
    // ------------------------------------------------------------------------
    if (i < n) {
        d_out[i] = s_data[tx];
    }
    if (i2 < n && (tx + blockSize) < (2 * blockSize)) {
        d_out[i2] = s_data[tx + blockSize];
    }

    // 這個 block 處理的最後一個位置 (2*blockSize - 1) 就是本區塊的掃描總和
    // 注意要確定不會超過 n 範圍
    if (d_blockSums && tx == 0) {
        int lastIndex = (2 * blockSize) - 1;
        int realEndIndex = (start + lastIndex < n) ? lastIndex : (n - start - 1);
        d_blockSums[bx] = s_data[realEndIndex];
    }
}

/******************************************************************************
 * Kernel 2: 將之前的 blockSums 做掃描(通常資料量較小，可用同一段邏輯或更簡化)
 *****************************************************************************/
__global__
void scanBlockSumsKernel(int* d_blockSums)
{
    extern __shared__ int s_data[]; 

    int tx = threadIdx.x;
    int n = blockDim.x; // 這裡 n == gridDim.x

    // 載入 blockSums 進來
    if (tx < n) {
        s_data[tx] = d_blockSums[tx];
    } else {
        s_data[tx] = 0;
    }
    __syncthreads();

    // Blelloch 上掃
    for (int step = 1; step < n; step <<= 1) {
        int idx = (tx + 1) * step * 2 - 1;
        if (idx < n) {
            s_data[idx] += s_data[idx - step];
        }
        __syncthreads();
    }
    // 下掃
    for (int step = n >> 1; step > 0; step >>= 1) {
        int idx = (tx + 1) * step * 2 - 1 + step;
        if (idx < n) {
            s_data[idx] += s_data[idx - step];
        }
        __syncthreads();
    }

    // 寫回 global memory
    if (tx < n) {
        d_blockSums[tx] = s_data[tx];
    }
}

/******************************************************************************
 * Kernel 3: 加上前面 blocks 的偏移量
 *****************************************************************************/
__global__
void addBlockOffsetsKernel(int* d_out, const int* d_blockSums, int n, int blockSize)
{
    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // 每個 block 處理 2 * blockSize
    int start = bx * (blockSize * 2);

    // 需要加的偏移量
    int offset = (bx == 0) ? 0 : d_blockSums[bx - 1];

    int i  = start + tx;
    int i2 = start + tx + blockSize;

    if (i < n)
        d_out[i] += offset;
    if (i2 < n && (tx + blockSize) < (2 * blockSize))
        d_out[i2] += offset;
}

/******************************************************************************
 * Kernel 4: scatter 步驟
 *****************************************************************************/
__global__
void scatterKernel(const int* d_A, const int* d_scanA, int* d_B, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && d_A[i] == 1)
    {
        // inclusive scan -> 索引要 -1
        int pos = d_scanA[i] - 1;
        d_B[pos] = i;
    }
}

/******************************************************************************
 * prefixSumAndScatter:
 *   - d_A: 只含 0/1
 *   - d_B: 最後要裝下所有「A 中為 1 的索引」
 *   - d_scanA: 中間結果，用來存 prefix sum
 *   - n:   A 的長度
 *   - totalOnes: 傳回 A 中 1 的總數
 *****************************************************************************/
void prefixSumAndScatter(const int* d_A,
                         int* d_B,
                         int* d_scanA,
                         int  n,
                         int &totalOnes)
{
    if (n <= 0) {
        totalOnes = 0;
        return;
    }

    // 1) 依照 n 與 GPU 性能動態挑選 blockSize
    int blockSize = pickBlockSize(n);

    // 每個 block 處理 2 * blockSize
    int numBlocks = (n + (2 * blockSize) - 1) / (2 * blockSize);

    // 為了做「多 block 的 prefix sum」，要存每個 block 的總和
    int* d_blockSums = nullptr;
    if (numBlocks > 1) {
        CHECK_CUDA(cudaMalloc(&d_blockSums, numBlocks * sizeof(int)));
    }

    // ------------------------------------------------------------------------
    // 1. kernel 1: 對各 block 自己區段做掃描
    //    使用動態 shared memory: size = 2 * blockSize * sizeof(int)
    // ------------------------------------------------------------------------
    size_t smemSize = 2ULL * blockSize * sizeof(int);
    blockScanKernel<<<numBlocks, blockSize, smemSize>>>(d_A, d_scanA, d_blockSums, n, blockSize);
    CHECK_CUDA(cudaGetLastError());

    // 如果不只 1 個 block，還要把各 block 的最後值再做一次掃描
    if (numBlocks > 1) {
        // 1.1 kernel 2: 把 blockSums 本身再掃描
        //     只需要 1 個 block, block 大小 = numBlocks
        //     動態 shared memory: numBlocks * sizeof(int) 即可
        scanBlockSumsKernel<<<1, numBlocks, numBlocks * sizeof(int)>>>(d_blockSums);
        CHECK_CUDA(cudaGetLastError());

        // 1.2 kernel 3: 把前面 block 的偏移量加回
        addBlockOffsetsKernel<<<numBlocks, blockSize>>>(d_scanA, d_blockSums, n, blockSize);
        CHECK_CUDA(cudaGetLastError());
    }

    // 此時 d_scanA[i] 為「從 A[0] 到 A[i] 的 1 總數 (inclusive)」
    // 讀取最後一個元素 => A 中 1 的總個數
    CHECK_CUDA(cudaMemcpy(&totalOnes, &d_scanA[n-1], sizeof(int), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // 2. scatter: 依照 prefix sum 寫出所有「1 的索引」
    // ------------------------------------------------------------------------
    // 同樣可動態選個 scatter blockSize
    int blockScatter = pickBlockSize(n);
    int gridScatter  = (n + blockScatter - 1) / blockScatter;
    scatterKernel<<<gridScatter, blockScatter>>>(d_A, d_scanA, d_B, n);
    CHECK_CUDA(cudaGetLastError());

    // 回收
    if (d_blockSums) {
        cudaFree(d_blockSums);
    }
}


int main()
{
    // -----------------------------
    // 範例：輸入 A (host) = [0,1,0,0,1]
    // -----------------------------
    std::vector<int> hA = {0, 1, 0, 0, 1};
    int n = (int)hA.size();

    // 1) 配置並拷貝到 device
    int *dA, *dScanA, *dB;
    CHECK_CUDA(cudaMalloc(&dA,     n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dScanA, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dB,     n * sizeof(int))); // 最多不會超過 n 個「1」

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // 2) 呼叫 prefix sum + scatter
    int totalOnes = 0;
    prefixSumAndScatter(dA, dB, dScanA, n, totalOnes);

    // 3) 從 device 把 B 拉回 host
    std::vector<int> hB(totalOnes);
    CHECK_CUDA(cudaMemcpy(hB.data(), dB, totalOnes * sizeof(int), cudaMemcpyDeviceToHost));

    // 4) 印出結果
    std::cout << "A = [ ";
    for (auto &x : hA) std::cout << x << " ";
    std::cout << "]\n";

    std::cout << "B = [ ";
    for (auto &x : hB) std::cout << x << " ";
    std::cout << "]\n";  // 期待 [1, 4]

    std::cout << "Total ones: " << totalOnes << std::endl;

    // 收尾
    cudaFree(dA);
    cudaFree(dScanA);
    cudaFree(dB);
    return 0;
}
