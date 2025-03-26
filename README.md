# GPUHUSP_SP-stl-version
/*****************************************************************************
 * 編譯 (假設檔名為 dynamic_block_filter.cu)：
 *   nvcc dynamic_block_filter.cu -o dynamic_block_filter
 * 執行：
 *   ./dynamic_block_filter
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cassert>

/******************************************************************************
 * 根據輸入 n，回傳一個 <= 1024 的「2 的冪次方」block size。
 * 簡易策略：找到 >= n 的最小 2^k，如超過 1024 就取 1024。
 * 如果 n 很小，也別超過 n (最小至少 1)。
 *****************************************************************************/
int getOptimalBlockSize(int n)
{
    if (n <= 1) return 1; // 最小就給 1

    // 找到 >= n 的 2^k
    int p = 1;
    while (p < n) {
        p <<= 1;
        if (p > 1024) { // 超過硬體常見上限就停
            p = 1024;
            break;
        }
    }
    return p;
}

/******************************************************************************
 * Kernel 1) markKeepArray: 決定 keep[i] = (offset[i]!=0 ? 1 : 0)
 *****************************************************************************/
__global__
void markKeepArray(const int *offset, int *keep, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        keep[idx] = (offset[idx] != 0) ? 1 : 0;
    }
}

/******************************************************************************
 * (A) 在每個 block 內做 Blelloch exclusive prefix-sum，並將 block 總和存到 blockSums[]
 *     注意：此 kernel 一個 block 處理 2*blockDim.x 的資料
 *****************************************************************************/
__global__
void scanBlockKernel(const int *d_in, int *d_out, int *blockSums, int N)
{
    extern __shared__ int sh_data[];  // 動態 shared memory

    int tid = threadIdx.x;
    // 每個 block 負責 2*blockDim.x
    int gid = blockIdx.x * (blockDim.x * 2) + tid;

    // 載入 shared memory
    if (gid < N) {
        sh_data[tid] = d_in[gid];
    } else {
        sh_data[tid] = 0;
    }

    int gid2 = gid + blockDim.x;
    if (gid2 < N) {
        sh_data[tid + blockDim.x] = d_in[gid2];
    } else {
        sh_data[tid + blockDim.x] = 0;
    }
    __syncthreads();

    // Blelloch upsweep
    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            sh_data[bi] += sh_data[ai];
        }
        offset <<= 1;
    }
    __syncthreads();

    // 暫存最後總和到 blockSums
    if (tid == 0) {
        blockSums[blockIdx.x] = sh_data[2 * blockDim.x - 1];
        // exclusive：將最後歸 0
        sh_data[2 * blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Blelloch downsweep
    for (int d = 1; d < 2*blockDim.x; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(2*tid + 1) - 1;
            int bi = offset*(2*tid + 2) - 1;
            int t  = sh_data[ai];
            sh_data[ai] = sh_data[bi];
            sh_data[bi] += t;
        }
    }
    __syncthreads();

    // 寫回 global
    if (gid < N) {
        d_out[gid] = sh_data[tid];
    }
    if (gid2 < N) {
        d_out[gid2] = sh_data[tid + blockDim.x];
    }
}

/******************************************************************************
 * (C) addBlockSumsKernel:
 *   已經對 blockSums 做 prefix-sum 後，再將這些 prefix 值加回各 block 的資料
 *****************************************************************************/
__global__
void addBlockSumsKernel(int *d_data, const int *blockSums, int N)
{
    int blockId = blockIdx.x;
    if (blockId == 0) return; // 第 0 block 不需加任何前綴

    int start = blockId * (blockDim.x * 2);
    int end   = start + (blockDim.x * 2);
    int addVal = blockSums[blockId];  // 這個 block 要加的 offset

    // 利用多個 thread 并行加回
    int idx = start + threadIdx.x;
    while (idx < end && idx < N) {
        d_data[idx] += addVal;
        idx += blockDim.x;
    }
}

/******************************************************************************
 * prefixSumExclusiveLarge:
 *   對 d_data 進行「exclusive prefix-sum」(就地修改)，支援 N 可能非常大
 *   流程：block-level scan + (遞迴)掃描 blockSums + 加回 (addBlockSumsKernel)
 *****************************************************************************/
void prefixSumExclusiveLarge(int *d_data, int N)
{
    if (N <= 1) {
        // prefix-sum 也沒啥可做
        return;
    }

    // 動態計算 block size
    // 為簡化，我們一個 block 處理 2*blockDim.x 的元素
    // 所以 threads = getOptimalBlockSize(N/2)（或 N，隨你判斷）
    int threads = getOptimalBlockSize(N / 2);
    // 如果計算後仍為 0，至少設成 1
    if (threads < 1) {
        threads = 1;
    }

    // block 數量
    int blocks = (N + (threads * 2) - 1) / (threads * 2);

    // 配置一個 blockSums，用來收集各 block 的部分和
    int *d_blockSums = nullptr;
    cudaMalloc(&d_blockSums, blocks * sizeof(int));

    // (A) scanBlockKernel：每個 block 做部分掃描
    {
        dim3 grid(blocks);
        dim3 block(threads);
        // 需要 2*threads 個 int 的 shared memory
        size_t smemSize = 2 * threads * sizeof(int);

        scanBlockKernel<<<grid, block, smemSize>>>(d_data, d_data, d_blockSums, N);
        cudaDeviceSynchronize();
    }

    // (B) 如果 block 數量 > 1，要針對 d_blockSums 自己做 prefix-sum
    if (blocks > 1) {
        prefixSumExclusiveLarge(d_blockSums, blocks);
    }

    // (C) addBlockSumsKernel：把 prefix 運算結果加回 d_data
    {
        dim3 grid(blocks);
        dim3 block(threads);
        addBlockSumsKernel<<<grid, block>>>(d_data, d_blockSums, N);
        cudaDeviceSynchronize();
    }

    cudaFree(d_blockSums);
}

/******************************************************************************
 * scatterFilteredArray:
 *   依 keep[i] 分配 offset[i]/sid[i] 到 newOffset/newSid
 *****************************************************************************/
__global__
void scatterFilteredArray(const int *offset, const int *sid,
                          const int *keep, const int *keepScan,
                          int       *newOffset, int *newSid,
                          int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (keep[idx] == 1) {
            int pos = keepScan[idx];
            newOffset[pos] = offset[idx];
            newSid[pos]    = sid[idx];
        }
    }
}

/******************************************************************************
 * buildRealOffset:
 *   題主需求：newOffset => finalOffset，其中 finalOffset[0]=0, finalOffset[i+1]=newOffset[i]
 *   之後再對 finalOffset[1..validCount] 做 prefixSumExclusiveLarge
 *****************************************************************************/
__global__
void buildRealOffset(const int *newOffset, int *finalOffset, int validCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 只需一個 thread 設 finalOffset[0] = 0
    if (idx == 0) {
        finalOffset[0] = 0;
    }
    // 再把 newOffset[i] 搬到 finalOffset[i+1]
    if (idx < validCount) {
        finalOffset[idx + 1] = newOffset[idx];
    }
}

/******************************************************************************
 * Host helper: 印出陣列
 *****************************************************************************/
void printArray(const char *name, const int *arr, int len)
{
    printf("%s (len=%d): ", name, len);
    for (int i = 0; i < len; i++){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

/******************************************************************************
 * 主程式：示範對 offset=[0,4,1,0], sid=[1,5,8,10] 篩選
 *   也可改用大於 1024 的測資來測試
 *****************************************************************************/
int main()
{
    //--------------------------------------------------------------------------
    // 1) 準備資料 (小範例)
    //--------------------------------------------------------------------------
    int h_offset[] = {0, 4, 1, 0};
    int h_sid[]    = {1, 5, 8, 10};
    int N = 4;

    // 若要測試大於 1024，可改用動態分配 + 隨機值：
    /*
    N = 2000; 
    int *h_offset = (int*)malloc(N*sizeof(int));
    int *h_sid    = (int*)malloc(N*sizeof(int));
    for (int i=0; i<N; i++){
        h_offset[i] = ( (rand()%5==0) ? 0 : (rand()%10+1) );
        h_sid[i]    = rand()%100;
    }
    */

    // Device 配置
    int *d_offset=nullptr, *d_sid=nullptr;
    int *d_keep=nullptr, *d_keepScan=nullptr;
    cudaMalloc(&d_offset, N*sizeof(int));
    cudaMalloc(&d_sid,    N*sizeof(int));
    cudaMalloc(&d_keep,   N*sizeof(int));
    cudaMalloc(&d_keepScan,N*sizeof(int));

    cudaMemcpy(d_offset, h_offset, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sid,    h_sid,    N*sizeof(int), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // 2) markKeepArray
    //--------------------------------------------------------------------------
    {
        int blockSize = 256; // 這裡可再動態調整
        int gridSize  = (N + blockSize - 1) / blockSize;
        markKeepArray<<<gridSize, blockSize>>>(d_offset, d_keep, N);
        cudaDeviceSynchronize();
    }

    //--------------------------------------------------------------------------
    // 3) 對 keep 做 prefix-sum (exclusive)
    //--------------------------------------------------------------------------
    cudaMemcpy(d_keepScan, d_keep, N*sizeof(int), cudaMemcpyDeviceToDevice);
    prefixSumExclusiveLarge(d_keepScan, N);

    //--------------------------------------------------------------------------
    // 4) 算出有效元素數量 validCount
    //--------------------------------------------------------------------------
    int h_keepScanEnd = 0, h_keepLast=0;
    cudaMemcpy(&h_keepScanEnd, d_keepScan+(N-1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_keepLast,    d_keep+(N-1),     sizeof(int), cudaMemcpyDeviceToHost);

    int validCount = h_keepScanEnd + h_keepLast;

    //--------------------------------------------------------------------------
    // 5) scatter 到 newOffset, newSid
    //--------------------------------------------------------------------------
    int *d_newOffset=nullptr, *d_newSid=nullptr;
    cudaMalloc(&d_newOffset, validCount*sizeof(int));
    cudaMalloc(&d_newSid,    validCount*sizeof(int));

    {
        int blockSize = 256;
        int gridSize  = (N + blockSize - 1)/blockSize;
        scatterFilteredArray<<<gridSize, blockSize>>>(d_offset, d_sid,
                                                      d_keep, d_keepScan,
                                                      d_newOffset, d_newSid,
                                                      N);
        cudaDeviceSynchronize();
    }

    //--------------------------------------------------------------------------
    // 6) 把 newOffset 再做「真正 offset」
    //    finalOffset 長度 = validCount + 1
    //--------------------------------------------------------------------------
    int *d_finalOffset=nullptr;
    cudaMalloc(&d_finalOffset, (validCount+1)*sizeof(int));

    {
        // 先把 finalOffset[0]=0, finalOffset[i+1] = newOffset[i]
        int blockSize = 256;
        int gridSize  = (validCount + blockSize - 1)/blockSize;
        buildRealOffset<<<gridSize, blockSize>>>(d_newOffset, d_finalOffset, validCount);
        cudaDeviceSynchronize();
    }
    // 針對 finalOffset[1..validCount] 做 prefix-sum
    if (validCount > 0) {
        prefixSumExclusiveLarge(d_finalOffset+1, validCount);
    }

    //--------------------------------------------------------------------------
    // 7) 複製結果回主機、顯示
    //--------------------------------------------------------------------------
    int *h_newOffset    = (int*) malloc(validCount*sizeof(int));
    int *h_newSid       = (int*) malloc(validCount*sizeof(int));
    int *h_finalOffset  = (int*) malloc((validCount+1)*sizeof(int));

    cudaMemcpy(h_newOffset,   d_newOffset,   validCount*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_newSid,      d_newSid,      validCount*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_finalOffset, d_finalOffset, (validCount+1)*sizeof(int), cudaMemcpyDeviceToHost);

    printf("=== 結果 ===\n");
    printf("N = %d\n", N);
    printArray("newOffset",   h_newOffset,   validCount);
    printArray("newSid",      h_newSid,      validCount);
    printArray("finalOffset", h_finalOffset, validCount+1);

    //--------------------------------------------------------------------------
    // 8) 清理
    //--------------------------------------------------------------------------
    cudaFree(d_offset);
    cudaFree(d_sid);
    cudaFree(d_keep);
    cudaFree(d_keepScan);
    cudaFree(d_newOffset);
    cudaFree(d_newSid);
    cudaFree(d_finalOffset);

    free(h_newOffset);
    free(h_newSid);
    free(h_finalOffset);

    return 0;
}
