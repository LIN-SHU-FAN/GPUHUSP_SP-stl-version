#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <stack>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>



const int max_num_threads = 1024;

//2025/02/11紀錄
//可以在一開始就開成攤平陣列減少重複開陣列的時間

//2025/03/04紀錄
//最後可以想一下哪邊可以同時做，有些步驟應該不用循序做



using namespace std;
class DB{
public:
    vector<vector<int>> item,tid;
    vector<vector<int>> iu,ru;
    vector<int> sequence_len;

    vector<int> SWUtility;

    int DButility=0;

    unordered_map<int,int> item_swu;
    unordered_set<int> DB_item_set;

    map<int,map<int,vector<int>>> single_item_chain;//item->sid->vector(實例)
    map<int,map<int,vector<int>>> indices_table;//sid->item->vector(實例)
};

class GPU_DB {
public:
    vector<int> DB_item_set;//index->item
    unordered_map<int,int> DB_item_set_hash;//item->index

    ///DB
    int **item,**tid;
    int **iu,**ru;

    int sid_len;
    int *sequence_len;
    int max_sequence_len;

    ///single item chain
    int ***single_item_chain;//item->sid->instance (item已經做hash所以從0開始對應index) 紀錄single item 在DB上的位置
    int **chain_sid;//長度是c_item_len 寬度是c_sid_len（紀錄single item 存在在哪些sid）

    int c_item_len;
    int *c_sid_len;//長度是c_item_len
    int max_c_sid_len;//c_sid_len最大值
    int **c_seq_len;//長度是c_item_len 寬度是c_sid_len
    //int max_c_seq_len;//
    vector<int> max_c_seq_len;//每個item的最大instance


    //vector<int> max_n;//每個single item的 max(每個投影的sid中sid長度-第一個投影點) ->結果是每個single item的最大值

    ///item indices table(這裡建構在攤平陣列就好)
    map<int,map<int,vector<int>>> indices_table;//sid->item->instance(實例)
//    vector<vector<vector<int>>> indices_table;//sid->item->instance 紀錄sid中的item分別 在DB上的哪些位置
//    vector<vector<int>> table_item;//長度是t_sid_len 寬度是t_item_len（紀錄真正的item）
//
//    int t_sid_len;
//    vector<int> t_item_len;//長度是t_sid_len
//    vector<vector<int>> t_seq_len;//長度是t_sid_len 寬度是t_item_len

};

class Tree_node{//要存pattern、chain、i and s list
public:
    string pattern;

    int d_tree_node_chain_size;
    int *d_tree_node_chain_instance;//存DB上的位置
    int *d_tree_node_chain_utility;//存utility

    int d_tree_node_chain_offset_size;
    int *d_tree_node_chain_offset;//chain_offset
    int *d_tree_node_chain_sid;//真正的sid

    int d_tree_node_i_list_size;
    int *d_tree_node_i_list;

    int d_tree_node_s_list_size;
    int *d_tree_node_s_list;

};


//int test_max_seq=0;

void parseData(ifstream &file,DB &DBdata) {


    regex numberRegex("(\\d+)\\[(\\d+)\\]");

    vector<int> item,tid;
    vector<int> iu,ru;

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int Tid = 1;
        int seq_len=0;
        string token;
        while (iss >> token) {
            if (token == "-1") {
                Tid++;
            } else if (token == "-2") {
                Tid = 1;

                DBdata.sequence_len.push_back(seq_len);
                seq_len=0;
            } else if (token.find("SUtility:") != string::npos) {
                int sUtility = stoi(token.substr(token.find(":") + 1));
                DBdata.SWUtility.push_back(sUtility);
                DBdata.DButility+=sUtility;
                unordered_set<int> uniqueSet(item.begin(),item.end());
                for(int i:uniqueSet){
                    if(DBdata.item_swu.find(i)!=DBdata.item_swu.end()){
                        DBdata.item_swu[i]+=sUtility;
                    }else{
                        DBdata.item_swu[i]=sUtility;
                    }
                }

//                if(item.size()>test_max_seq){
//                    test_max_seq=item.size();
//                }
                DBdata.item.push_back(item);
                DBdata.iu.push_back(iu);
                DBdata.ru.push_back(ru);
                DBdata.tid.push_back(tid);

                item.clear();
                iu.clear();
                ru.clear();
                tid.clear();
            } else {
                smatch match;
                if (regex_match(token, match, numberRegex)) {
                    int firstValue = stoi(match[1]);
                    int secondValue = stod(match[2]);
                    item.push_back(firstValue);
                    iu.push_back(secondValue);
                    ru.push_back(0);
                    tid.push_back(Tid);
                    seq_len++;
                }
            }
        }
    }

    int sid_len = int(DBdata.sequence_len.size());
    int ru_tmp;
    for(int i=0;i<sid_len;i++) {
        ru_tmp =0;
        for (int j = 0; j < DBdata.sequence_len[i]; j++) {
            ru_tmp +=DBdata.iu[i][j];
            DBdata.ru[i][j] =  DBdata.SWUtility[i] - ru_tmp;
        }
    }
}

void SWUpruning(int const minUtility,DB &DBdata){
    DB update_DB;
    vector<int> item,tid;
    vector<int> iu,ru;
    int seq_len;

    int sid_len = int(DBdata.sequence_len.size());
    for(int i=0;i<sid_len;i++){
        seq_len=0;
        for(int j=0;j<DBdata.sequence_len[i];j++){
            if(DBdata.item_swu[DBdata.item[i][j]]<minUtility){
                for(int k=0;k<seq_len;k++){
                    ru[k]-=DBdata.iu[i][j];
                }
            }else{
                item.push_back(DBdata.item[i][j]);
                tid.push_back(DBdata.tid[i][j]);
                iu.push_back(DBdata.iu[i][j]);
                ru.push_back(DBdata.ru[i][j]);
                seq_len++;

                update_DB.DB_item_set.insert(DBdata.item[i][j]);
                update_DB.single_item_chain[DBdata.item[i][j]][update_DB.sequence_len.size()].push_back(item.size()-1);
                update_DB.indices_table[update_DB.sequence_len.size()][DBdata.item[i][j]].push_back(item.size()-1);
            }
        }


        if(!item.empty()){
            update_DB.sequence_len.push_back(seq_len);

            update_DB.item.push_back(item);
            update_DB.iu.push_back(iu);
            update_DB.ru.push_back(ru);
            update_DB.tid.push_back(tid);

            item.clear();
            iu.clear();
            ru.clear();
            tid.clear();
        }

    }
    DBdata = update_DB;
}

void Bulid_GPU_DB(DB &DBdata,GPU_DB &Gpu_Db){
    //將item改成index表示並建立hash對應item->index
    Gpu_Db.DB_item_set.reserve(DBdata.DB_item_set.size());
    copy(DBdata.DB_item_set.begin(), DBdata.DB_item_set.end(),back_inserter(Gpu_Db.DB_item_set));
    sort(Gpu_Db.DB_item_set.begin(), Gpu_Db.DB_item_set.end());

    for (size_t i = 0; i < Gpu_Db.DB_item_set.size(); i++) {
        Gpu_Db.DB_item_set_hash[Gpu_Db.DB_item_set[i]] = static_cast<int>(i); // 值 -> 索引
    }


    ///建DB
    Gpu_Db.sid_len=int(DBdata.sequence_len.size());
    Gpu_Db.sequence_len = new int[Gpu_Db.sid_len];

    Gpu_Db.item = new int*[Gpu_Db.sid_len];
    Gpu_Db.iu = new int*[Gpu_Db.sid_len];
    Gpu_Db.ru = new int*[Gpu_Db.sid_len];
    Gpu_Db.tid = new int*[Gpu_Db.sid_len];

    int max_sequence_len=0;

    for(int i=0;i<Gpu_Db.sid_len;i++){
        Gpu_Db.sequence_len[i] = DBdata.sequence_len[i];

        if(Gpu_Db.sequence_len[i]>max_sequence_len){
            max_sequence_len=Gpu_Db.sequence_len[i];
        }

        Gpu_Db.item[i]=DBdata.item[i].data();
        Gpu_Db.iu[i]=DBdata.iu[i].data();
        Gpu_Db.ru[i]=DBdata.ru[i].data();
        Gpu_Db.tid[i]=DBdata.tid[i].data();


//        Gpu_Db.item[i] = new int[Gpu_Db.sequence_len[i]];
//        Gpu_Db.iu[i] = new int[Gpu_Db.sequence_len[i]];
//        Gpu_Db.ru[i] = new int[Gpu_Db.sequence_len[i]];
//        Gpu_Db.tid[i] = new int[Gpu_Db.sequence_len[i]];



        for(int j=0;j<Gpu_Db.sequence_len[i];j++){
            Gpu_Db.item[i][j]=Gpu_Db.DB_item_set_hash[DBdata.item[i][j]];
//            Gpu_Db.iu[i][j]=DBdata.iu[i][j];
//            Gpu_Db.ru[i][j]=DBdata.ru[i][j];
//            Gpu_Db.tid[i][j]=DBdata.tid[i][j];

//            cout<<Gpu_Db.item[i][j]<<" ";
//            cout<<Gpu_Db.iu[i][j]<<" ";
//            cout<<Gpu_Db.ru[i][j]<<" ";
//            cout<<Gpu_Db.tid[i][j]<<"\n";
        }
//        cout<<"\n";

//        cout<<Gpu_Db.sequence_len[i]<<endl;

    }

    Gpu_Db.max_sequence_len=max_sequence_len;

    //Gpu_Db.DB_item_set=DBdata.DB_item_set;

    ///建single_item_chain
    int item_len=DBdata.single_item_chain.size();

    Gpu_Db.single_item_chain = new int**[item_len];
    Gpu_Db.chain_sid = new int*[item_len]; //每個item出現在哪些sid

    Gpu_Db.c_item_len = item_len;
    Gpu_Db.c_sid_len = new int[item_len]; //每個item的sid長度
    Gpu_Db.c_seq_len = new int*[item_len]; //每個item在不同sid中的seq長度(實例)

    int i_index=0, sid_len, j_index=0,seq_len;
    int max_sid_len=0,max_seq_len=0;
    //int max_n;
    for(auto i=DBdata.single_item_chain.begin();i!=DBdata.single_item_chain.end();i++){//歷遍item
        sid_len = i->second.size();
        Gpu_Db.single_item_chain[i_index] = new int*[sid_len];
        Gpu_Db.chain_sid[i_index] = new int[sid_len];

        Gpu_Db.c_sid_len[i_index] = sid_len;
        if(max_sid_len<sid_len){
            max_sid_len=sid_len;
        }
        Gpu_Db.c_seq_len[i_index] = new int[sid_len];

        //max_n = 0;
        max_seq_len=0;
        j_index = 0;
        for(auto j = i->second.begin();j!=i->second.end();j++){//歷遍sid
            if(max_seq_len < j->second.size()){
                max_seq_len = int(j->second.size());
            }
            seq_len = int(j->second.size());
            Gpu_Db.single_item_chain[i_index][j_index] = j->second.data();
            Gpu_Db.chain_sid[i_index][j_index] = j->first;

            Gpu_Db.c_seq_len[i_index][j_index] = seq_len;

            int tmp=Gpu_Db.sequence_len[j->first]-Gpu_Db.single_item_chain[i_index][j_index][0]-1;
//            if(max_n<tmp){
//                max_n = tmp;
//            }

            j_index++;
        }
        Gpu_Db.max_c_seq_len.push_back(max_seq_len);

        //Gpu_Db.max_n.push_back(max_n);
        i_index++;
    }

    Gpu_Db.max_c_sid_len = max_sid_len;

    ///建indices_table
    Gpu_Db.indices_table=DBdata.indices_table;
//    for(auto i=DBdata.indices_table.begin();i!=DBdata.indices_table.end();i++){//歷遍sid
//        for(auto j = i->second.begin();j!=i->second.end();j++){//歷遍item
//            Gpu_Db.indices_table
//        }
//    }
}


__global__ void PEUcounter(int project_item,
                           int** d_item_project, int** d_iu_project, int** d_ru_project, int** d_tid_project,
                           int* d_sequence_len,int sid_len,
                           int *d_PEU_seq,int *d_Utility_seq) {

    if(threadIdx.x<d_sequence_len[blockIdx.x]){
        if(d_item_project[blockIdx.x][threadIdx.x]==project_item){
            //這裡應該能用redution在加速一點
            atomicMax(&d_PEU_seq[blockIdx.x], d_iu_project[blockIdx.x][threadIdx.x]+d_ru_project[blockIdx.x][threadIdx.x]);
            atomicMax(&d_Utility_seq[blockIdx.x], d_iu_project[blockIdx.x][threadIdx.x]);
        }
    }
}

__global__ void PeuCounter(int project_item,int sid,
                           int** d_item_project, int** d_iu_project, int** d_ru_project, int** d_tid_project,
                           int* d_sequence_len,int sid_len,
                           int *d_PEU_seq,int *d_Utility_seq,int *d_project_point) {
    if(threadIdx.x<d_sequence_len[sid]){
        if(d_item_project[sid][threadIdx.x]==project_item){
            atomicMax(&d_PEU_seq[sid], d_iu_project[sid][threadIdx.x]+d_ru_project[sid][threadIdx.x]);

        }
    }
}

__global__ void Array_add_reduction(int Array_len,int *inputArr,int *outputArr){
    __shared__ int shared_data[max_num_threads];
    int tid = threadIdx.x;
    /*
     * 假設有10個block，每個block有1024個thread
     * blockIdx.x = 0~9
     * blockDim.x = 1024
     * threadIdx.x = 0~1023
     * */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    shared_data[tid] = (idx < Array_len) ? inputArr[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputArr[blockIdx.x] = shared_data[0];
    }
}

__global__ void test(int ***d_single_item_chain,int d_c_item_len,int *d_c_sid_len,int **d_c_seq_len,int **d_chain_sid){
    for(int i=0;i<d_c_item_len;i++){
        printf("item:%d\n",i);
        for(int j=0;j<d_c_sid_len[i];j++){
            printf("sid:%d\n",d_chain_sid[i][j]);
            for(int k=0;k<d_c_seq_len[i][j];k++){
                printf("%d ",d_single_item_chain[i][j][k]);
            }
            printf("\n");
        }
    }
}

__global__ void test1(int *d_item,int *d_tid,int *d_iu,int *d_ru,int *d_offsets,int *d_sequence_len,int d_sid_len){
    //int flat_index = sequence_len_offsets[sid] + pos;
    //flat_item[flat_index];
    for(int i = 0;i<d_sid_len;i++){
        for(int j=d_offsets[i];j<d_offsets[i]+d_sequence_len[i];j++){
            printf("i[%d]j[%d] = %d ",i,j-d_offsets[i],d_item[j]);
        }

        printf("sequence_len:%d\n",d_sequence_len[i]);
    }
    printf("%d\n",d_sid_len);
}



__global__ void test2(int *d_flat_single_item_chain,int *d_chain_offsets_level1,int *d_chain_offsets_level2,
                      int *d_flat_chain_sid,int *d_chain_sid_offsets,
                      int d_c_item_len,
                      int *d_c_sid_len,
                      int *d_flat_c_seq_len,int *d_c_seq_len_offsets
){

    //index = offsets_level2[offsets_level1[item] + sid] + instance 三維陣列
    //這裡的sid不是db中真的sid 要用d_flat_chain_sid解碼才知道db的sid

    //int index = d_c_seq_len_offsets[item] + sid; 二維陣列
    //int value = d_flat_c_seq_len[index];

    for(int i = 0;i<d_c_item_len;i++){
        for(int j=0;j<d_c_sid_len[i];j++){
            for(int k=0;k<d_flat_c_seq_len[d_c_seq_len_offsets[i]+j];k++){
                printf("item[%d]sid[%d]instance[%d] = %d \n",i,d_flat_chain_sid[d_chain_sid_offsets[i]+j],k,d_flat_single_item_chain[d_chain_offsets_level2[d_chain_offsets_level1[i]+j]+k]);
            }
        }
    }
//    for(int i = 0;i<d_sid_len;i++){
//        for(int j=d_offsets[i];j<d_offsets[i]+d_sequence_len[i];j++){
//            printf("i[%d]j[%d] = %d ",i,j-d_offsets[i],d_item[j]);
//        }
//
//        printf("sequence_len:%d\n",d_sequence_len[i]);
//    }
//    printf("%d\n",d_sid_len);
}

///可以用多 Block 分段歸約 會比現在用的Grid-stride loop快 但缺點是要額外開記憶體存中間值
__global__ void count_chain_memory_size(int * __restrict__  d_sequence_len,
                                        int * __restrict__  d_flat_chain_sid,int * __restrict__  d_chain_sid_offsets,
                                        int * __restrict__  d_c_sid_len,
                                        int *  __restrict__ d_flat_single_item_chain,int *  __restrict__ d_chain_offsets_level1,int * __restrict__  d_chain_offsets_level2,
                                        int * __restrict__ d_item_memory_overall_size){

    //extern __shared__ int sub_data[];
    __shared__ int sub_data[max_num_threads];//把資料寫到shared memory且縮小到1024內（如果有超過1024）且順便用梯形公式算好

    //index = d_chain_offsets_level2[d_chain_offsets_level1[item] + sid] + instance 三維陣列

    //int index = d_c_seq_len_offsets[item] + sid; 二維陣列
    //int value = d_flat_c_seq_len[index];

    int sum = 0,first_project,seq_len,n;
    //d_c_sid_len[blockIdx.x]=>每個item的sid數量
    for (int i = threadIdx.x; i < d_c_sid_len[blockIdx.x]; i += blockDim.x) {
        first_project = d_flat_single_item_chain[d_chain_offsets_level2[d_chain_offsets_level1[blockIdx.x] + i] + 0];//blockIdx.x對應item,i對應sid
        seq_len = d_sequence_len[d_flat_chain_sid[d_chain_sid_offsets[blockIdx.x]+i]];
        n = seq_len - first_project - 1;

        n>1 ? n=(n+1)*n/2 : n=n;//梯形公式

//        printf("item=%d sid=%d real sid=%d seq_len=%d first_project=%d n=%d\n",
//               blockIdx.x,i,d_flat_chain_sid[d_chain_sid_offsets[blockIdx.x]+i],seq_len,first_project,n);
        sum += n;
    }

    sub_data[threadIdx.x] = sum;
    //printf("threadIdx.x=%d sum=%d\n",threadIdx.x,sum);

    __syncthreads();

    // 使用平行 reduction 計算陣列的總和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sub_data[threadIdx.x] += sub_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    //printf("blockDim.x=%d threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
    if(threadIdx.x==0){
        //printf("blockIdx.x=%d sub_data[0]=%d\n",blockIdx.x,sub_data[0]);
        d_item_memory_overall_size[blockIdx.x] = sub_data[0];
    }

}

// 這個 Kernel 用於在每個 Block 內做一次平行歸約，
// 將該 Block 內的最大值寫到 blockResults[blockIdx.x].
__global__ void reduceMaxKernel(const int* __restrict__ d_in,
                                int* __restrict__ d_out,
                                int n)
{
    // 每個 Block 負責一段資料 (blockSize * 2) 個元素, 以減少 kernel 呼叫次數

    extern __shared__ int sdata[];  // 動態配置的 Shared memory
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 將 global memory 的資料讀進 shared memory
    int myVal = (globalIdx < n) ? d_in[globalIdx] : INT_MIN;

    // 若仍有空間，再多取一次(做 unrolling)
    int secondIdx = globalIdx + blockDim.x;
    if(secondIdx < n) {
        int val2 = d_in[secondIdx];
        if(val2 > myVal) {
            myVal = val2;
        }
    }
    sdata[tid] = myVal;
    __syncthreads();

    // 在 shared memory 做平行歸約 (reduce to max)
    // 這裡的程式參考了 CUDA SDK 內的 reduction 範例
    // 每次迭代讓活躍的 thread 數減半
    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if(tid < stride) {
            if(sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }
        __syncthreads();
    }

    // 最後 32 個 thread 繼續使用 unrolled warp
    // (此時不再需要 __syncthreads() 因為同一 warp 中可保證同步)
    if(tid < 32) {
        volatile int* v_sdata = sdata;
        // 依序消去 stride=32, 16, 8, 4, 2, 1
        if(tid + 32 < blockDim.x && v_sdata[tid + 32] > v_sdata[tid])  v_sdata[tid] = v_sdata[tid + 32];
        if(tid + 16 < blockDim.x && v_sdata[tid + 16] > v_sdata[tid])  v_sdata[tid] = v_sdata[tid + 16];
        if(tid + 8 < blockDim.x && v_sdata[tid +  8] > v_sdata[tid])  v_sdata[tid] = v_sdata[tid +  8];
        if(tid + 4 < blockDim.x && v_sdata[tid +  4] > v_sdata[tid])  v_sdata[tid] = v_sdata[tid +  4];
        if(tid + 2 < blockDim.x && v_sdata[tid +  2] > v_sdata[tid])  v_sdata[tid] = v_sdata[tid +  2];
        if(tid + 1 < blockDim.x && v_sdata[tid +  1] > v_sdata[tid])  v_sdata[tid] = v_sdata[tid +  1];
    }

    // 用 block 內的第 0 個 thread 將結果寫到 global memory
    if(tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__ void count_chain_offset_size(int *  __restrict__  d_sequence_len,//DB長度
                                        int * __restrict__  d_flat_chain_sid,int * __restrict__  d_chain_sid_offsets,//真正sid
                                        int * __restrict__  d_flat_single_item_chain,int *  __restrict__  d_chain_offsets_level1,int *  __restrict__ d_chain_offsets_level2,//chain資料
                                        int * __restrict__  d_c_sid_len,//每個item的sid投影數量
                                        int * __restrict__  d_item_memory_overall_size,//輸出
                                        int * __restrict__  d_max_n
){
    __shared__ int sub_data[max_num_threads];
    int max_n=INT_MIN,first_project,seq_len,n;
    for (int i = threadIdx.x; i < d_c_sid_len[blockIdx.x]; i += blockDim.x) {
        //blockIdx.x對應item,i對應sid
        first_project = d_flat_single_item_chain[d_chain_offsets_level2[d_chain_offsets_level1[blockIdx.x] + i] + 0];
        seq_len = d_sequence_len[d_flat_chain_sid[d_chain_sid_offsets[blockIdx.x]+i]];
        n = seq_len - first_project - 1;

        max_n = (n>max_n) ? n:max_n ;
    }
    sub_data[threadIdx.x] = max_n;

    __syncthreads();


    // 使用平行 reduction 計算陣列的MAX
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sub_data[threadIdx.x] = max(sub_data[threadIdx.x],sub_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    //printf("blockDim.x=%d threadIdx.x=%d\n",blockIdx.x,threadIdx.x);
    if(threadIdx.x==0){
        d_max_n[blockIdx.x] = sub_data[0];
        d_item_memory_overall_size[blockIdx.x] = sub_data[0] * d_c_sid_len[blockIdx.x];//max_n * item的sid投影點數量
        //printf("blockIdx.x=%d sub_data[0]=%d d_c_sid_len[blockIdx.x]=%d d_item_memory_overall_size[blockIdx.x]=%d\n",blockIdx.x,sub_data[0],d_c_sid_len[blockIdx.x],d_item_memory_overall_size[blockIdx.x]);
    }

}

__device__ int binary_search_in_thread(const int* arr, int size, int key)
{
    int left = 0;
    int right = size - 1;
    while (left <= right) {
        int mid = (left + right) >> 1; // (left + right) / 2
        int mid_val = arr[mid];
        if (mid_val == key) {
            return mid;  // 找到就回傳索引
        } else if (mid_val < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return -1; // 沒找到
}


__global__ void count_single_item_s_candidate(int total_item_num,
                                              int *__restrict__ d_sid_map_item,
                                              int *__restrict__ d_sid_accumulate,
                                              int *__restrict__ d_tid,
                                              int *__restrict__ d_db_offsets,
                                              int *__restrict__ d_flat_chain_sid,int *__restrict__ d_chain_sid_offsets,
                                              int *__restrict__ d_table_item_len,
                                              int *__restrict__ d_flat_indices_table,int *__restrict__ d_table_offsets_level1,int *__restrict__ d_table_offsets_level2,
                                              int *__restrict__ d_flat_table_item,int *__restrict__ d_table_item_offsets,
                                              int *__restrict__ d_single_item_s_candidate,
                                              int *__restrict__ d_chain_sid_num_utility,
                                              int *__restrict__ d_chain_sid_num_peu,
                                              bool *__restrict__ d_TSU_bool,
                                              int *__restrict__ d_iu,
                                              int *__restrict__ d_ru,
                                              int *__restrict__ d_flat_table_seq_len,int *__restrict__ d_table_seq_len_offsets,
                                              int *__restrict__ d_single_item_s_candidate_TSU
){
    //blockIdx.x = 0～single item chain總共有多少sid
    //d_sid_map_item[blockIdx.x] = item
    //blockIdx.x-d_sid_accumulate[blockIdx.x] = sid(還不是真的sid 轉化後才是)
    //threadIdx.x ＝ 0～1024 這個sid 的 table中有幾個item


//    int index = d_chain_sid_offsets[item] + sid;
//    int value = d_flat_chain_sid[index];


    int project_item = d_sid_map_item[blockIdx.x];
    int chain_sid = blockIdx.x-d_sid_accumulate[blockIdx.x];

    int sid = d_flat_chain_sid[d_chain_sid_offsets[project_item] + chain_sid];

    int sid_item_num = d_table_item_len[sid];//這個sid有多少種item

//    index = offsets_level2[offsets_level1[item] + sid] + instance
//    value = d_flat_single_item_chain[index]
    //blockDim.x = 1024
    //item有錯 project_item不能直接+ 要看這個item對應的位置

    // 建議使用 shared memory 暫存
    __shared__ int item_index;// block 內共用

    // 只由 threadIdx.x == 0 計算一次 binary search
    if (threadIdx.x == 0) {
        item_index = binary_search_in_thread(&d_flat_table_item[d_table_item_offsets[sid]],sid_item_num,project_item);
    }
    __syncthreads();

    if(item_index == -1){
        printf("Thread %d encountered an error and is exiting!\n", threadIdx.x);
        return;  // 該 thread 終止
    }

    int item_fist_project_position = d_flat_indices_table[d_table_offsets_level2[d_table_offsets_level1[sid]+item_index]+0];

    int table_item_last_project_position;
    for (int i = threadIdx.x; i < sid_item_num; i += blockDim.x) {
        //i對應到table中sid有多少item
        //table中sid中每個i的最後一個位置
        table_item_last_project_position=d_flat_indices_table[d_table_offsets_level2[d_table_offsets_level1[sid]+i+1]-1];

//        if(sid==28){//看最長那個序列有沒有對
//            printf("sid=%d item=%d item_index=%d item_fist_project_position=%d i=%d table_item_last_project_position=%d s_item=%d\n",sid,project_item,item_index,item_fist_project_position,i,table_item_last_project_position,d_flat_table_item[d_table_item_offsets[sid]+i]);
//        }


        if(item_fist_project_position<table_item_last_project_position){
            //item_fist_project_position的tid比較小＝>是s candidate
            if(d_tid[d_db_offsets[sid]+item_fist_project_position]<d_tid[d_db_offsets[sid]+table_item_last_project_position]){
                //printf("sid=%d item=%d f_tid=%d s_item=%d s_tid=%d\n",sid,project_item,d_tid[d_db_offsets[sid]+item_fist_project_position],d_flat_table_item[d_table_item_offsets[sid]+i],d_tid[d_db_offsets[sid]+table_item_last_project_position]);

                //printf("sid=%d item=%d i=%d s_item=%d item_tid=%d s_candidate=%d sid_item_num=%d\n",sid,project_item,i,)
                //printf("sid=%d item=%d i=%d s_item=%d item_tid=%d s_candidate=%d sid_item_num=%d\n",sid,project_item,i,d_item[d_db_offsets[sid]+i],d_tid[d_db_offsets[sid]+item_fist_project_position],d_tid[d_db_offsets[sid]+table_item_last_project_position],sid_item_num);
                atomicOr(&d_single_item_s_candidate[project_item*total_item_num+d_flat_table_item[d_table_item_offsets[sid]+i]],1);


                ///算TSU
                if(d_TSU_bool[blockIdx.x]){//=true代表要用pattern(這裡是single item)的utility+(第一個可擴展的candidate item的iu+ru)
                    int instance_idx = 0;
                    int while_project_position =d_flat_indices_table[d_table_offsets_level2[d_table_offsets_level1[sid]+i]+instance_idx];
                    //while_project_position代表candidate在DB上sid中的投影點
                    //找到>item_fist_project_position的投影點就是第一個可擴展的candidate item

                    //能進來到這裡代表一定有可以投影的投影點，所以不用設邊界
                    while(while_project_position<=item_fist_project_position){
                        instance_idx++;
                        while_project_position =d_flat_indices_table[d_table_offsets_level2[d_table_offsets_level1[sid]+i]+instance_idx];
                    }
                    atomicAdd(&d_single_item_s_candidate_TSU[project_item*total_item_num+d_flat_table_item[d_table_item_offsets[sid]+i]]
                    ,d_chain_sid_num_utility[blockIdx.x]+d_iu[d_db_offsets[sid]+while_project_position]+d_ru[d_db_offsets[sid]+while_project_position]);

                }else{//=false用peu
                    atomicAdd(&d_single_item_s_candidate_TSU[project_item*total_item_num+d_flat_table_item[d_table_item_offsets[sid]+i]]
                            ,d_chain_sid_num_peu[blockIdx.x]);

                }
            }
        }
    }


}

__global__ void count_single_item_i_candidate(int total_item_num,
                                              int* __restrict__ d_sid_map_item,
                                              int* __restrict__ d_sid_accumulate,
                                              int* __restrict__ d_item,
                                              int* __restrict__ d_tid,
                                              int* __restrict__ d_db_offsets,
                                              int* __restrict__ d_sequence_len,
                                              int* __restrict__ d_c_sid_len,
                                              int * __restrict__ d_flat_c_seq_len,int * __restrict__ d_c_seq_len_offsets,
                                              int *__restrict__ d_flat_single_item_chain,int * __restrict__ d_chain_offsets_level1,int * __restrict__ d_chain_offsets_level2,
                                              int *__restrict__ d_flat_chain_sid,int *__restrict__ d_chain_sid_offsets,
                                              int *__restrict__ d_single_item_i_candidate,
                                              int *__restrict__ d_chain_sid_num_utility,
                                              int *__restrict__ d_chain_sid_num_peu,
                                              bool *__restrict__ d_TSU_bool,
                                              int *__restrict__ d_iu,
                                              int *__restrict__ d_ru,
                                              int *__restrict__ d_single_item_i_candidate_TSU_chain_sid_num
){
    //blockIdx.x = 0～single item chain總共有多少sid
    //threadIdx.x ＝ 0～1024 代表chain中sid上的投影點

    int project_item = d_sid_map_item[blockIdx.x];
    int chain_sid = blockIdx.x-d_sid_accumulate[blockIdx.x];

    int sid = d_flat_chain_sid[d_chain_sid_offsets[project_item] + chain_sid];

    int project_len = d_flat_c_seq_len[d_c_seq_len_offsets[project_item]+chain_sid];

//    if(threadIdx.x == 0){
//        printf("project_item=%d chain_sid=%d project_len=%d\n",project_item,chain_sid,project_len);
//    }
    int project_position,project_position_tid,next_position,next_position_tid;
    for (int i = threadIdx.x; i < project_len; i += blockDim.x) {
        project_position = d_flat_single_item_chain[d_chain_offsets_level2[d_chain_offsets_level1[project_item]+chain_sid]+i];
        project_position_tid=d_tid[d_db_offsets[sid]+project_position];
        next_position = project_position+1;
        while(next_position<d_sequence_len[sid]){
            next_position_tid = d_tid[d_db_offsets[sid]+next_position];
            if(project_position_tid==next_position_tid){
                //printf("sid=%d project_item=%d project_position=%d project_position_tid=%d i_item=%d i_position=%d i_tid=%d\n",sid,project_item,project_position,project_position_tid,d_item[d_db_offsets[sid]+next_position],next_position,next_position_tid);
                atomicOr(&d_single_item_i_candidate[project_item * total_item_num+d_item[d_db_offsets[sid]+next_position]],1);

                ///TSU
                if(d_TSU_bool[blockIdx.x]){
                    //MAX是因為candidate的第一個可擴展投影點iu+ru一定最大
                    atomicMax(&d_single_item_i_candidate_TSU_chain_sid_num[blockIdx.x * total_item_num+d_item[d_db_offsets[sid]+next_position]],
                              d_chain_sid_num_utility[blockIdx.x]+d_iu[d_db_offsets[sid]+next_position]+d_ru[d_db_offsets[sid]+next_position]);
//                    d_single_item_i_candidate_TSU[project_item * total_item_num+d_item[d_db_offsets[sid]+next_position]]
//                    = atomicMax(&d_maxVal, arr[idx]);
                }else{
                    d_single_item_i_candidate_TSU_chain_sid_num[blockIdx.x * total_item_num+d_item[d_db_offsets[sid]+next_position]]
                    = d_chain_sid_num_peu[blockIdx.x];

                }
            }else{
                break;
            }
            next_position++;
        }

    }

}

__global__ void sum_i_candidate_TSU_chain_sid_num_Segments_LargeN(
        const int* __restrict__ d_in,
        const int*   __restrict__ d_offsets,
        int offsetCount,
        int n,
        int* __restrict__ d_out)
{
    // 獲取線性索引 (grid-stride loop)
    //   blockDim.x = 每個 block 有多少 threads
    //   gridDim.x  = block 的數量
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < (offsetCount - 1) * n;
         idx += blockDim.x * gridDim.x)
    {
        int segment = idx / n;     // 第幾個區段
        int col     = idx % n;     // 第幾個欄位

        int startRow = d_offsets[segment];
        int endRow   = d_offsets[segment + 1];  // 不含 endRow

        int sumValue = 0;
        for (int row = startRow; row < endRow; ++row) {
            sumValue += d_in[row * n + col];
        }

        d_out[segment * n + col] = sumValue;
    }
}

__global__ void single_item_TSU_pruning(
        int minUtility,
        int n,
        int* __restrict__ d_single_item_i_candidate,
        int*   __restrict__ d_single_item_i_candidate_TSU,
        int*   __restrict__ d_single_item_s_candidate,
        int*   __restrict__ d_single_item_s_candidate_TSU
        )
{
    int item = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if(d_single_item_i_candidate_TSU[item*n+i]<minUtility){
            d_single_item_i_candidate[item*n+i] = 0;
        }

        if(d_single_item_s_candidate_TSU[item*n+i]<minUtility){
            d_single_item_s_candidate[item*n+i] = 0;
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
// Kernel: reduceSum2Dkernel
// 每個 block 處理一個 row (或說一個陣列)，長度 = n。
// 內部採用 parallel reduction (加總)，含「2倍讀取」與「unrolled warp」等技巧。
// 同時有 boundary check，確保 n 不是2次方時亦能正確。
////////////////////////////////////////////////////////////////////////////////
__global__ void reduceSum2Dkernel(const int* __restrict__ d_data,  // [n*n] 大小
                                  int* __restrict__ d_sums,       // [n] 大小，存放每列的加總
                                  int   n)
{
    extern __shared__ int sdata[];   // 動態配置的 Shared memory，大小=blockDim.x * sizeof(int)
    int tid      = threadIdx.x;
    int row      = blockIdx.x;      // 每個 block 負責一個 row (陣列)
    int baseIdx  = row * n;         // 這個 row 在 d_data 裡的起始位置 (row-major)

    // (1) 對全域資料做 2 倍讀取
    //     globalIdx = baseIdx + tid
    //     secondIdx = baseIdx + tid + blockDim.x
    int globalIdx = baseIdx + tid;
    int myVal = (globalIdx < baseIdx + n) ? d_data[globalIdx] : 0; // 若超出 row 範圍就當 0
    int secondIdx = globalIdx + blockDim.x;
    if (secondIdx < baseIdx + n) {
        myVal += d_data[secondIdx];
    }

    sdata[tid] = myVal;
    __syncthreads();

    // (2) block 內平行歸約 (reduction)
    //     先用 for-loop 收斂到 32 以內，每回合 stride 減半
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // (3) 最後 32 個 threads 使用 unrolled warp 處理
    //     在同一個 warp 中，不需 __syncthreads()，但要加索引檢查
    ///要用volatile才會對
    if (tid < 32) {
        volatile int* v_sdata = sdata;
        if (tid + 32 < blockDim.x) v_sdata[tid] += v_sdata[tid + 32];
        if (tid + 16 < blockDim.x) v_sdata[tid] += v_sdata[tid + 16];
        if (tid +  8 < blockDim.x) v_sdata[tid] += v_sdata[tid +  8];
        if (tid +  4 < blockDim.x) v_sdata[tid] += v_sdata[tid +  4];
        if (tid +  2 < blockDim.x) v_sdata[tid] += v_sdata[tid +  2];
        if (tid +  1 < blockDim.x) v_sdata[tid] += v_sdata[tid +  1];
    }

    // (4) block 內的第 0 號 thread，把結果寫回 d_sums[row]
    if (tid == 0) {
        d_sums[row] = sdata[0];
    }
}
//A和B點對點相乘後結果放到A
__global__ void Arr_Multiplication(int * __restrict__ A,int * __restrict__ B,int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n){
        A[idx] = A[idx] * B[idx];
    }
}


__global__ void single_item_peu_utility_count_max(int total_item_num,
                                                  int * __restrict__ d_sid_map_item,
                                                  int * __restrict__ d_sid_accumulate,
                                                  int * __restrict__ d_iu,
                                                  int * __restrict__ d_ru,
                                                  int * __restrict__ d_db_offsets,
                                                  int * __restrict__ d_flat_single_item_chain,int * __restrict__ d_chain_offsets_level1,int * __restrict__ d_chain_offsets_level2,
                                                  int * __restrict__ d_flat_c_seq_len,int * __restrict__ d_c_seq_len_offsets,
                                                  int * __restrict__ d_flat_chain_sid,int *__restrict__ d_chain_sid_offsets,
                                                  int * __restrict__ d_chain_sid_num_utility,
                                                  int * __restrict__ d_chain_sid_num_peu,
                                                  bool * __restrict__ d_TSU_bool
){

    __shared__ int sub_data_utility[max_num_threads];
    __shared__ int sub_data_peu[max_num_threads];
    int tid = threadIdx.x;

    int project_item = d_sid_map_item[blockIdx.x];
    int chain_sid = blockIdx.x-d_sid_accumulate[blockIdx.x];

    int sid = d_flat_chain_sid[d_chain_sid_offsets[project_item] + chain_sid];

    int project_len = d_flat_c_seq_len[d_c_seq_len_offsets[project_item]+chain_sid];

    int max_utility=INT_MIN,max_peu=INT_MIN;
    int project_position,project_utility,project_peu;

    int first_project_utility;
    if(tid==0){
        project_position = d_flat_single_item_chain[d_chain_offsets_level2[d_chain_offsets_level1[project_item]+chain_sid]+tid];
        project_utility=d_iu[d_db_offsets[sid]+project_position];
        first_project_utility = project_utility;
    }

    for (int i = tid; i < project_len; i += blockDim.x) {
        project_position = d_flat_single_item_chain[d_chain_offsets_level2[d_chain_offsets_level1[project_item]+chain_sid]+i];
        project_utility=d_iu[d_db_offsets[sid]+project_position];
        max_utility = (project_utility>max_utility) ? project_utility:max_utility;

        project_peu = d_iu[d_db_offsets[sid]+project_position]+d_ru[d_db_offsets[sid]+project_position];
        max_peu = (project_peu>max_peu) ? project_peu:max_peu;
    }
    sub_data_utility[tid] = max_utility;

    sub_data_peu[tid] = max_peu;

    __syncthreads();




    // 在 shared memory 做平行歸約 (reduce to max)
    // 這裡的程式參考了 CUDA SDK 內的 reduction 範例
    // 每次迭代讓活躍的 thread 數減半
    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if(tid < stride) {
            if(sub_data_utility[tid + stride] > sub_data_utility[tid]) {
                sub_data_utility[tid] = sub_data_utility[tid + stride];
            }

            if(sub_data_peu[tid + stride] > sub_data_peu[tid]) {
                sub_data_peu[tid] = sub_data_peu[tid + stride];
            }
        }
        __syncthreads();
    }

    // 最後 32 個 thread 繼續使用 unrolled warp
    // (此時不再需要 __syncthreads() 因為同一 warp 中可保證同步)
    if(tid < 32) {
        volatile int* v_sdata_utility = sub_data_utility;
        // 依序消去 stride=32, 16, 8, 4, 2, 1
        if(tid + 32 < blockDim.x && v_sdata_utility[tid + 32] > v_sdata_utility[tid])  v_sdata_utility[tid] = v_sdata_utility[tid + 32];
        if(tid + 16 < blockDim.x && v_sdata_utility[tid + 16] > v_sdata_utility[tid])  v_sdata_utility[tid] = v_sdata_utility[tid + 16];
        if(tid + 8 < blockDim.x && v_sdata_utility[tid +  8] > v_sdata_utility[tid])  v_sdata_utility[tid] = v_sdata_utility[tid +  8];
        if(tid + 4 < blockDim.x && v_sdata_utility[tid +  4] > v_sdata_utility[tid])  v_sdata_utility[tid] = v_sdata_utility[tid +  4];
        if(tid + 2 < blockDim.x && v_sdata_utility[tid +  2] > v_sdata_utility[tid])  v_sdata_utility[tid] = v_sdata_utility[tid +  2];
        if(tid + 1 < blockDim.x && v_sdata_utility[tid +  1] > v_sdata_utility[tid])  v_sdata_utility[tid] = v_sdata_utility[tid +  1];

        volatile int* v_sdata_peu = sub_data_peu;
        // 依序消去 stride=32, 16, 8, 4, 2, 1
        if(tid + 32 < blockDim.x && v_sdata_peu[tid + 32] > v_sdata_peu[tid])  v_sdata_peu[tid] = v_sdata_peu[tid + 32];
        if(tid + 16 < blockDim.x && v_sdata_peu[tid + 16] > v_sdata_peu[tid])  v_sdata_peu[tid] = v_sdata_peu[tid + 16];
        if(tid + 8 < blockDim.x && v_sdata_peu[tid +  8] > v_sdata_peu[tid])  v_sdata_peu[tid] = v_sdata_peu[tid +  8];
        if(tid + 4 < blockDim.x && v_sdata_peu[tid +  4] > v_sdata_peu[tid])  v_sdata_peu[tid] = v_sdata_peu[tid +  4];
        if(tid + 2 < blockDim.x && v_sdata_peu[tid +  2] > v_sdata_peu[tid])  v_sdata_peu[tid] = v_sdata_peu[tid +  2];
        if(tid + 1 < blockDim.x && v_sdata_peu[tid +  1] > v_sdata_peu[tid])  v_sdata_peu[tid] = v_sdata_peu[tid +  1];

    }

    // 用 block 內的第 0 個 thread 將結果寫到 global memory
    if(tid == 0) {
        d_chain_sid_num_utility[blockIdx.x] = sub_data_utility[0];

        d_chain_sid_num_peu[blockIdx.x] = sub_data_peu[0];

        sub_data_utility[0]==first_project_utility?d_TSU_bool[blockIdx.x]=true:d_TSU_bool[blockIdx.x]=false;
    }


}

__global__ void single_item_peu_utility_count(int * __restrict__ d_chain_sid_num_peu,
                                              int * __restrict__ d_chain_sid_num_utility,
                                              int * __restrict__ d_c_seq_len_offsets,
        //d_chain_sid_num_peu剛好可以用d_c_seq_len_offsets算index
                                              int * __restrict__ d_c_sid_len,
                                              int * __restrict__ d_chain_single_item_peu,
                                              int * __restrict__ d_chain_single_item_utility,
                                              int minUtility,
                                              bool * __restrict__ d_chain_single_item_utility_bool,
                                              bool * __restrict__ d_chain_single_item_peu_bool){
    __shared__ int sub_data_utility[max_num_threads];
    __shared__ int sub_data_peu[max_num_threads];

    //int idx = blockIdx.x * blockDim.x+threadIdx.x;
    int tid = threadIdx.x;

    int sid_len = d_c_sid_len[blockIdx.x];

    int sum_utility=0,sum_peu=0;


    for(int i=tid;i<sid_len;i+=blockDim.x){
        sum_utility += d_chain_sid_num_utility[d_c_seq_len_offsets[blockIdx.x]+i];

        sum_peu += d_chain_sid_num_peu[d_c_seq_len_offsets[blockIdx.x]+i];
    }

    sub_data_utility[tid] = sum_utility;

    sub_data_peu[tid] = sum_peu;


    //printf("sid=%d\n",);

    __syncthreads();

    // 在 shared memory 做平行歸約 (reduce sum)
    // 每次迭代讓活躍的 thread 數減半
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sub_data_utility[tid] += sub_data_utility[tid + stride];
            sub_data_peu[tid]     += sub_data_peu[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile int* v_sdata_utility = sub_data_utility;
        // 依序消去 stride = 32, 16, 8, 4, 2, 1
        if (tid + 32 < blockDim.x)
            v_sdata_utility[tid] += v_sdata_utility[tid + 32];
        if (tid + 16 < blockDim.x)
            v_sdata_utility[tid] += v_sdata_utility[tid + 16];
        if (tid + 8 < blockDim.x)
            v_sdata_utility[tid] += v_sdata_utility[tid + 8];
        if (tid + 4 < blockDim.x)
            v_sdata_utility[tid] += v_sdata_utility[tid + 4];
        if (tid + 2 < blockDim.x)
            v_sdata_utility[tid] += v_sdata_utility[tid + 2];
        if (tid + 1 < blockDim.x)
            v_sdata_utility[tid] += v_sdata_utility[tid + 1];

        volatile int* v_sdata_peu = sub_data_peu;
        // 同樣做加法
        if (tid + 32 < blockDim.x)
            v_sdata_peu[tid] += v_sdata_peu[tid + 32];
        if (tid + 16 < blockDim.x)
            v_sdata_peu[tid] += v_sdata_peu[tid + 16];
        if (tid + 8 < blockDim.x)
            v_sdata_peu[tid] += v_sdata_peu[tid + 8];
        if (tid + 4 < blockDim.x)
            v_sdata_peu[tid] += v_sdata_peu[tid + 4];
        if (tid + 2 < blockDim.x)
            v_sdata_peu[tid] += v_sdata_peu[tid + 2];
        if (tid + 1 < blockDim.x)
            v_sdata_peu[tid] += v_sdata_peu[tid + 1];
    }

    // 用 block 內的第 0 個 thread 將結果寫到 global memory
    if (tid == 0) {
        d_chain_single_item_utility[blockIdx.x] = sub_data_utility[0];
        d_chain_single_item_peu[blockIdx.x]  = sub_data_peu[0];

        sub_data_utility[0]>=minUtility?d_chain_single_item_utility_bool[blockIdx.x]=true:d_chain_single_item_utility_bool[blockIdx.x]=false;
        sub_data_peu[0]>=minUtility?d_chain_single_item_peu_bool[blockIdx.x]=true:d_chain_single_item_peu_bool[blockIdx.x]=false;

    }

}
//__global__ void single_item_rus_pruning(int item,
//                                        int * __restrict__ d_chain_sid_num_peu,
//                                        int * __restrict__ d_c_seq_len_offsets,
//                                        //d_chain_sid_num_peu剛好可以用d_c_seq_len_offsets算index
//
//                                        ){
//
//}



void GPUHUSP(const GPU_DB &Gpu_Db,const DB &DB_test,int const minUtility,int &HUSP_num){

    size_t freeMem = 0;
    size_t totalMem = 0;
    //獲取 GPU 的內存信息
    cudaError_t status = cudaMemGetInfo(&freeMem, &totalMem);
    cout <<endl;
    if (status == cudaSuccess) {
        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
    } else {
        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
    }
    cout <<endl;

    //###################
    ///project DB初始
    //###################

    vector<int> flat_item, flat_tid, flat_iu, flat_ru;
    vector<int> db_offsets;
    int offset = 0;

    for (int i = 0; i < Gpu_Db.sid_len; i++) {
        db_offsets.push_back(offset);
        for (int j = 0; j < Gpu_Db.sequence_len[i]; j++) {
            flat_item.push_back(Gpu_Db.item[i][j]);
            flat_tid.push_back(Gpu_Db.tid[i][j]);
            flat_iu.push_back(Gpu_Db.iu[i][j]);
            flat_ru.push_back(Gpu_Db.ru[i][j]);
        }
        offset += Gpu_Db.sequence_len[i];
    }
    db_offsets.push_back(offset);

    int *d_item, *d_tid, *d_iu, *d_ru;
    int *d_db_offsets;//offsets裡面存陣列偏移量 從0開始
    int *d_sequence_len;

    cudaMalloc(&d_item, flat_item.size() * sizeof(int));
    cudaMalloc(&d_tid, flat_tid.size() * sizeof(int));
    cudaMalloc(&d_iu, flat_iu.size() * sizeof(int));
    cudaMalloc(&d_ru, flat_ru.size() * sizeof(int));
    cudaMalloc(&d_db_offsets, db_offsets.size() * sizeof(int));
    cudaMalloc(&d_sequence_len, Gpu_Db.sid_len * sizeof(int));

    cudaMemcpy(d_item, flat_item.data(), flat_item.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tid, flat_tid.data(), flat_tid.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iu, flat_iu.data(), flat_iu.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ru, flat_ru.data(), flat_ru.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_db_offsets, db_offsets.data(), db_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequence_len, Gpu_Db.sequence_len, Gpu_Db.sid_len * sizeof(int), cudaMemcpyHostToDevice);

//    test1<<<1,1>>>(d_item,d_tid,d_iu,d_ru,d_db_offsets,d_sequence_len,Gpu_Db.sid_len);
//    cudaDeviceSynchronize();
//    cout<<"\n";

    /*
     * 假設 single_item_chain 為以下結構：
        Item 0:
          SID 0: [1, 2, 3]
          SID 1: [4, 5]

        Item 1:
          SID 0: [6]
          SID 1: [7, 8, 9, 10]
          SID 2: [11, 12]

        Item 2:
          SID 0: [13, 14, 15, 16, 17]


        offsets_level1=[0, 2, 5, 6]
        對於 Item 0，二維陣列在 flat_chain 的偏移量為 0。
        對於 Item 1，二維陣列在 flat_chain 的偏移量為 2（Item 0 有 2 個 sid）。
        對於 Item 2，二維陣列在 flat_chain 的偏移量為 5（Item 0 和 Item 1 共計 5 個 sid）。

        offsets_level2=[0, 3, 5, 6, 10, 12, 17]
        對於 Item 0, SID 0，偏移量為 0。
        對於 Item 0, SID 1，偏移量為 3。
        對於 Item 1, SID 0，偏移量為 5。
        ...


        offsets_level1：記錄每個 item 在 offsets_level2 中的 "起始" 索引。
        offsets_level2：記錄每個 sid 在 flat_single_item_chain 中的 "起始" 索引。

        三維陣列
        single_item_chain[item][sid][instance]
        等於
        index = offsets_level2[offsets_level1[item] + sid] + instance (table要反向offsets_level1[sid] + item)
        value = d_flat_single_item_chain[index]

        二維陣列
        int index = d_c_seq_len_offsets[row] + col;
        int value = d_flat_c_seq_len[index];
    */

    //###################
    ///single item chain 初始
    //###################

    vector<int> flat_single_item_chain;
    vector<int> chain_offsets_level1(Gpu_Db.c_item_len + 1, 0);// 長度為 c_item_len + 1
    vector<int> chain_offsets_level2;

    vector<int> flat_chain_sid;//真正的sid
    vector<int> chain_sid_offsets;
    int chain_sid_offset=0;

    vector<int> flat_c_seq_len;
    vector<int> c_seq_len_offsets;
    int c_seq_len_offset=0;

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        chain_sid_offsets.push_back(chain_sid_offset);

        c_seq_len_offsets.push_back(c_seq_len_offset);
        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
            // 將偏移量存入 chain_offsets_level2
            chain_offsets_level2.push_back(int(flat_single_item_chain.size()));

            for(int k=0;k<Gpu_Db.c_seq_len[i][j];k++){
                flat_single_item_chain.push_back(Gpu_Db.single_item_chain[i][j][k]);
            }

            flat_chain_sid.push_back(Gpu_Db.chain_sid[i][j]);

            flat_c_seq_len.push_back(Gpu_Db.c_seq_len[i][j]);
        }

        // 計算 offsets_level1_chain
        chain_offsets_level1[i + 1] = int(chain_offsets_level2.size());

        chain_sid_offset+=Gpu_Db.c_sid_len[i];

        c_seq_len_offset+=Gpu_Db.c_sid_len[i];
    }
    chain_sid_offsets.push_back(chain_sid_offset);
    c_seq_len_offsets.push_back(c_seq_len_offset);
    chain_offsets_level2.push_back(int(flat_single_item_chain.size()));//把最後一個位置放入flat_single_item_chain總長度

    int *d_flat_single_item_chain,*d_chain_offsets_level1,*d_chain_offsets_level2;

    cudaMalloc(&d_flat_single_item_chain, flat_single_item_chain.size() * sizeof(int));
    cudaMalloc(&d_chain_offsets_level1, chain_offsets_level1.size() * sizeof(int));
    cudaMalloc(&d_chain_offsets_level2, chain_offsets_level2.size() * sizeof(int));

    cudaMemcpy(d_flat_single_item_chain, flat_single_item_chain.data(), flat_single_item_chain.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chain_offsets_level1, chain_offsets_level1.data(), chain_offsets_level1.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chain_offsets_level2, chain_offsets_level2.data(), chain_offsets_level2.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_flat_chain_sid,*d_chain_sid_offsets;
    cudaMalloc(&d_flat_chain_sid, flat_chain_sid.size() * sizeof(int));
    cudaMalloc(&d_chain_sid_offsets, chain_sid_offsets.size() * sizeof(int));

    cudaMemcpy(d_flat_chain_sid, flat_chain_sid.data(), flat_chain_sid.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chain_sid_offsets, chain_sid_offsets.data(), chain_sid_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_flat_c_seq_len,*d_c_seq_len_offsets;
    cudaMalloc(&d_flat_c_seq_len, flat_c_seq_len.size() * sizeof(int));
    cudaMalloc(&d_c_seq_len_offsets, c_seq_len_offsets.size() * sizeof(int));

    cudaMemcpy(d_flat_c_seq_len, flat_c_seq_len.data(), flat_c_seq_len.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_seq_len_offsets, c_seq_len_offsets.data(), c_seq_len_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_c_sid_len;
    cudaMalloc(&d_c_sid_len, Gpu_Db.c_item_len * sizeof(int));
    cudaMemcpy(d_c_sid_len, Gpu_Db.c_sid_len , Gpu_Db.c_item_len * sizeof(int), cudaMemcpyHostToDevice);


//    test2<<<1,1>>>(d_flat_single_item_chain,d_chain_offsets_level1,d_chain_offsets_level2,
//                   d_flat_chain_sid,d_chain_sid_offsets,
//                   Gpu_Db.c_item_len,
//                   d_c_sid_len,
//                   d_flat_c_seq_len,d_c_seq_len_offsets
//                   );
//    cudaDeviceSynchronize();
//    cout<<"";


    //###################
    ///indices table 初始
    //###################

    //    vector<vector<vector<int>>> indices_table;//sid->item->instance 紀錄sid中的item分別 在DB上的哪些位置
    //    vector<vector<int>> table_item;//長度是t_sid_len 寬度是t_item_len（紀錄真正的item）
    //
    //    int table_sid_len;
    //    vector<int> table_item_len;//長度是table_sid_len
    //    vector<vector<int>> table_seq_len;//長度是table_sid_len 寬度是table_item_len

    vector<int> flat_indices_table;//sid->item->instance 紀錄sid中的item分別 在DB上的哪些位置
    vector<int> table_offsets_level1;// 長度為 sid_len + 1
    table_offsets_level1.push_back(0);
    vector<int> table_offsets_level2;

    vector<int> flat_table_item;//長度是table_sid_len 寬度是t_item_len（紀錄真正的item）
    vector<int> table_item_offsets;
    int table_item_offset=0;

    int table_sid_len=Gpu_Db.sid_len;

    vector<int> table_item_len;//長度是table_sid_len，放每個sid有多少item

    vector<int> flat_table_seq_len;//長度是table_sid_len 寬度是table_item_len，放每個sid中每個item中有多少instance
    vector<int> table_seq_len_offsets;
    int table_seq_len_offset=0;



    int count_table_item_len;

    int max_table_item_len=0;//s擴展要用來算blocksize

    for(auto i=Gpu_Db.indices_table.begin();i!=Gpu_Db.indices_table.end();i++){
        count_table_item_len = 0;

        table_item_offsets.push_back(table_item_offset);

        table_seq_len_offsets.push_back(table_seq_len_offset);

        for(auto j=i->second.begin();j!=i->second.end();j++){
            table_offsets_level2.push_back(int(flat_indices_table.size()));

            flat_table_item.push_back(Gpu_Db.DB_item_set_hash.at(j->first));

            table_item_offset++;

            count_table_item_len++;

            table_seq_len_offset++;


            flat_table_seq_len.push_back(int(j->second.size()));


            for(auto k=j->second.begin();k!=j->second.end();k++){
                flat_indices_table.push_back(*k);
            }

        }
        table_offsets_level1.push_back(int(table_offsets_level2.size()));

        max_table_item_len = count_table_item_len>max_table_item_len?count_table_item_len:max_table_item_len;
        table_item_len.push_back(count_table_item_len);

    }
    table_item_offsets.push_back(table_item_offset);
    table_offsets_level2.push_back(int(flat_indices_table.size()));


    int *d_flat_indices_table,*d_table_offsets_level1,*d_table_offsets_level2;//sid->item->instance 紀錄sid中的item分別 在DB上的哪些位置

    cudaMalloc(&d_flat_indices_table, flat_indices_table.size() * sizeof(int));
    cudaMalloc(&d_table_offsets_level1, table_offsets_level1.size() * sizeof(int));
    cudaMalloc(&d_table_offsets_level2, table_offsets_level2.size() * sizeof(int));

    cudaMemcpy(d_flat_indices_table, flat_indices_table.data(), flat_indices_table.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_offsets_level1, table_offsets_level1.data(), table_offsets_level1.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_offsets_level2, table_offsets_level2.data(), table_offsets_level2.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_flat_table_item,*d_table_item_offsets;//長度是table_sid_len 寬度是t_item_len（紀錄真正的item）

    cudaMalloc(&d_flat_table_item, flat_table_item.size() * sizeof(int));
    cudaMalloc(&d_table_item_offsets, table_item_offsets.size() * sizeof(int));

    cudaMemcpy(d_flat_table_item, flat_table_item.data(), flat_table_item.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_item_offsets, table_item_offsets.data(), table_item_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_table_item_len;//長度是table_sid_len，放每個sid有多少item

    cudaMalloc(&d_table_item_len, table_item_len.size() * sizeof(int));

    cudaMemcpy(d_table_item_len, table_item_len.data(), table_item_len.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_flat_table_seq_len,*d_table_seq_len_offsets;//長度是table_sid_len 寬度是table_item_len，放每個sid中每個item中有多少instance

    cudaMalloc(&d_flat_table_seq_len, flat_table_seq_len.size() * sizeof(int));
    cudaMalloc(&d_table_seq_len_offsets, table_seq_len_offsets.size() * sizeof(int));

    cudaMemcpy(d_flat_table_seq_len, flat_table_seq_len.data(), flat_table_seq_len.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_seq_len_offsets, table_seq_len_offsets.data(), table_seq_len_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);




    //###################
    ///計算chain空間
    ///算DFS所需要的最大空間（用最壞情況估 a,a,a,a...）
    //###################

    int *d_item_memory_overall_size;
    //每個item在每個投影seq中第一個投影點到投影資料庫的末端長度算梯形公式的加總
    //長度是c_item_len

    cudaMalloc(&d_item_memory_overall_size,Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_item_memory_overall_size, 0, Gpu_Db.c_item_len * sizeof(int));

    int max_c_sid_len_num_thread=0;
    if(Gpu_Db.max_c_sid_len>max_num_threads){
        max_c_sid_len_num_thread=max_num_threads;
    }else{
        max_c_sid_len_num_thread=Gpu_Db.max_c_sid_len;
    }

    count_chain_memory_size<<<Gpu_Db.c_item_len,max_c_sid_len_num_thread>>>(d_sequence_len,
                                                                            d_flat_chain_sid,d_chain_sid_offsets,
                                                                            d_c_sid_len,
                                                                            d_flat_single_item_chain,d_chain_offsets_level1,d_chain_offsets_level2,
                                                                            d_item_memory_overall_size);
    cudaDeviceSynchronize();


//    int *h_test = new int[Gpu_Db.c_item_len];
//    cudaMemcpy(h_test, d_item_memory_overall_size, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_test[i]<<" ";
//    }
//    cout<<"\n";

    //找d_item_memory_overall_size的max值
    int *d_blockResults;///接收第一次reduce資料，之後可以重用
    cudaMalloc(&d_blockResults, sizeof(int) * Gpu_Db.c_item_len);

    // ===== 第一次呼叫 =====
    // - blockSize 選擇 <= max_num_threads
    // - 每個 block 負責 2*blockSize 個元素
    // => blocksPerGrid = (n + blockSize*2 - 1) / (blockSize*2)
    int blockSize = (Gpu_Db.c_item_len < max_num_threads) ? Gpu_Db.c_item_len : max_num_threads;
    int blocksPerGrid = (Gpu_Db.c_item_len + blockSize * 2 - 1) / (blockSize * 2);

    // 呼叫 Kernel
    reduceMaxKernel<<<blocksPerGrid, blockSize, blockSize * sizeof(int)>>>(
            d_item_memory_overall_size,
            d_blockResults,
            Gpu_Db.c_item_len
    );
    cudaDeviceSynchronize();

    // 現在 blockResults 裏面有 blocksPerGrid 個 block 的最大值
    // 若 blocksPerGrid > 1，還需要繼續歸約
    int curSize = blocksPerGrid;
    while(curSize > 1) {
        int newBlockSize = (curSize < max_num_threads) ? curSize : max_num_threads;
        int newBlocksPerGrid = (curSize + newBlockSize * 2 - 1) / (newBlockSize * 2);

        reduceMaxKernel<<<newBlocksPerGrid, newBlockSize, newBlockSize * sizeof(int)>>>(
                d_blockResults,  // 輸入放這裡
                d_blockResults,  // 輸出也放這裡 (in-place)
                curSize
        );
        cudaDeviceSynchronize();

        curSize = newBlocksPerGrid;
    }

    // 此時 d_blockResults[0] 就是整個陣列的最大值
    int tree_node_chain_max_memory;
    cudaMemcpy(&tree_node_chain_max_memory, d_blockResults, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "tree_node_chain_max_memory:" << tree_node_chain_max_memory << std::endl;




    //###################
    ///建樹上節點的chain空間
    //###################
    int d_tree_node_chain_global_memory_index=0;//目前用多少空間
    int *d_tree_node_chain_instance_global_memory;//裝資料->投影位置
    cudaMalloc(&d_tree_node_chain_instance_global_memory, tree_node_chain_max_memory * sizeof(int));

    int *d_tree_node_chain_utility_global_memory;//裝Utility
    cudaMalloc(&d_tree_node_chain_utility_global_memory, tree_node_chain_max_memory * sizeof(int));

    //###################
    ///計算offsets和chain_sid空間
    ///max(max_n* item的sid投影點數量)
    //###################
    //max_n
    //重用d_item_memory_overall_size算每個single中最大的offset大小
    //可重用max_c_sid_len_num_thread


    int *d_max_n;//將max_n記起來，之後算i list、s list空間的時候用的到
    cudaMalloc(&d_max_n,Gpu_Db.c_item_len * sizeof(int));

    count_chain_offset_size<<<Gpu_Db.c_item_len,max_c_sid_len_num_thread>>>(d_sequence_len,
                                                                            d_flat_chain_sid,d_chain_sid_offsets,
                                                                            d_flat_single_item_chain,d_chain_offsets_level1,d_chain_offsets_level2,
                                                                            d_c_sid_len,
                                                                            d_item_memory_overall_size,
                                                                            d_max_n
    );
    cudaDeviceSynchronize();

//    int *h_test = new int[Gpu_Db.c_item_len];
//    cudaMemcpy(h_test, d_item_memory_overall_size, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_test[i]<<" ";
//    }
//    cout<<"\n";

    // ===== 第一次呼叫 =====
    // - blockSize 選擇 <= max_num_threads
    // - 每個 block 負責 2*blockSize 個元素
    // => blocksPerGrid = (n + blockSize*2 - 1) / (blockSize*2)
    blockSize = (Gpu_Db.c_item_len < max_num_threads) ? Gpu_Db.c_item_len : max_num_threads;
    blocksPerGrid = (Gpu_Db.c_item_len + blockSize * 2 - 1) / (blockSize * 2);

    // 呼叫 Kernel
    reduceMaxKernel<<<blocksPerGrid, blockSize, blockSize * sizeof(int)>>>(
            d_item_memory_overall_size,
            d_blockResults,
            Gpu_Db.c_item_len
    );
    cudaDeviceSynchronize();

    // 現在 blockResults 裏面有 blocksPerGrid 個 block 的最大值
    // 若 blocksPerGrid > 1，還需要繼續歸約
    curSize = blocksPerGrid;
    while(curSize > 1) {
        int newBlockSize = (curSize < max_num_threads) ? curSize : max_num_threads;
        int newBlocksPerGrid = (curSize + newBlockSize * 2 - 1) / (newBlockSize * 2);

        reduceMaxKernel<<<newBlocksPerGrid, newBlockSize, newBlockSize * sizeof(int)>>>(
                d_blockResults,  // 輸入放這裡
                d_blockResults,  // 輸出也放這裡 (in-place)
                curSize
        );
        cudaDeviceSynchronize();

        curSize = newBlocksPerGrid;
    }

    // 此時 d_blockResults[0] 就是整個陣列的最大值
    int tree_node_chain_offset_max_memory;
    cudaMemcpy(&tree_node_chain_offset_max_memory, d_blockResults, sizeof(int), cudaMemcpyDeviceToHost);
    cout<<"tree_node_chain_offset_max_memory:"<<tree_node_chain_offset_max_memory<<endl;



    //###################
    ///建樹上節點的chain的offset和chain_sid(真正的sid)空間
    //###################
    int d_tree_node_chain_offset_global_memory_index=0;//目前用多少空間
    int *d_tree_node_chain_offset_global_memory;//裝chain的offset
    cudaMalloc(&d_tree_node_chain_offset_global_memory, tree_node_chain_offset_max_memory * sizeof(int));

    int d_tree_node_chain_sid_global_memory_index=0;//目前用多少空間
    int *d_tree_node_chain_sid_global_memory;//裝chain_sid(真正的sid)
    cudaMalloc(&d_tree_node_chain_sid_global_memory, tree_node_chain_offset_max_memory * sizeof(int));





    //###################
    /// 計算single item 的peu、utility
    //###################

    vector<int> sid_map_item;//用來一對一對應sid屬於哪個item
    vector<int> sid_accumulate;//用來知道前面的sid數量
    int sid_num=0;//紀錄每個single item總共投影在多少sid中
    for(int i=0;i<Gpu_Db.c_item_len;i++){
        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
            sid_map_item.push_back(i);
            sid_accumulate.push_back(sid_num);
        }
        sid_num+=Gpu_Db.c_sid_len[i];
    }
    int *d_sid_map_item;
    cudaMalloc(&d_sid_map_item, sid_map_item.size() * sizeof(int));
    cudaMemcpy(d_sid_map_item, sid_map_item.data(), sid_map_item.size() * sizeof(int), cudaMemcpyHostToDevice);

    int *d_sid_accumulate;
    cudaMalloc(&d_sid_accumulate, sid_accumulate.size() * sizeof(int));
    cudaMemcpy(d_sid_accumulate, sid_accumulate.data(), sid_accumulate.size() * sizeof(int), cudaMemcpyHostToDevice);



    int chain_sid_num = chain_offsets_level1.at(chain_offsets_level1.size()-1);//single item chain中總共的sid數量 = sid_num

    int *d_chain_sid_num_utility,*d_chain_sid_num_peu;

    cudaMalloc(&d_chain_sid_num_utility, chain_sid_num * sizeof(int));
    cudaMalloc(&d_chain_sid_num_peu, chain_sid_num * sizeof(int));

    bool *d_TSU_bool;
    cudaMalloc(&d_TSU_bool, chain_sid_num * sizeof(bool));

    auto max_it = std::max_element(Gpu_Db.max_c_seq_len.begin(), Gpu_Db.max_c_seq_len.end());
    int block_size=max_num_threads>*max_it?*max_it:max_num_threads;

    //將chain上每個item中所有seq上投影點找出各自的最大值
    single_item_peu_utility_count_max<<<chain_sid_num,block_size>>>(Gpu_Db.c_item_len,
                                                                    d_sid_map_item,
                                                                    d_sid_accumulate,
                                                                    d_iu,
                                                                    d_ru,
                                                                    d_db_offsets,
                                                                    d_flat_single_item_chain,d_chain_offsets_level1,d_chain_offsets_level2,
                                                                    d_flat_c_seq_len,d_c_seq_len_offsets,
                                                                    d_flat_chain_sid,d_chain_sid_offsets,
                                                                    d_chain_sid_num_utility,
                                                                    d_chain_sid_num_peu,
                                                                    d_TSU_bool
    );
    cudaDeviceSynchronize();

//    int *h_t= new int[chain_sid_num];
//    cudaMemcpy(h_t, d_chain_sid_num_utility, chain_sid_num * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_chain_sid_num_utility:";
//    for(int i=0;i<chain_sid_num;i++){
//        cout<<i<<":"<<h_t[i]<<" ";
//    }
//    cout<<endl;
//
//    cudaMemcpy(h_t, d_chain_sid_num_peu, chain_sid_num * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_chain_sid_num_peu:";
//    for(int i=0;i<chain_sid_num;i++){
//        cout<<i<<":"<<h_t[i]<<" ";
//    }
//    cout<<endl;

    bool *h_rt= new bool[chain_sid_num];
    cudaMemcpy(h_rt, d_TSU_bool, chain_sid_num * sizeof(bool), cudaMemcpyDeviceToHost);

    cout<<"d_TSU_bool:";
    for(int i=0;i<chain_sid_num;i++){
        cout<<h_rt[i]<<" ";
    }
    cout<<endl;

    int *d_chain_single_item_utility,*d_chain_single_item_peu;

    cudaMalloc(&d_chain_single_item_utility, Gpu_Db.c_item_len * sizeof(int));
    cudaMalloc(&d_chain_single_item_peu, Gpu_Db.c_item_len * sizeof(int));

    bool *d_chain_single_item_utility_bool,*d_chain_single_item_peu_bool;

    cudaMalloc(&d_chain_single_item_utility_bool, Gpu_Db.c_item_len * sizeof(bool));
    cudaMemset(d_chain_single_item_utility_bool, 0, Gpu_Db.c_item_len * Gpu_Db.c_item_len * sizeof(bool));

    cudaMalloc(&d_chain_single_item_peu_bool, Gpu_Db.c_item_len * sizeof(bool));
    cudaMemset(d_chain_single_item_peu_bool, 0, Gpu_Db.c_item_len * Gpu_Db.c_item_len * sizeof(bool));


    block_size=max_num_threads>Gpu_Db.max_c_sid_len?Gpu_Db.max_c_sid_len:max_num_threads;

    single_item_peu_utility_count<<<Gpu_Db.c_item_len,block_size>>>(d_chain_sid_num_peu,
                                                                    d_chain_sid_num_utility,
                                                                    d_c_seq_len_offsets,
                                                                    d_c_sid_len,
                                                                    d_chain_single_item_peu,
                                                                    d_chain_single_item_utility,
                                                                    minUtility,
                                                                    d_chain_single_item_utility_bool,
                                                                    d_chain_single_item_peu_bool
    );
    cudaDeviceSynchronize();

    //cout<<Gpu_Db.c_sid_len[18]<<endl;

//    int *h_tt= new int[Gpu_Db.c_item_len];
//    cudaMemcpy(h_tt, d_chain_single_item_utility, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_chain_single_item_utility:";
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<i<<":"<<h_tt[i]<<" ";
//    }
//    cout<<endl;
//
//    cudaMemcpy(h_tt, d_chain_single_item_peu, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_chain_single_item_peu:";
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<i<<":"<<h_tt[i]<<" ";
//    }
//    cout<<endl;


    bool *h_chain_single_item_utility_bool= new bool[Gpu_Db.c_item_len];
    cudaMemcpy(h_chain_single_item_utility_bool, d_chain_single_item_utility_bool, Gpu_Db.c_item_len * sizeof(bool), cudaMemcpyDeviceToHost);

    cout<<"d_chain_single_item_utility_bool:";
    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cout<<i<<":"<<h_chain_single_item_utility_bool[i]<<" ";
    }
    cout<<endl;

    bool *h_chain_single_item_peu_bool= new bool[Gpu_Db.c_item_len];
    cudaMemcpy(h_chain_single_item_peu_bool, d_chain_single_item_peu_bool, Gpu_Db.c_item_len * sizeof(bool), cudaMemcpyDeviceToHost);

    cout<<"d_chain_single_item_peu_bool:";
    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cout<<i<<":"<<h_chain_single_item_peu_bool[i]<<" ";
    }
    cout<<endl;

    cout<<minUtility<<endl;

    //###################
    ///計算I list和S list空間 可順便將single item的candidate用TSU算好
    ///max(每個item的s擴展item數量*max_n)
    //###################
    //除非建single item*single item的空間 不然好像沒法將第一層的candidate記住
    //先做簡單版->i list和s list空間＝single item num*max_n ＝>但這樣就等於 db最長的seq*single item num 所以也不用算


    int *d_single_item_s_candidate,*d_single_item_i_candidate;
    cudaMalloc(&d_single_item_s_candidate, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_single_item_s_candidate, 0, Gpu_Db.c_item_len * Gpu_Db.c_item_len * sizeof(int));

    cudaMalloc(&d_single_item_i_candidate, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_single_item_i_candidate, 0, Gpu_Db.c_item_len * Gpu_Db.c_item_len * sizeof(int));

    int *d_single_item_s_candidate_TSU,*d_single_item_i_candidate_TSU;
    cudaMalloc(&d_single_item_s_candidate_TSU, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_single_item_s_candidate_TSU, 0, Gpu_Db.c_item_len * Gpu_Db.c_item_len * sizeof(int));

    cudaMalloc(&d_single_item_i_candidate_TSU, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_single_item_i_candidate_TSU, 0, Gpu_Db.c_item_len * Gpu_Db.c_item_len * sizeof(int));

    int * d_single_item_i_candidate_TSU_chain_sid_num;//用chain_sid_num*n=>之後在聚合成d_single_item_i_candidate_TSU
    cudaMalloc(&d_single_item_i_candidate_TSU_chain_sid_num, sid_num* Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_single_item_i_candidate_TSU_chain_sid_num, 0, sid_num * Gpu_Db.c_item_len * sizeof(int));

    block_size=max_num_threads>max_table_item_len?max_table_item_len:max_num_threads;

    ///TSU有錯*** => TSU要開sid_num*n的大小才對 之後在聚合
    count_single_item_s_candidate<<<sid_num,block_size>>>(Gpu_Db.c_item_len,
                                                          d_sid_map_item,
                                                          d_sid_accumulate,
                                                          d_tid,
                                                          d_db_offsets,
                                                          d_flat_chain_sid,d_chain_sid_offsets,
                                                          d_table_item_len,
                                                          d_flat_indices_table,d_table_offsets_level1,d_table_offsets_level2,
                                                          d_flat_table_item,d_table_item_offsets,
                                                          d_single_item_s_candidate,
                                                          d_chain_sid_num_utility,
                                                          d_chain_sid_num_peu,
                                                          d_TSU_bool,
                                                          d_iu,
                                                          d_ru,
                                                          d_flat_table_seq_len,d_table_seq_len_offsets,
                                                          d_single_item_s_candidate_TSU
    );
//    cudaDeviceSynchronize();


//    int *h_test = new int[Gpu_Db.c_item_len* Gpu_Db.c_item_len];
//    cudaMemcpy(h_test, d_single_item_s_candidate, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<i<<" : ";
//        for(int j=0;j<Gpu_Db.c_item_len;j++){
//            cout<<h_test[i*Gpu_Db.c_item_len+j]<<" ";
//        }
//        cout<<endl;
//    }
    max_it = std::max_element(Gpu_Db.max_c_seq_len.begin(), Gpu_Db.max_c_seq_len.end());

    block_size=max_num_threads>*max_it?*max_it:max_num_threads;

    count_single_item_i_candidate<<<sid_num,block_size>>>(Gpu_Db.c_item_len,
                                                          d_sid_map_item,
                                                          d_sid_accumulate,
                                                          d_item,
                                                          d_tid,
                                                          d_db_offsets,
                                                          d_sequence_len,
                                                          d_c_sid_len,
                                                          d_flat_c_seq_len,d_c_seq_len_offsets,
                                                          d_flat_single_item_chain,d_chain_offsets_level1,d_chain_offsets_level2,
                                                          d_flat_chain_sid,d_chain_sid_offsets,
                                                          d_single_item_i_candidate,
                                                          d_chain_sid_num_utility,
                                                          d_chain_sid_num_peu,
                                                          d_TSU_bool,
                                                          d_iu,
                                                          d_ru,
                                                          d_single_item_i_candidate_TSU_chain_sid_num
    );
    cudaDeviceSynchronize();

    int *h_test = new int[Gpu_Db.c_item_len* Gpu_Db.c_item_len];
//    cudaMemcpy(h_test, d_single_item_s_candidate, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<i<<" : ";
//        for(int j=0;j<Gpu_Db.c_item_len;j++){
//            cout<<h_test[i*Gpu_Db.c_item_len+j]<<" ";
//        }
//        cout<<endl;
//    }
////
//    cudaMemcpy(h_test, d_single_item_s_candidate_TSU, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<i<<" : ";
//        for(int j=0;j<Gpu_Db.c_item_len;j++){
//            cout<<h_test[i*Gpu_Db.c_item_len+j]<<" ";
//        }
//        cout<<endl;
//    }

    cudaMemcpy(h_test, d_single_item_i_candidate, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cout<<i<<" : ";
        for(int j=0;j<Gpu_Db.c_item_len;j++){
            cout<<h_test[i*Gpu_Db.c_item_len+j]<<" ";
        }
        cout<<endl;
    }
//
//    int *h_testt = new int[sid_num* Gpu_Db.c_item_len];
//    cudaMemcpy(h_testt, d_single_item_i_candidate_TSU_chain_sid_num, sid_num* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<sid_num;i++){
//        cout<<i<<" : ";
//        for(int j=0;j<Gpu_Db.c_item_len;j++){
//            cout<<h_testt[i*Gpu_Db.c_item_len+j]<<" ";
//        }
//        cout<<endl;
//    }

    int gridSize = int(flat_c_seq_len.size());
    blockSize = Gpu_Db.c_item_len > max_num_threads ? max_num_threads : Gpu_Db.c_item_len;
    ///聚合d_single_item_i_candidate_TSU_chain_sid_num
    sum_i_candidate_TSU_chain_sid_num_Segments_LargeN<<<gridSize,blockSize>>>(d_single_item_i_candidate_TSU_chain_sid_num,
                                                            d_c_seq_len_offsets,
                                                            int(flat_c_seq_len.size()),//offsets count
                                                            Gpu_Db.c_item_len,
                                                            d_single_item_i_candidate_TSU
                                                        );
    cudaDeviceSynchronize();

    cudaMemcpy(h_test, d_single_item_i_candidate_TSU, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cout<<i<<" : ";
        for(int j=0;j<Gpu_Db.c_item_len;j++){
            cout<<h_test[i*Gpu_Db.c_item_len+j]<<" ";
        }
        cout<<endl;
    }

    ///將i和s candidate用TSU砍掉沒過門檻的candidate
    blockSize = Gpu_Db.c_item_len > max_num_threads ? max_num_threads : Gpu_Db.c_item_len;
    single_item_TSU_pruning<<<Gpu_Db.c_item_len,blockSize>>>(minUtility,
                                                             Gpu_Db.c_item_len,
                                                             d_single_item_i_candidate,
                                                             d_single_item_i_candidate_TSU,
                                                             d_single_item_s_candidate,
                                                             d_single_item_s_candidate_TSU);
    cudaDeviceSynchronize();

    cudaMemcpy(h_test, d_single_item_i_candidate, Gpu_Db.c_item_len* Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cout<<i<<" : ";
        for(int j=0;j<Gpu_Db.c_item_len;j++){
            cout<<h_test[i*Gpu_Db.c_item_len+j]<<" ";
        }
        cout<<endl;
    }

    ///計算空間大小

    // ---------------------------
    // 啟動 kernel 做 parallel reduction
    // ---------------------------
    // 一個 block 處理一個 row => gridDim = n
    int *d_single_item_s_candidate_sum;
    cudaMalloc(&d_single_item_s_candidate_sum, Gpu_Db.c_item_len*sizeof(int));

    int *d_single_item_i_candidate_sum;
    cudaMalloc(&d_single_item_i_candidate_sum, Gpu_Db.c_item_len*sizeof(int));

    gridSize = Gpu_Db.c_item_len;

    // blockSize 根據 n 來簡單選擇「最小的 1024 與 n 」，也可加更複雜的「最佳化」邏輯
    blockSize = (Gpu_Db.c_item_len < max_num_threads) ? Gpu_Db.c_item_len : max_num_threads;

    // 動態 shared memory 大小
    size_t smemSize = blockSize * sizeof(int);

    // 呼叫 kernel
    reduceSum2Dkernel<<<gridSize, blockSize, smemSize>>>(
            d_single_item_s_candidate, d_single_item_s_candidate_sum, Gpu_Db.c_item_len);

    reduceSum2Dkernel<<<gridSize, blockSize, smemSize>>>(
            d_single_item_i_candidate, d_single_item_i_candidate_sum, Gpu_Db.c_item_len);
    cudaDeviceSynchronize();



//    int *h_sums= new int[Gpu_Db.c_item_len];
//    cudaMemcpy(h_sums, d_single_item_s_candidate_sum, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_single_item_s_candidate_sum:";
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_sums[i]<<" ";
//    }
//    cout<<endl;
//
//    cudaMemcpy(h_sums, d_single_item_i_candidate_sum, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_single_item_i_candidate_sum:";
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_sums[i]<<" ";
//    }
//    cout<<endl;
//
//    cudaMemcpy(h_sums, d_max_n, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    cout<<"d_max_n:";
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_sums[i]<<" ";
//    }
//    cout<<endl;

    blockSize = (Gpu_Db.c_item_len < max_num_threads) ? Gpu_Db.c_item_len : max_num_threads;
    gridSize = (Gpu_Db.c_item_len + (blockSize - 1))/blockSize;
    Arr_Multiplication<<<gridSize,blockSize>>>(d_single_item_s_candidate_sum,d_max_n,Gpu_Db.c_item_len);
    Arr_Multiplication<<<gridSize,blockSize>>>(d_single_item_i_candidate_sum,d_max_n,Gpu_Db.c_item_len);
    cudaDeviceSynchronize();

//    cudaMemcpy(h_sums, d_single_item_s_candidate_sum, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_sums[i]<<" ";
//    }
//    cout<<endl;
//
//    cudaMemcpy(h_sums, d_single_item_i_candidate_sum, Gpu_Db.c_item_len * sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cout<<h_sums[i]<<" ";
//    }
//    cout<<endl;


    // ===== 第一次呼叫 =====
    // - blockSize 選擇 <= max_num_threads
    // - 每個 block 負責 2*blockSize 個元素
    // => blocksPerGrid = (n + blockSize*2 - 1) / (blockSize*2)
    blockSize = (Gpu_Db.c_item_len < max_num_threads) ? Gpu_Db.c_item_len : max_num_threads;
    blocksPerGrid = (Gpu_Db.c_item_len + blockSize * 2 - 1) / (blockSize * 2);

    int *d_s_candidate_blockResults;
    cudaMalloc(&d_s_candidate_blockResults, sizeof(int) * Gpu_Db.c_item_len);

    int *d_i_candidate_blockResults;
    cudaMalloc(&d_i_candidate_blockResults, sizeof(int) * Gpu_Db.c_item_len);

    // 呼叫 Kernel
    reduceMaxKernel<<<blocksPerGrid, blockSize, blockSize * sizeof(int)>>>(
            d_single_item_s_candidate_sum,
            d_s_candidate_blockResults,
            Gpu_Db.c_item_len
    );
    reduceMaxKernel<<<blocksPerGrid, blockSize, blockSize * sizeof(int)>>>(
            d_single_item_i_candidate_sum,
            d_i_candidate_blockResults,
            Gpu_Db.c_item_len
    );
    cudaDeviceSynchronize();

    // 現在 blockResults 裏面有 blocksPerGrid 個 block 的最大值
    // 若 blocksPerGrid > 1，還需要繼續歸約
    curSize = blocksPerGrid;
    while(curSize > 1) {
        int newBlockSize = (curSize < max_num_threads) ? curSize : max_num_threads;
        int newBlocksPerGrid = (curSize + newBlockSize * 2 - 1) / (newBlockSize * 2);

        reduceMaxKernel<<<newBlocksPerGrid, newBlockSize, newBlockSize * sizeof(int)>>>(
                d_s_candidate_blockResults,  // 輸入放這裡
                d_s_candidate_blockResults,  // 輸出也放這裡 (in-place)
                curSize
        );

        reduceMaxKernel<<<newBlocksPerGrid, newBlockSize, newBlockSize * sizeof(int)>>>(
                d_i_candidate_blockResults,  // 輸入放這裡
                d_i_candidate_blockResults,  // 輸出也放這裡 (in-place)
                curSize
        );
        cudaDeviceSynchronize();

        curSize = newBlocksPerGrid;
    }

    // 此時 d_s_candidate_blockResults[0] 就是整個陣列的最大值
    int single_item_s_candidate_max_memory;
    cudaMemcpy(&single_item_s_candidate_max_memory, d_s_candidate_blockResults, sizeof(int), cudaMemcpyDeviceToHost);
    cout<<"single_item_s_candidate_max_memory:"<<single_item_s_candidate_max_memory<<endl;

    // 此時 d_i_candidate_blockResults[0] 就是整個陣列的最大值
    int single_item_i_candidate_max_memory;
    cudaMemcpy(&single_item_i_candidate_max_memory, d_i_candidate_blockResults, sizeof(int), cudaMemcpyDeviceToHost);
    cout<<"single_item_i_candidate_max_memory:"<<single_item_i_candidate_max_memory<<endl;

    //###################
    ///建樹上節點的i s list空間
    //###################
    int d_tree_node_i_list_global_memory_index=0;//目前用多少空間
    int *d_tree_node_i_list_global_memory;//裝chain的offset
    cudaMalloc(&d_tree_node_i_list_global_memory, single_item_i_candidate_max_memory * sizeof(int));

    int d_tree_node_s_list_global_memory_index=0;//目前用多少空間
    int *d_tree_node_s_list_global_memory;//裝chain_sid(真正的sid)
    cudaMalloc(&d_tree_node_s_list_global_memory, single_item_s_candidate_max_memory * sizeof(int));



    //獲取 GPU 的內存信息
    status = cudaMemGetInfo(&freeMem, &totalMem);
    cout <<endl;
    if (status == cudaSuccess) {
        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
    } else {
        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
    }
    cout <<endl;


    ///DFS前將用不到的device空間刪除
    cudaFree(d_item_memory_overall_size);
    cudaFree(d_blockResults);
    cudaFree(d_max_n);
    cudaFree(d_sid_map_item);
    cudaFree(d_sid_accumulate);
    cudaFree(d_chain_sid_num_utility);
    cudaFree(d_chain_sid_num_peu);
    cudaFree(d_TSU_bool);
    cudaFree(d_chain_single_item_utility);
    cudaFree(d_chain_single_item_peu);
    cudaFree(d_single_item_s_candidate_TSU);
    cudaFree(d_single_item_i_candidate_TSU);
    cudaFree(d_single_item_i_candidate_TSU_chain_sid_num);

    cudaFree(d_single_item_s_candidate_sum);
    cudaFree(d_single_item_i_candidate_sum);
    cudaFree(d_s_candidate_blockResults);
    cudaFree(d_i_candidate_blockResults);

    //獲取 GPU 的內存信息
    status = cudaMemGetInfo(&freeMem, &totalMem);
    cout <<endl;
    if (status == cudaSuccess) {
        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
    } else {
        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
    }
    cout <<endl;

    ///建構tree_node到DFS_stack中

    stack<Tree_node*> DFS_stack;
    Tree_node *node;
    for(int single_item=0;single_item<Gpu_Db.c_item_len;single_item++){
        if(h_chain_single_item_utility_bool[single_item]){
            HUSP_num++;
        }
        if(!h_chain_single_item_peu_bool[single_item]){
            continue;
        }

        node = new Tree_node;
        node->pattern = to_string(single_item);

        //建立single item candidate
        


    }





//    int *h_item_memory_overall_size = new int[Gpu_Db.c_item_len];
//    cudaMemcpy(h_item_memory_overall_size, d_item_memory_overall_size, Gpu_Db.c_item_len*sizeof(int), cudaMemcpyDeviceToHost);
//
//    int single_item_max_memory=0;
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        if(single_item_max_memory<h_item_memory_overall_size[i]){
//            single_item_max_memory=h_item_memory_overall_size[i];
//        }
//        //cout<<"item:"<<i<<"="<<h_item_memory_overall_size[i]<<endl;
//    }






//    size_t freeMem = 0;
//    size_t totalMem = 0;
//
//    // 獲取 GPU 的內存信息
//    cudaError_t status = cudaMemGetInfo(&freeMem, &totalMem);
//
//    if (status == cudaSuccess) {
//        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
//        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
//    } else {
//        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
//    }
//
//    //project DB初始
//    int **d_item_project;
//    cudaMalloc(&d_item_project, Gpu_Db.sid_len * sizeof(int*));
//
//    int **d_iu_project;
//    cudaMalloc(&d_iu_project, Gpu_Db.sid_len * sizeof(int*));
//
//    int **d_ru_project;
//    cudaMalloc(&d_ru_project, Gpu_Db.sid_len * sizeof(int*));
//
//    int **d_tid_project;
//    cudaMalloc(&d_tid_project, Gpu_Db.sid_len * sizeof(int*));
//
//    // 主機上的指標陣列，用於存放每一行的 d_tmp 指標
//    //cudaMemcpy(&d_item_project[i], &d_tmp, sizeof(int*), cudaMemcpyDeviceToDevice);不能這樣,因為d_item_project[i]在host不能讀取
//    //要開主機指標的指標（陣列）存Device指標
//    int **h_item_project = new int*[Gpu_Db.sid_len];
//    int **h_iu_project = new int*[Gpu_Db.sid_len];
//    int **h_ru_project = new int*[Gpu_Db.sid_len];
//    int **h_tid_project = new int*[Gpu_Db.sid_len];
//
//
//    for(int i=0;i<Gpu_Db.sid_len;i++){
//        int *d_tmp;
//        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
//        cudaMemcpy(d_tmp,Gpu_Db.item[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
//        h_item_project[i] = d_tmp;
//
//        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
//        cudaMemcpy(d_tmp,Gpu_Db.iu[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
//        h_iu_project[i] = d_tmp;
//
//        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
//        cudaMemcpy(d_tmp,Gpu_Db.ru[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
//        h_ru_project[i] = d_tmp;
//
//        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
//        cudaMemcpy(d_tmp,Gpu_Db.tid[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
//        h_tid_project[i] = d_tmp;
//    }
//
//    cudaMemcpy(d_item_project, h_item_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_iu_project, h_iu_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_ru_project, h_ru_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_tid_project, h_tid_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
//
//
//    int *d_sequence_len;
//    cudaMalloc(&d_sequence_len, Gpu_Db.sid_len * sizeof(int));
//    cudaMemcpy(d_sequence_len,Gpu_Db.sequence_len,Gpu_Db.sid_len * sizeof(int),cudaMemcpyHostToDevice);
//
//
//    //single item chain 初始
//    int ***d_single_item_chain;
//    cudaMalloc((void ***)&d_single_item_chain, Gpu_Db.c_item_len * sizeof(int **));
//
//    int ***h_three_dimension_single_item_chain = new int**[Gpu_Db.c_item_len];//用來存device指標的三維指標
//    int **h_two_dimension_single_item_chain;
//
//    int ***h_delete_tmp = new int**[Gpu_Db.c_item_len];//用來cudafree用的
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        h_two_dimension_single_item_chain = new int*[Gpu_Db.c_sid_len[i]];//用來存device指標的二維指標
//        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
//            int *d_one_dimension_tmp;
//            cudaMalloc(&d_one_dimension_tmp,Gpu_Db.c_seq_len[i][j]*sizeof(int));
//            cudaMemcpy(d_one_dimension_tmp,Gpu_Db.single_item_chain[i][j],Gpu_Db.c_seq_len[i][j]*sizeof(int),cudaMemcpyHostToDevice);
//            h_two_dimension_single_item_chain[j] = d_one_dimension_tmp;
//        }
//
//        int **d_two_dimension_tmp;
//        cudaMalloc(&d_two_dimension_tmp,Gpu_Db.c_sid_len[i]*sizeof(int*));
//        cudaMemcpy(d_two_dimension_tmp,h_two_dimension_single_item_chain,Gpu_Db.c_sid_len[i]*sizeof(int*),cudaMemcpyHostToDevice);
//
//        h_three_dimension_single_item_chain[i] = d_two_dimension_tmp;
//        h_delete_tmp[i]=h_two_dimension_single_item_chain;
//
//        //delete[] h_two_dimension_single_item_chain;
//    }
//    cudaMemcpy(d_single_item_chain,h_three_dimension_single_item_chain,Gpu_Db.c_item_len*sizeof(int**),cudaMemcpyHostToDevice);
//    //delete[] h_three_dimension_single_item_chain;
//
//    int *d_c_sid_len;
//    cudaMalloc(&d_c_sid_len,Gpu_Db.c_item_len*sizeof(int));
//    cudaMemcpy(d_c_sid_len,Gpu_Db.c_sid_len,Gpu_Db.c_item_len*sizeof(int),cudaMemcpyHostToDevice);
//
//    int **d_c_seq_len;
//    cudaMalloc(&d_c_seq_len,Gpu_Db.c_item_len*sizeof(int*));
//    int **h_tmp_c_seq_len = new int*[Gpu_Db.c_item_len];
//
//    int **d_chain_sid;
//    cudaMalloc(&d_chain_sid,Gpu_Db.c_item_len*sizeof(int*));
//    int **h_tmp_chain_sid = new int*[Gpu_Db.c_item_len];
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        int *d_tmp;
//        cudaMalloc(&d_tmp,Gpu_Db.c_sid_len[i]*sizeof(int));
//        cudaMemcpy(d_tmp,Gpu_Db.c_seq_len[i],Gpu_Db.c_sid_len[i]*sizeof(int),cudaMemcpyHostToDevice);
//        h_tmp_c_seq_len[i] = d_tmp;
//
//        cudaMalloc(&d_tmp,Gpu_Db.c_sid_len[i]*sizeof(int));
//        cudaMemcpy(d_tmp,Gpu_Db.chain_sid[i],Gpu_Db.c_sid_len[i]*sizeof(int),cudaMemcpyHostToDevice);
//        h_tmp_chain_sid[i] = d_tmp;
//    }
//
//    cudaMemcpy(d_c_seq_len,h_tmp_c_seq_len,Gpu_Db.c_item_len*sizeof(int*),cudaMemcpyHostToDevice);
//
//    cudaMemcpy(d_chain_sid,h_tmp_chain_sid,Gpu_Db.c_item_len*sizeof(int*),cudaMemcpyHostToDevice);
//
//
//
//    int *d_PEU_Arr;//長度是seq數量 用來算peu(可重用)
//    cudaMalloc(&d_PEU_Arr,Gpu_Db.sid_len*sizeof(int));
//    cudaMemset(d_PEU_Arr, 0, Gpu_Db.sid_len * sizeof(int));
//
//    int *d_I_list_RSU_Arr,*d_S_list_RSU_Arr;//長度是item數量 用來算rsu(可重用)
//    cudaMalloc(&d_I_list_RSU_Arr,Gpu_Db.c_item_len*sizeof(int));
//    cudaMalloc(&d_S_list_RSU_Arr,Gpu_Db.c_item_len*sizeof(int));
//    cudaMemset(d_I_list_RSU_Arr, 0, Gpu_Db.c_item_len * sizeof(int));
//    cudaMemset(d_S_list_RSU_Arr, 0, Gpu_Db.c_item_len * sizeof(int));
//
//
//
//
//    test<<<1,1>>>(d_single_item_chain,Gpu_Db.c_item_len,d_c_sid_len,d_c_seq_len,d_chain_sid);
//
////    for(int i=0;i<Gpu_Db.sid_len;i++){
////        for(int j=0;j<Gpu_Db.sequence_len[i];j++){
////            cout<<Gpu_Db.item[i][j]<<" ";
////        }
////        //cout<<Gpu_Db.sequence_len[i]<<endl;
////        cout<<endl;
////    }
//
//    stack<Tree_node> Main_stack;
//    Tree_node top1_Node;
//
//    top1_Node.pattern = "";
//    top1_Node.ProjectArr_first_position=new int[Gpu_Db.sid_len];
//    top1_Node.ProjectArr_len=new int[Gpu_Db.sid_len];
//
//    //DB clearDBdata;
//    //DBdata = clearDBdata;
//    //cout<<Gpu_Db.max_sequence_len;
//
//
//
//
//
//
//
//
//
//
////    for(int i=0;i<Gpu_Db.sid_len;i++){
////        cout<<Gpu_Db.sequence_len[i]<<endl;
////    }
////    cout<<endl;
//    //kernelfunction操作
//
//    //如果sequence長度有超過1024需要額外處理
//    //int *block_per_sequence = new int[Gpu_Db.sid_len];
////    if(Gpu_Db.max_sequence_len>1024){
////        threadsPerBlock=1024;
////        for(int i=0;i<Gpu_Db.sid_len;i++){
////            blocksPerGrid+= (Gpu_Db.sequence_len[i]+threadsPerBlock-1)/threadsPerBlock;
////            block_per_sequence[i] = (Gpu_Db.sequence_len[i]+threadsPerBlock-1)/threadsPerBlock;
////        }
////    }else{
////        threadsPerBlock=Gpu_Db.max_sequence_len;
////        blocksPerGrid=Gpu_Db.sid_len;
////    }
//
//    int *d_PEU_seq;
//    cudaMalloc(&d_PEU_seq, Gpu_Db.sid_len * sizeof(int));
//    cudaMemset(d_PEU_seq, 0, Gpu_Db.sid_len * sizeof(int));
//
//    int d_PEU_add_len = (Gpu_Db.sid_len + 1024 - 1) / 1024;//看有多少seq在用1024切
//    int *d_PEU_add_result;
//    cudaMalloc(&d_PEU_add_result, d_PEU_add_len * sizeof(int));
//
//    int *d_Utility_seq;
//    cudaMalloc(&d_Utility_seq, Gpu_Db.sid_len * sizeof(int));
//    cudaMemset(d_Utility_seq, 0, Gpu_Db.sid_len * sizeof(int));
//
//
//
//    int threadsPerBlock;
//    int blocksPerGrid;
//
//
//
//    for(int i:Gpu_Db.DB_item_set){
//        threadsPerBlock=Gpu_Db.max_sequence_len;
//        blocksPerGrid=Gpu_Db.sid_len;
////        for(int j=0;j<Gpu_Db.sid_len;j++){
////            PeuCounter<<<1,Gpu_Db.sequence_len[j]>>>(i,j,d_item_project,d_iu_project,d_ru_project,d_tid_project
////                    ,d_sequence_len,Gpu_Db.sid_len,d_PEU_seq,d_Utility_seq,d_project_point);
////        }
//
//        //一次開好比較快
//        PEUcounter<<<blocksPerGrid,threadsPerBlock>>>(i,d_item_project,d_iu_project,d_ru_project,d_tid_project
//                ,d_sequence_len,Gpu_Db.sid_len,d_PEU_seq,d_Utility_seq);
//
////        int *h_PEU_seq=new int[Gpu_Db.sid_len];
////        cudaMemcpy(h_PEU_seq, d_PEU_seq, Gpu_Db.sid_len*sizeof(int), cudaMemcpyDeviceToHost);
////
////        //這裡可以用加總
////        int PEU_count=0;
////        for(int j=0;j<Gpu_Db.sid_len;j++){
////            PEU_count+=h_PEU_seq[j];
////        }
////        cout<<"item:"<<i<<endl;
////        cout<<"PEU:"<<PEU_count<<endl<<endl;
//        //
////        int *h_Utility_seq=new int[Gpu_Db.sid_len];
////        cudaMemcpy(h_Utility_seq, d_Utility_seq, Gpu_Db.sid_len*sizeof(int), cudaMemcpyDeviceToHost);
//
//
//        Array_add_reduction<<<d_PEU_add_len,1024>>>(Gpu_Db.sid_len,d_PEU_seq,d_PEU_add_result);
//        int *h_PEU_add_result=new int[d_PEU_add_len];
//        cudaMemcpy(h_PEU_add_result, d_PEU_add_result, d_PEU_add_len*sizeof(int), cudaMemcpyDeviceToHost);
//
//        int PEU_count = 0;
//        for (int j = 0; j < d_PEU_add_len; j++) {
//            PEU_count += h_PEU_add_result[j];
//        }
////        cout<<"item:"<<i<<endl;
////        cout<<"PEU:"<<PEU_count<<endl;
//
//
//
//
//
////        if(PEU_count<threshold){
////            continue;
////        }
//
//        cudaMemset(d_PEU_seq, 0, Gpu_Db.sid_len * sizeof(int));
//        //cout<<"*/*****\n";
//    }
//
//
//
////    PEUcounter<<<blocksPerGrid,threadsPerBlock>>>(821024,d_item_project,d_iu_project,d_ru_project,d_tid_project,d_sequence_len,Gpu_Db.sid_len,d_PEU_seq);
////
////    int *h_Result=new int[Gpu_Db.sid_len];
////    cudaMemcpy(h_Result, d_PEU_seq, Gpu_Db.sid_len*sizeof(int), cudaMemcpyDeviceToHost);
////
////    for(int j=0;j<Gpu_Db.sid_len;j++){
////        cout<<h_Result[j]<<endl;
////    }
//
//    //project free
//    cudaFree(d_item_project);
//    cudaFree(d_iu_project);
//    cudaFree(d_ru_project);
//    cudaFree(d_tid_project);
//
//    for(int i=0;i<Gpu_Db.sid_len;i++) {
//        cudaFree(h_item_project[i]);
//        cudaFree(h_iu_project[i]);
//        cudaFree(h_ru_project[i]);
//        cudaFree(h_tid_project[i]);
//    }
//
//    delete[] h_item_project;
//    delete[] h_iu_project;
//    delete[] h_ru_project;
//    delete[] h_tid_project;
//    //delete[] h_Result;
//
//    //chain free
//    cudaFree(d_single_item_chain);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cudaFree(h_three_dimension_single_item_chain[i]);
//        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
//            cudaFree(h_delete_tmp[i][j]);
//        }
//    }
//
//    delete[] h_three_dimension_single_item_chain;
//    delete[] h_delete_tmp;
//
//    cudaFree(d_c_seq_len);
//    cudaFree(d_chain_sid);
//
//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        cudaFree(h_tmp_c_seq_len[i]);
//        cudaFree(h_tmp_chain_sid[i]);
//    }
//
//    delete [] h_tmp_c_seq_len;
//    delete [] h_tmp_chain_sid;
//
//
//    cudaFree(d_sequence_len);
//    cudaFree(d_PEU_seq);
//
//    // 獲取 GPU 的內存信息
//    status = cudaMemGetInfo(&freeMem, &totalMem);
//
//    if (status == cudaSuccess) {
//        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
//        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
//    } else {
//        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
//    }
////    // 獲取 GPU 的內存信息
////    status = cudaMemGetInfo(&freeMem, &totalMem);
////
////    if (status == cudaSuccess) {
////        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
////        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
////    } else {
////        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
////    }
//
//
////    for(int i=0;i<Gpu_Db.sid_len;i++) {
////        for (int j = 0; j < Gpu_Db.sequence_len[i]; j++) {
////            cout << Gpu_Db.item[i][j] << " ";
////            cout << Gpu_Db.iu[i][j] << " ";
////            cout << Gpu_Db.ru[i][j] << " ";
////            cout << Gpu_Db.tid[i][j] << "\n";
////        }
////        cout << Gpu_Db.sequence_len[i] << endl;
////    }

}


int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Blocks per Grid: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);


    auto start = std::chrono::high_resolution_clock::now();

    // 指定要讀取的檔案名稱
    string filename = "YoochooseSmaller.txt";
    //string filename = "SIGN.txt";
    //string filename = "Yoochoose.txt";
    ifstream file(filename);
    vector<string> lines;

    // 檢查檔案是否成功開啟
    if (!file.is_open()) {
        cerr << "無法開啟檔案: " << filename << endl;
        return 1; // 返回錯誤代碼
    }


    DB DBdata;
    parseData(file,DBdata);

//    cout<<test_max_seq;
    file.close(); // 關閉檔案

    double threshold = 0.01;
    //double threshold = 0.017;
    //double threshold = 0.00024;

    int minUtility = int(threshold * DBdata.DButility);

    SWUpruning(minUtility,DBdata);


    GPU_DB Gpu_Db;


    Bulid_GPU_DB(DBdata,Gpu_Db);


//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
//            cout<<Gpu_Db.chain_sid[i][j]<<"\n";
//            for(int k=0;k<Gpu_Db.c_seq_len[i][j];k++){
//                cout<<Gpu_Db.single_item_chain[i][j][k]<<" "<<endl;
//            }
//        }
//    }




    int HUSP_num=0;

    GPUHUSP(Gpu_Db,DBdata,minUtility,HUSP_num);

    cout<<"HUSP_num:"<<HUSP_num<<endl;



    auto end = std::chrono::high_resolution_clock::now();
    // 計算持續時間，並轉換為毫秒
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;




    return 0;
}
