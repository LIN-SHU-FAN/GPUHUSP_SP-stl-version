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
};

class GPU_DB {
public:
    vector<int> DB_item_set;//index->item
    unordered_map<int,int> DB_item_set_hash;//item->index

    int **item,**tid;
    int **iu,**ru;

    int sid_len;
    int *sequence_len;
    int max_sequence_len;


    int ***single_item_chain;//item->sid->instance (item已經做hash所以從0開始對應index)
    int **chain_sid;//長度是c_item_len 寬度是c_sid_len

    int c_item_len;
    int *c_sid_len;//長度是c_item_len
    int **c_seq_len;//長度是c_item_len 寬度是c_sid_len

};
class Tree_node{
public:
    string pattern;
    int *ProjectArr_first_position,*ProjectArr_len;
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

void SWUpruning(double threshold,DB &DBdata){
    DB update_DB;
    vector<int> item,tid;
    vector<int> iu,ru;
    int seq_len;

    int sid_len = int(DBdata.sequence_len.size());
    for(int i=0;i<sid_len;i++){
        seq_len=0;
        for(int j=0;j<DBdata.sequence_len[i];j++){
            if(DBdata.item_swu[DBdata.item[i][j]]<threshold){
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


    int item_len=DBdata.single_item_chain.size();

    Gpu_Db.single_item_chain = new int**[item_len];
    Gpu_Db.chain_sid = new int*[item_len]; //每個item出現在哪些sid

    Gpu_Db.c_item_len = item_len;
    Gpu_Db.c_sid_len = new int[item_len]; //每個item的sid長度
    Gpu_Db.c_seq_len = new int*[item_len]; //每個item在不同sid中的seq長度(實例)

    int i_index=0, sid_len, j_index=0,seq_len;
    for(auto i=DBdata.single_item_chain.begin();i!=DBdata.single_item_chain.end();i++){
        sid_len = i->second.size();
        Gpu_Db.single_item_chain[i_index] = new int*[sid_len];
        Gpu_Db.chain_sid[i_index] = new int[sid_len];

        Gpu_Db.c_sid_len[i_index] = sid_len;
        Gpu_Db.c_seq_len[i_index] = new int[sid_len];

        j_index = 0;
        for(auto j = i->second.begin();j!=i->second.end();j++){
            seq_len = j->second.size();
            Gpu_Db.single_item_chain[i_index][j_index] = j->second.data();
            Gpu_Db.chain_sid[i_index][j_index] = j->first;

            Gpu_Db.c_seq_len[i_index][j_index] = seq_len;

            j_index++;
        }
        i_index++;
    }


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
    __shared__ int shared_data[1024];
    int tid = threadIdx.x;
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

void GPUHUSP(const GPU_DB &Gpu_Db){


    size_t freeMem = 0;
    size_t totalMem = 0;

    // 獲取 GPU 的內存信息
    cudaError_t status = cudaMemGetInfo(&freeMem, &totalMem);

    if (status == cudaSuccess) {
        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
    } else {
        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
    }

    //project DB初始
    int **d_item_project;
    cudaMalloc(&d_item_project, Gpu_Db.sid_len * sizeof(int*));

    int **d_iu_project;
    cudaMalloc(&d_iu_project, Gpu_Db.sid_len * sizeof(int*));

    int **d_ru_project;
    cudaMalloc(&d_ru_project, Gpu_Db.sid_len * sizeof(int*));

    int **d_tid_project;
    cudaMalloc(&d_tid_project, Gpu_Db.sid_len * sizeof(int*));

    // 主機上的指標陣列，用於存放每一行的 d_tmp 指標
    //cudaMemcpy(&d_item_project[i], &d_tmp, sizeof(int*), cudaMemcpyDeviceToDevice);不能這樣,因為d_item_project[i]在host不能讀取
    //要開主機指標的指標（陣列）存Device指標
    int **h_item_project = new int*[Gpu_Db.sid_len];
    int **h_iu_project = new int*[Gpu_Db.sid_len];
    int **h_ru_project = new int*[Gpu_Db.sid_len];
    int **h_tid_project = new int*[Gpu_Db.sid_len];


    for(int i=0;i<Gpu_Db.sid_len;i++){
        int *d_tmp;
        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
        cudaMemcpy(d_tmp,Gpu_Db.item[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_item_project[i] = d_tmp;

        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
        cudaMemcpy(d_tmp,Gpu_Db.iu[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_iu_project[i] = d_tmp;

        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
        cudaMemcpy(d_tmp,Gpu_Db.ru[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_ru_project[i] = d_tmp;

        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * sizeof(int));
        cudaMemcpy(d_tmp,Gpu_Db.tid[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_tid_project[i] = d_tmp;
    }

    cudaMemcpy(d_item_project, h_item_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iu_project, h_iu_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ru_project, h_ru_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tid_project, h_tid_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);


    int *d_sequence_len;
    cudaMalloc(&d_sequence_len, Gpu_Db.sid_len * sizeof(int));
    cudaMemcpy(d_sequence_len,Gpu_Db.sequence_len,Gpu_Db.sid_len * sizeof(int),cudaMemcpyHostToDevice);


    //single item chain 初始
    int ***d_single_item_chain;
    cudaMalloc((void ***)&d_single_item_chain, Gpu_Db.c_item_len * sizeof(int **));

    int ***h_three_dimension_single_item_chain = new int**[Gpu_Db.c_item_len];//用來存device指標的三維指標
    int **h_two_dimension_single_item_chain;

    int ***h_delete_tmp = new int**[Gpu_Db.c_item_len];//用來cudafree用的

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        h_two_dimension_single_item_chain = new int*[Gpu_Db.c_sid_len[i]];//用來存device指標的二維指標
        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
            int *d_one_dimension_tmp;
            cudaMalloc(&d_one_dimension_tmp,Gpu_Db.c_seq_len[i][j]*sizeof(int));
            cudaMemcpy(d_one_dimension_tmp,Gpu_Db.single_item_chain[i][j],Gpu_Db.c_seq_len[i][j]*sizeof(int),cudaMemcpyHostToDevice);
            h_two_dimension_single_item_chain[j] = d_one_dimension_tmp;
        }

        int **d_two_dimension_tmp;
        cudaMalloc(&d_two_dimension_tmp,Gpu_Db.c_sid_len[i]*sizeof(int*));
        cudaMemcpy(d_two_dimension_tmp,h_two_dimension_single_item_chain,Gpu_Db.c_sid_len[i]*sizeof(int*),cudaMemcpyHostToDevice);

        h_three_dimension_single_item_chain[i] = d_two_dimension_tmp;
        h_delete_tmp[i]=h_two_dimension_single_item_chain;

        //delete[] h_two_dimension_single_item_chain;
    }
    cudaMemcpy(d_single_item_chain,h_three_dimension_single_item_chain,Gpu_Db.c_item_len*sizeof(int**),cudaMemcpyHostToDevice);
    //delete[] h_three_dimension_single_item_chain;

    int *d_c_sid_len;
    cudaMalloc(&d_c_sid_len,Gpu_Db.c_item_len*sizeof(int));
    cudaMemcpy(d_c_sid_len,Gpu_Db.c_sid_len,Gpu_Db.c_item_len*sizeof(int),cudaMemcpyHostToDevice);

    int **d_c_seq_len;
    cudaMalloc(&d_c_seq_len,Gpu_Db.c_item_len*sizeof(int*));
    int **h_tmp_c_seq_len = new int*[Gpu_Db.c_item_len];

    int **d_chain_sid;
    cudaMalloc(&d_chain_sid,Gpu_Db.c_item_len*sizeof(int*));
    int **h_tmp_chain_sid = new int*[Gpu_Db.c_item_len];

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        int *d_tmp;
        cudaMalloc(&d_tmp,Gpu_Db.c_sid_len[i]*sizeof(int));
        cudaMemcpy(d_tmp,Gpu_Db.c_seq_len[i],Gpu_Db.c_sid_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_tmp_c_seq_len[i] = d_tmp;

        cudaMalloc(&d_tmp,Gpu_Db.c_sid_len[i]*sizeof(int));
        cudaMemcpy(d_tmp,Gpu_Db.chain_sid[i],Gpu_Db.c_sid_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_tmp_chain_sid[i] = d_tmp;
    }

    cudaMemcpy(d_c_seq_len,h_tmp_c_seq_len,Gpu_Db.c_item_len*sizeof(int*),cudaMemcpyHostToDevice);

    cudaMemcpy(d_chain_sid,h_tmp_chain_sid,Gpu_Db.c_item_len*sizeof(int*),cudaMemcpyHostToDevice);



    int *d_PEU_Arr;//長度是seq數量 用來算peu(可重用)
    cudaMalloc(&d_PEU_Arr,Gpu_Db.sid_len*sizeof(int));
    cudaMemset(d_PEU_Arr, 0, Gpu_Db.sid_len * sizeof(int));

    int *d_I_list_RSU_Arr,*d_S_list_RSU_Arr;//長度是item數量 用來算rsu(可重用)
    cudaMalloc(&d_I_list_RSU_Arr,Gpu_Db.c_item_len*sizeof(int));
    cudaMalloc(&d_S_list_RSU_Arr,Gpu_Db.c_item_len*sizeof(int));
    cudaMemset(d_I_list_RSU_Arr, 0, Gpu_Db.c_item_len * sizeof(int));
    cudaMemset(d_S_list_RSU_Arr, 0, Gpu_Db.c_item_len * sizeof(int));




    test<<<1,1>>>(d_single_item_chain,Gpu_Db.c_item_len,d_c_sid_len,d_c_seq_len,d_chain_sid);

//    for(int i=0;i<Gpu_Db.sid_len;i++){
//        for(int j=0;j<Gpu_Db.sequence_len[i];j++){
//            cout<<Gpu_Db.item[i][j]<<" ";
//        }
//        //cout<<Gpu_Db.sequence_len[i]<<endl;
//        cout<<endl;
//    }

    stack<Tree_node> Main_stack;
    Tree_node top1_Node;

    top1_Node.pattern = "";
    top1_Node.ProjectArr_first_position=new int[Gpu_Db.sid_len];
    top1_Node.ProjectArr_len=new int[Gpu_Db.sid_len];

    //DB clearDBdata;
    //DBdata = clearDBdata;
    //cout<<Gpu_Db.max_sequence_len;










//    for(int i=0;i<Gpu_Db.sid_len;i++){
//        cout<<Gpu_Db.sequence_len[i]<<endl;
//    }
//    cout<<endl;
    //kernelfunction操作

    //如果sequence長度有超過1024需要額外處理
    //int *block_per_sequence = new int[Gpu_Db.sid_len];
//    if(Gpu_Db.max_sequence_len>1024){
//        threadsPerBlock=1024;
//        for(int i=0;i<Gpu_Db.sid_len;i++){
//            blocksPerGrid+= (Gpu_Db.sequence_len[i]+threadsPerBlock-1)/threadsPerBlock;
//            block_per_sequence[i] = (Gpu_Db.sequence_len[i]+threadsPerBlock-1)/threadsPerBlock;
//        }
//    }else{
//        threadsPerBlock=Gpu_Db.max_sequence_len;
//        blocksPerGrid=Gpu_Db.sid_len;
//    }

    int *d_PEU_seq;
    cudaMalloc(&d_PEU_seq, Gpu_Db.sid_len * sizeof(int));
    cudaMemset(d_PEU_seq, 0, Gpu_Db.sid_len * sizeof(int));

    int d_PEU_add_len = (Gpu_Db.sid_len + 1024 - 1) / 1024;//看有多少seq在用1024切
    int *d_PEU_add_result;
    cudaMalloc(&d_PEU_add_result, d_PEU_add_len * sizeof(int));

    int *d_Utility_seq;
    cudaMalloc(&d_Utility_seq, Gpu_Db.sid_len * sizeof(int));
    cudaMemset(d_Utility_seq, 0, Gpu_Db.sid_len * sizeof(int));



    int threadsPerBlock;
    int blocksPerGrid;



    for(int i:Gpu_Db.DB_item_set){
        threadsPerBlock=Gpu_Db.max_sequence_len;
        blocksPerGrid=Gpu_Db.sid_len;
//        for(int j=0;j<Gpu_Db.sid_len;j++){
//            PeuCounter<<<1,Gpu_Db.sequence_len[j]>>>(i,j,d_item_project,d_iu_project,d_ru_project,d_tid_project
//                    ,d_sequence_len,Gpu_Db.sid_len,d_PEU_seq,d_Utility_seq,d_project_point);
//        }

        //一次開好比較快
        PEUcounter<<<blocksPerGrid,threadsPerBlock>>>(i,d_item_project,d_iu_project,d_ru_project,d_tid_project
                ,d_sequence_len,Gpu_Db.sid_len,d_PEU_seq,d_Utility_seq);

//        int *h_PEU_seq=new int[Gpu_Db.sid_len];
//        cudaMemcpy(h_PEU_seq, d_PEU_seq, Gpu_Db.sid_len*sizeof(int), cudaMemcpyDeviceToHost);
//
//        //這裡可以用加總
//        int PEU_count=0;
//        for(int j=0;j<Gpu_Db.sid_len;j++){
//            PEU_count+=h_PEU_seq[j];
//        }
//        cout<<"item:"<<i<<endl;
//        cout<<"PEU:"<<PEU_count<<endl<<endl;
        //
//        int *h_Utility_seq=new int[Gpu_Db.sid_len];
//        cudaMemcpy(h_Utility_seq, d_Utility_seq, Gpu_Db.sid_len*sizeof(int), cudaMemcpyDeviceToHost);


        Array_add_reduction<<<d_PEU_add_len,1024>>>(Gpu_Db.sid_len,d_PEU_seq,d_PEU_add_result);
        int *h_PEU_add_result=new int[d_PEU_add_len];
        cudaMemcpy(h_PEU_add_result, d_PEU_add_result, d_PEU_add_len*sizeof(int), cudaMemcpyDeviceToHost);

        int PEU_count = 0;
        for (int j = 0; j < d_PEU_add_len; j++) {
            PEU_count += h_PEU_add_result[j];
        }
//        cout<<"item:"<<i<<endl;
//        cout<<"PEU:"<<PEU_count<<endl;





//        if(PEU_count<threshold){
//            continue;
//        }

        cudaMemset(d_PEU_seq, 0, Gpu_Db.sid_len * sizeof(int));
        //cout<<"*/*****\n";
    }



//    PEUcounter<<<blocksPerGrid,threadsPerBlock>>>(821024,d_item_project,d_iu_project,d_ru_project,d_tid_project,d_sequence_len,Gpu_Db.sid_len,d_PEU_seq);
//
//    int *h_Result=new int[Gpu_Db.sid_len];
//    cudaMemcpy(h_Result, d_PEU_seq, Gpu_Db.sid_len*sizeof(int), cudaMemcpyDeviceToHost);
//
//    for(int j=0;j<Gpu_Db.sid_len;j++){
//        cout<<h_Result[j]<<endl;
//    }

    //project free
    cudaFree(d_item_project);
    cudaFree(d_iu_project);
    cudaFree(d_ru_project);
    cudaFree(d_tid_project);

    for(int i=0;i<Gpu_Db.sid_len;i++) {
        cudaFree(h_item_project[i]);
        cudaFree(h_iu_project[i]);
        cudaFree(h_ru_project[i]);
        cudaFree(h_tid_project[i]);
    }

    delete[] h_item_project;
    delete[] h_iu_project;
    delete[] h_ru_project;
    delete[] h_tid_project;
    //delete[] h_Result;

    //chain free
    cudaFree(d_single_item_chain);

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cudaFree(h_three_dimension_single_item_chain[i]);
        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
            cudaFree(h_delete_tmp[i][j]);
        }
    }

    delete[] h_three_dimension_single_item_chain;
    delete[] h_delete_tmp;

    cudaFree(d_c_seq_len);
    cudaFree(d_chain_sid);

    for(int i=0;i<Gpu_Db.c_item_len;i++){
        cudaFree(h_tmp_c_seq_len[i]);
        cudaFree(h_tmp_chain_sid[i]);
    }

    delete [] h_tmp_c_seq_len;
    delete [] h_tmp_chain_sid;


    cudaFree(d_sequence_len);
    cudaFree(d_PEU_seq);

    // 獲取 GPU 的內存信息
    status = cudaMemGetInfo(&freeMem, &totalMem);

    if (status == cudaSuccess) {
        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
    } else {
        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
    }
//    // 獲取 GPU 的內存信息
//    status = cudaMemGetInfo(&freeMem, &totalMem);
//
//    if (status == cudaSuccess) {
//        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
//        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
//    } else {
//        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
//    }


//    for(int i=0;i<Gpu_Db.sid_len;i++) {
//        for (int j = 0; j < Gpu_Db.sequence_len[i]; j++) {
//            cout << Gpu_Db.item[i][j] << " ";
//            cout << Gpu_Db.iu[i][j] << " ";
//            cout << Gpu_Db.ru[i][j] << " ";
//            cout << Gpu_Db.tid[i][j] << "\n";
//        }
//        cout << Gpu_Db.sequence_len[i] << endl;
//    }

}


int main() {
    // 指定要讀取的檔案名稱
    string filename = "YoochooseSamller.txt";
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

    double threshold = 0.01 * DBdata.DButility;

    SWUpruning(threshold,DBdata);

    GPU_DB Gpu_Db;

    auto start = std::chrono::high_resolution_clock::now();
    Bulid_GPU_DB(DBdata,Gpu_Db);
    auto end = std::chrono::high_resolution_clock::now();
    // 計算持續時間，並轉換為毫秒
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

//    for(int i=0;i<Gpu_Db.c_item_len;i++){
//        for(int j=0;j<Gpu_Db.c_sid_len[i];j++){
//            cout<<Gpu_Db.chain_sid[i][j]<<"\n";
//            for(int k=0;k<Gpu_Db.c_seq_len[i][j];k++){
//                cout<<Gpu_Db.single_item_chain[i][j][k]<<" "<<endl;
//            }
//        }
//    }



    GPUHUSP(Gpu_Db);

    return 0;
}
