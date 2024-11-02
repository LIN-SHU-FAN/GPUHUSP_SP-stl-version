#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <map>
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
};

class GPU_DB {
public:
    int **item,**tid;
    int **iu,**ru;

    int *sequence_len;
    int sid_len;
};

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
    unordered_set<int> ItemSwuUnderThreshold;
    for(pair<int,int>i:DBdata.item_swu){
        if(i.second<threshold){
            ItemSwuUnderThreshold.insert(i.first);
        }
    }

    DB update_DB;
    vector<int> item,tid;
    vector<int> iu,ru;
    int seq_len;

    int sid_len = int(DBdata.sequence_len.size());
    for(int i=0;i<sid_len;i++){
        seq_len=0;
        for(int j=0;j<DBdata.sequence_len[i];j++){
            if(ItemSwuUnderThreshold.find(DBdata.item[i][j])!=ItemSwuUnderThreshold.end()){
                for(int k=0;k<seq_len;k++){
                    ru[k]-=DBdata.iu[i][j];
                }
            }else{
                item.push_back(DBdata.item[i][j]);
                tid.push_back(DBdata.tid[i][j]);
                iu.push_back(DBdata.iu[i][j]);
                ru.push_back(DBdata.ru[i][j]);
                seq_len++;
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
    //cout<<"";
    DBdata = update_DB;

}

void Bulid_GPU_DB(DB &DBdata,GPU_DB &Gpu_Db){
    Gpu_Db.sid_len=int(DBdata.sequence_len.size());
    Gpu_Db.sequence_len = new int[Gpu_Db.sid_len];

    Gpu_Db.item = new int*[Gpu_Db.sid_len];
    Gpu_Db.iu = new int*[Gpu_Db.sid_len];
    Gpu_Db.ru = new int*[Gpu_Db.sid_len];
    Gpu_Db.tid = new int*[Gpu_Db.sid_len];

    for(int i=0;i<Gpu_Db.sid_len;i++){
        Gpu_Db.sequence_len[i] = DBdata.sequence_len[i];


        Gpu_Db.item[i] = new int[Gpu_Db.sequence_len[i]];
        Gpu_Db.iu[i] = new int[Gpu_Db.sequence_len[i]];
        Gpu_Db.ru[i] = new int[Gpu_Db.sequence_len[i]];
        Gpu_Db.tid[i] = new int[Gpu_Db.sequence_len[i]];
        for(int j=0;j<Gpu_Db.sequence_len[i];j++){
            Gpu_Db.item[i][j]=DBdata.item[i][j];
            Gpu_Db.iu[i][j]=DBdata.iu[i][j];
            Gpu_Db.ru[i][j]=DBdata.ru[i][j];
            Gpu_Db.tid[i][j]=DBdata.tid[i][j];

//            cout<<Gpu_Db.item[i][j]<<" ";
//            cout<<Gpu_Db.iu[i][j]<<" ";
//            cout<<Gpu_Db.ru[i][j]<<" ";
//            cout<<Gpu_Db.tid[i][j]<<"\n";
        }

//        cout<<Gpu_Db.sequence_len[i]<<endl;

    }
//    cout<<"";
}

int main() {

    // 指定要讀取的檔案名稱
    string filename = "Yoochoose.txt";
    ifstream file(filename);
    vector<string> lines;

    // 檢查檔案是否成功開啟
    if (!file.is_open()) {
        cerr << "無法開啟檔案: " << filename << endl;
        return 1; // 返回錯誤代碼
    }

    DB DBdata;
    parseData(file,DBdata);

    file.close(); // 關閉檔案

    double threshold = 0.000024 * DBdata.DButility;

    auto start = std::chrono::high_resolution_clock::now();

    SWUpruning(threshold,DBdata);




    GPU_DB Gpu_Db;

    Bulid_GPU_DB(DBdata,Gpu_Db);

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
        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * 1000 * sizeof(int));//開大空間分配
        cudaMemcpy(d_tmp,Gpu_Db.item[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_item_project[i] = d_tmp;

        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * 1000 * sizeof(int));//開大空間分配
        cudaMemcpy(d_tmp,Gpu_Db.iu[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_iu_project[i] = d_tmp;

        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * 1000 * sizeof(int));//開大空間分配
        cudaMemcpy(d_tmp,Gpu_Db.ru[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_ru_project[i] = d_tmp;

        cudaMalloc(&d_tmp, Gpu_Db.sequence_len[i] * 1000 * sizeof(int));//開大空間分配
        cudaMemcpy(d_tmp,Gpu_Db.tid[i],Gpu_Db.sequence_len[i]*sizeof(int),cudaMemcpyHostToDevice);
        h_tid_project[i] = d_tmp;
    }

    cudaMemcpy(d_item_project, h_item_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iu_project, h_iu_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ru_project, h_ru_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tid_project, h_tid_project, Gpu_Db.sid_len * sizeof(int*), cudaMemcpyHostToDevice);


    //kernelfunction操作


    // 獲取 GPU 的內存信息
    status = cudaMemGetInfo(&freeMem, &totalMem);

    if (status == cudaSuccess) {
        cout << "GPU 總內存: " << totalMem / (1024 * 1024) << " MB" << endl;
        cout << "GPU 可用內存: " << freeMem / (1024 * 1024) << " MB" << endl;
    } else {
        cerr << "無法獲取內存信息，錯誤碼: " << cudaGetErrorString(status) << endl;
    }

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

    cudaFree(d_item_project);
    cudaFree(d_iu_project);
    cudaFree(d_ru_project);
    cudaFree(d_tid_project);


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
    auto end = std::chrono::high_resolution_clock::now();


    // 計算持續時間，並轉換為毫秒
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
