#include<iostream>
#include<vector>
#include<chrono>
using namespace std;
using namespace std::chrono;

__global__ void vecAdd(int *a, int *b, int *c, int N){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<N){
        c[tid]=a[tid]+b[tid];
    }
}

void printVec(vector<int> &vc){
    for(auto x:vc){
        cout<<x<<" ";
    }
    cout<<endl;
}

int main(){
    int size=4;
    vector<int> a={1,2,3,4};
    vector<int> b={5,6,7,8};
    vector<int> c(size);
    vector<int> c1(size);

    auto start1=high_resolution_clock::now();
    for(int i=0;i<size;i++){
        c[i]=a[i]+b[i];
    }
    auto end1=high_resolution_clock::now();
    cout<<"Vec A : ";printVec(a);
    cout<<"Vec B : ";printVec(b);
    cout<<"Vec C : ";printVec(c);
    duration<double,milli> timet=end1-start1;
    cout<<"Time for sequential : "<<timet.count()<<" ms."<<endl;

    int *da, *db,*dc;
    size_t bytes=size*sizeof(int);

    cudaMalloc(&da,bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc,bytes);

    cudaMemcpy(da,a.data(),bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),bytes, cudaMemcpyHostToDevice);

    int threads=256;
    int blocks=(size+threads-1)/threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vecAdd<<<blocks,threads>>>(da,db,dc,size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_vec_time;
    cudaEventElapsedTime(&gpu_vec_time, start,stop);
    cudaMemcpy(c1.data(),dc,bytes,cudaMemcpyDeviceToHost);
    cout<<"Vector C1 : ";printVec(c1);
    cout<<"Time taken for parallel : "<<gpu_vec_time<<"ms\n";

    cudaFree(da);cudaFree(db),cudaFree(dc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const int n1=1<<20; //1 million elements;
    vector<int> a1(n1,1);
    vector<int> a2(n1,2);
    vector<int> c2(n1);
    vector<int> c3(n1);

    auto start_time=chrono::high_resolution_clock::now();
    for(int i=0;i<n1;i++){
        c2[i]=a1[i]+a2[i];
    }
    auto end_time=chrono::high_resolution_clock::now();
    chrono::duration<double,milli> cpu_vec_time=end_time-start_time;
    cout<<"Sequential Execution Time Required: "<<cpu_vec_time.count()<<" ms\n";

    // int* da,*db,*dc;
    bytes=n1*sizeof(int);

    cudaMalloc(&da,bytes);
    cudaMalloc(&db,bytes);
    cudaMalloc(&dc, bytes);

    cudaMemcpy(da,a1.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(db,a2.data(),bytes,cudaMemcpyHostToDevice);

    // int threads=256;
    blocks=(n1+threads-1)/threads;
    cudaEvent_t start2,stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    vecAdd<<<blocks,threads>>>(da,db,dc,n1);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float gpu_vec_time1=0;
    cudaEventElapsedTime(&gpu_vec_time1,start2,stop2);
    cudaMemcpy(c3.data(),dc,bytes,cudaMemcpyDeviceToHost);

    cout<<"Parallel Execution Time Required: "<<gpu_vec_time<<" ms\n";
    cudaFree(da);cudaFree(db);cudaFree(dc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}