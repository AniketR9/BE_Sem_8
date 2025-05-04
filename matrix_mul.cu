#include<iostream>
#include<vector>
#include<chrono>
using namespace std;
using namespace std::chrono;

__global__ void matmul(int *a , int *b, int *c, int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;

    if(row<N && col<N){
        int sum=0;
        for(int i=0;i<N;i++){
            sum+=a[row*N+k]*b[k*N+col];
        }
        c[row*N+col]=sum;
    }
}

void printMat(vector<int> vc, int N){
    for(int i=0;i<N*N;i++){
        cout<<vc[i]<<" ";
        if((i+1)%N==0)cout<<endl;
    }
    cout<<endl;
}

int main(){
    const int n1=3;
    vector<int> a={1,2,3,4,5,6,7,8,9};
    vector<int> b={9,8,7,6,5,4,3,2,1};
    vector<int> c(n1*n1);
    vector<int> c_seq(n1*n1);

    auto begint=high_resolution_clock::now();
    for(int i=0;i<n1;i++){
        for(int j=0;j<n1;j++){
            int sum=0;
            for(int k=0;k<n1;k++){
                sum+=a[i*n1+k]*b[k*n1+j];
            }
            c_seq[i*n1+j]=sum;
        }
    }
    auto endt=high_resolution_clock::now();
    duration<double, milli> timet1=endt-begint;
    cout<<"Mat A:\n ";printMat(a,n1);
    cout<<"Mat B:\n ";printMat(b,n1);
    cout<<"Mat C:\n ";printMat(c,n1);
    cout<<"Time for seq exec:"<<timet1.count()<<" ms.\n";

    int *da,*db,*dc;
    size_t bytes=n1*n1*sizeof(int);

    cudaMalloc(&da,bytes);
    cudaMalloc(&db,bytes);
    cudaMalloc(&dc,bytes);

    cudaMemcpy(da,a.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),bytes,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(n1,n1);
    dim3 blocksPerGrid(1,1);

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmul<<<blocksPerGrid,threadsPerBlock>>>(da,db,dc,n1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(c.data(),dc, bytes, cudaMemcpyDeviceToHost);
    cout<<"Matrix C:\n";printMat(c);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time,start,stop);
    cudaFree(da);cudaFree(db);cudaFree(dc);
    cudaEventDestroy(start);
cudaEventDestroy(stop);

const int N = 512; // 512x512 matrix
    vector<int> matA(N * N, 1);
    vector<int> matB(N * N, 2);
    vector<int> matC(N * N, 0);
    vector<int> matC_seq(N * N, 0);

    // Sequential Matrix Multiplication
    auto start_cpu = chrono::high_resolution_clock::now();
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += matA[row * N + k] * matB[k * N + col];
            matC_seq[row * N + col] = sum;
        }
    }
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_mat_time = end_cpu - start_cpu;
    cout << "[CPU] Matrix Mul Time: " << cpu_mat_time.count() << " ms\n";

    // Parallel Matrix Multiplication
    int *d_matA, *d_matB, *d_matC;
    size_t matrixBytes = N * N * sizeof(int);
    cudaMalloc(&d_matA, matrixBytes);
    cudaMalloc(&d_matB, matrixBytes);
    cudaMalloc(&d_matC, matrixBytes);

    cudaMemcpy(d_matA, matA.data(), matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB.data(), matrixBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_matA, d_matB, d_matC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_mat_time = 0;
    cudaEventElapsedTime(&gpu_mat_time, start, stop);

    cudaMemcpy(matC.data(), d_matC, matrixBytes, cudaMemcpyDeviceToHost);
    cout << "[GPU] Matrix Mul Time: " << gpu_mat_time << " ms\n";

    cudaFree(d_matA); cudaFree(d_matB); cudaFree(d_matC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
return 0;
}
