#include<iostream>
#include<omp.h>
#include<vector>
#include<chrono>
using namespace std;
using namespace std::chrono;

long long arr_max(vector<long long> arr){
    long long n=arr.size();
    long long maxi=arr[0];
    double start=omp_get_wtime();
    #pragma omp parallel for reduction(max:maxi)
    for(long long i=1;i<n;i++){
        maxi=max(maxi,arr[i]);
    }
    double end=omp_get_wtime();
    cout<<"Max Element : "<<maxi<<endl;
    cout<<"Total Time Taken : "<<end-start<<endl;
    return maxi;
}

long long seq_arr_max(vector<long long> arr){
    long long n=arr.size();
    long long maxi=arr[0];
    double start=omp_get_wtime();
    for(long long i=1;i<n;i++){
        maxi=max(maxi,arr[i]);
    }
    double end=omp_get_wtime();
    cout<<"Max Element : "<<maxi<<endl;
    cout<<"Total Time Taken : "<<end-start<<endl;
    return maxi;
}

long long arr_min(vector<long long> arr){
    long long mini=arr[0];
    long long n=arr.size();
    double start=omp_get_wtime();
    #pragma omp parallel for reduction(min:mini)
    for(long long i=1;i<n;i++){
        mini=min(mini,arr[i]);
    }
    double end=omp_get_wtime();
    cout<<"Min Element : "<<mini<<endl;
    cout<<"Total Time Taken : "<<end-start<<endl;
    return mini;
}

long long seq_arr_min(vector<long long> arr){
    long long n=arr.size();
    long long mini=arr[0];
    double start=omp_get_wtime();
    for(long long i=1;i<n;i++){
        mini=min(mini,arr[i]);
    }
    double end=omp_get_wtime();
    cout<<"Min Element : "<<mini<<endl;
    cout<<"Total Time Taken : "<<end-start<<endl;
    return mini;
}

long long arr_sum(vector<long long> arr){
    long long n=arr.size();
    long long sum=0;
    double start=omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    double end=omp_get_wtime();
    cout<<"Sum of Array Elements : "<<sum;
    cout<<"\nTotal time taken : "<<end-start<<endl;
    return sum;
}

long long seq_arr_sum(vector<long long> arr){
    long long n=arr.size();
    long long sum=0;
    double start=omp_get_wtime();
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    double end=omp_get_wtime();
    cout<<"Sum of Array Elements : "<<sum;
    cout<<"\nTotal time taken : "<<end-start<<endl;
    return sum;
}

double arr_avg(vector<long long> arr){
    long long n=arr.size();
    long long sum=0;
    double start=omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    double avg=sum/n;
    double end=omp_get_wtime();
    cout<<"Average of Array Elements : "<<avg<<endl;
    cout<<"Total time taken : "<<end-start<<endl;
    return avg;
}

double seq_arr_avg(vector<long long> arr){
    long long n=arr.size();
    long long sum=0;
    double start=omp_get_wtime();
    for(int i=0;i<n;i++){
        sum+=arr[i];
    }
    double avg=sum/n;
    double end=omp_get_wtime();
    cout<<"Average of Array Elements : "<<avg<<endl;
    cout<<"Total time taken : "<<end-start<<endl;
    return avg;
}

int main(){
    int size=1000000;
    vector<long long> arr = {1, 2, 4, 5, 7, 2, 6, 0};

    for (long long i = 0; i < size; i++)
    {
        arr.push_back(rand() % 1000);
    }

    arr_max(arr);
    seq_arr_max(arr);
    cout << endl;

    arr_min(arr);
    seq_arr_min(arr);
    cout << endl;

    arr_sum(arr);
    seq_arr_sum(arr);
    cout << endl;

    arr_avg(arr);
    seq_arr_avg(arr);
    cout << endl;

    return 0;
}