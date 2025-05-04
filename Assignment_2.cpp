#include<iostream>
#include<omp.h>
#include<chrono>
#include<vector>
using namespace std;
using namespace std::chrono;

void sequentialBubbleSort(vector<int> &arr){
    int n=arr.size();
    bool swapped;
    for(int i=0;i<n;i++){
        swapped=false;
        for(int j=0;j<n-i-1;j++){
            if(arr[j]>arr[j+1]){
                int temp=arr[j];
                arr[j]=arr[j+1];
                arr[j+1]=temp;
                swapped=true;
            }
        }
        if(!swapped) break;
    }
}

void parallelBubbleSort(vector<int> &arr){
    int n=arr.size();
    bool sorted=false;
    while(!sorted){
        bool localSorted=true;
        #pragma omp parallel for reduction(&&:localSorted)
        for(int i=0;i<n-1;i+=2){
            if(arr[i]>arr[i+1]){
                int temp=arr[i];
                arr[i]=arr[i+1];
                arr[i+1]=temp;
                localSorted=false;
            }
        }
        #pragma omp parallel for reduction(&&:localSorted)
        for(int i=1;i<n-1;i+=2){
            if(arr[i]>arr[i+1]){
                int temp=arr[i];
                arr[i]=arr[i+1];
                arr[i+1]=temp;
                localSorted=false;
            }
        }
        sorted=localSorted;
    }
}

void merge(vector<int> &arr,int low, int mid, int high){
    int n1=mid-low+1;
    int n2=high-mid;
    vector<int> arr1(n1);
    vector<int> arr2(n2);
    for(int i=0;i<n1;i++){
        arr1[i]=arr[low+i];
    }
    for(int i=0;i<n2;i++){
        arr2[i]=arr[mid+i];
    }
    int i=0,j=0,k=low;
    while(i<n1 && j<n2){
        if(arr1[i]>arr2[j]){
            arr[k]=arr2[j];
            k++;j++;
        }else{
            arr[k]=arr1[i];
            k++;i++;
        }
    }
    while(i<n1){
        arr[k]=arr1[i];
        i++;k++;
    }
    while(j<n2){
        arr[k]=arr2[j];
        k++;j++;
    }
}

void sequentialMergeSort(vector<int> &arr,int low,int high){
    if(low<high){
        int mid=(low+high)/2;
        sequentialMergeSort(arr,low,mid);
        sequentialMergeSort(arr,mid+1,high);
        merge(arr,low,mid,high);
    }
}

void parallelMergeSort(vector<int>& arr, int low, int high, int depth=0){
    if(low<high){
        int mid=(low+high)/2;
        if(depth<=3){
            #pragma omp parallel sections
            {
                #pragma omp section
                {
                    parallelMergeSort(arr,low,mid,depth+1);
                }
                #pragma omp section
                {
                    parallelMergeSort(arr,mid+1,high,depth+1);
                }
            }
        }else{
            sequentialMergeSort(arr,low,mid);
            sequentialMergeSort(arr,mid+1,high);
        }
        merge(arr,low,mid,high);
    }
}

vector<int> generateRandomArray(int size){
    vector<int>arr(size);
    for(int i=0;i<size;i++){
        arr[i]=rand() %100+1;
    }
    return arr;
}

int main(){
    int size=1000;
    vector<int> arr=generateRandomArray(size);
    vector<int> arr1=arr,arr2=arr,arr3=arr,arr4=arr;

    auto start=high_resolution_clock::now();
    sequentialBubbleSort(arr1);
    auto end=high_resolution_clock::now();
    duration<double> time_seq_bub=end-start;
    cout<<"Time taken for sequential Bubble Sort : "<<time_seq_bub.count()<<" seconds\n";

    start=high_resolution_clock::now();
    parallelBubbleSort(arr2);
    end=high_resolution_clock::now();
    duration<double> time_par_bub=end-start;
    cout<<"Time taken for parallel bubble sort : "<<time_par_bub.count()<<" seconds\n";

    start=high_resolution_clock::now();
    sequentialMergeSort(arr3,0,size-1);
    end=high_resolution_clock::now();
    duration<double> time_seq_mer=end-start;
    cout<<"Time taken for sequential merge sort : "<<time_seq_mer.count()<<" seconds\n";

    start=high_resolution_clock::now();
    parallelMergeSort(arr4,0,size-1);
    end=high_resolution_clock::now();
    duration<double> time_par_mer=end-start;
    cout<<"Time taken for parallel merge sort : "<<time_par_mer.count()<<" seconds\n";

    return 0;
}