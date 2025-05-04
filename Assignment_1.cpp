#include<iostream>
#include<vector>
#include<queue>
#include<chrono>
#include<omp.h>
using namespace std;
using namespace std::chrono;

class Graph{
    int V;
    vector<vector<int>> adj;

    public:
    Graph(int V){
        this->V=V;
        adj.resize(V);
    }

    void addEdge(int u, int v){
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void sequentialBFS(int source){
        vector<bool> visited(V,false);
        queue<int> q;
        visited[source]=true;
        q.push(source);

        while(!q.empty()){
            int u=q.front();
            q.pop();
            cout<<u<<" ";

            for(int v:adj[u]){
                if(!visited[v]){
                    visited[v]=true;
                    q.push(v);
                }
            }
        }
    }

    void sequentialDFS(int source){
        vector<bool> visited(V,false);
        sequentialDFSUtil(source, visited);
    }

    void sequentialDFSUtil(int u, vector<bool> &visited){
        visited[u]=true;
        cout<<u<<" ";

        for(int v:adj[u]){
            if(!visited[v]){
                sequentialDFSUtil(v,visited);
            }
        }
    }

    void parallelBFS(int source){
        vector<bool> visited(V,false);
        queue<int> q;
        visited[source]=true;
        q.push(source);

        while(!q.empty()){
            int u;
            #pragma omp parallel
            {
                #pragma omp single
                {
                    u=q.front();
                    q.pop();
                    cout<<u<<" ";
                }
                #pragma omp for
                for(int i=0;i<adj[u].size();i++){
                    int v=adj[u][i];
                    bool alreadyVisited=false;
                    #pragma omp critical
                    {
                        alreadyVisited=visited[v];
                        if(!alreadyVisited){
                            visited[v]=true;
                            q.push(v);
                        }
                    }
                }
            }
        }
    }

    void parallelDFS(int source){
        vector<bool> visited(V,false);
        #pragma omp parallel
        {
            #pragma omp single
            parallelDFSUtil(source,visited);
        }
    }

    void parallelDFSUtil(int u, vector<bool> &visited){
        bool alreadyVisited=false;
        #pragma omp critical
        {
            if(visited[u]) alreadyVisited=true;
            else visited[u]=true;
        }
        if(alreadyVisited)return;
        cout<<u<<" ";
        #pragma omp parallel for
        for(int i=0;i<adj[u].size();i++){
            int v=adj[u][i];
            #pragma omp task
            parallelDFSUtil(v,visited);
        }
    }
};

int main(){
    Graph g(6);
    g.addEdge(0,1);
    g.addEdge(0,2);
    g.addEdge(1,3);
    g.addEdge(1,4);
    g.addEdge(2,4);
    g.addEdge(3,5);
    g.addEdge(4,5);

    auto start=high_resolution_clock::now();
    cout<<"Sequential BFS : ";
    g.sequentialBFS(0);
    auto end=high_resolution_clock::now();
    duration<double> time_elapsed=end-start;
    cout<<"\nTime taken for Sequential Execution : "<<time_elapsed.count()<<" seconds."<<endl;
    start=high_resolution_clock::now();
    cout<<"Parallel BFS : ";
    g.parallelBFS(0);
    end=high_resolution_clock::now();
    time_elapsed=end-start;
    cout<<"\nTime taken for Parallel Execution : "<<time_elapsed.count()<<" seconds."<<endl;
    start=high_resolution_clock::now();
    cout<<"Sequential DFS : ";
    g.sequentialDFS(0);
    end=high_resolution_clock::now();
    time_elapsed=end-start;
    cout<<"\nTime taken for Sequential Execution : "<<time_elapsed.count()<<" seconds."<<endl;
    start=high_resolution_clock::now();
    cout<<"Parallel DFS : ";
    g.parallelDFS(0);
    end=high_resolution_clock::now();
    time_elapsed=end-start;
    cout<<"\nTime taken for Parallel Execution : "<<time_elapsed.count()<<" seconds."<<endl;
    return 0;
}