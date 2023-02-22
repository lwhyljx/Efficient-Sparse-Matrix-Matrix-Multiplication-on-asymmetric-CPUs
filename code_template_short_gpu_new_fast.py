_CpuCodeTemplate = '''
#define __USE_GNU

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include "dev_array.h"
#include <math.h>
#include <pthread.h>
#include<sys/syscall.h>
#include<unistd.h>
using namespace std;

#define gettid() syscall(SYS_gettid)

void normal_matmul(vector<float> A, vector<float> B, vector<float> &C_right, int M, int N, int K)
{{
    for (int m = 0; m < M; m++) {{
        for (int n = 0; n < N; n++) {{
            C_right[m*N+n] = 0;
            for (int k = 0; k < K; k++) {{
                C_right[m*N+n] += A[m*K+k] * B[k*N+n];
            }}
        }}
    }}
}}
typedef struct{{
  int tid;
  float *input0;
  float *output0;
}}MY_ARGS;

/*inline int set_cpu(int tid)
{{
  cpu_set_t mask;
  pid_t pid=gettid();
  CPU_ZERO(&mask);
  CPU_SET(tid,&mask);
  int stat=sched_setaffinity(pid,sizeof(cpu_set_t),&mask);
  return stat;
}}*/

{}

int main()
{{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int M = {};
    int N = {};
    int K = {};
    int A_size = M*K;
    int B_size = K*N;
    int output_size = M*N;
    // Allocate memory on the host
    vector<float> h_A(A_size);
    vector<float> h_B(B_size);
    vector<float> h_C(output_size);
    vector<float> h_C_right(output_size);
    string line;
    ifstream matrix_file("../data1/sparse_matrix_{}.txt");
    int cnt = 0;
    while (getline(matrix_file, line)) {{
        //int val = stoi(line);
        float val = stof(line);
        h_A[cnt] = (float)val;
        //std::cout << val << std::endl;
        cnt++;
    }}
    matrix_file.close();
    for (int i = 0; i < B_size; i++)
        h_B[i] = 1;
    for (int i = 0; i < output_size; i++) {{
        h_C[i] = 0;
        h_C_right[i] = 0;
    }}
    // Allocate memory on the device
    dev_array<float> d_A(A_size);
    dev_array<float> d_B(B_size);
    dev_array<float> d_C(output_size);
    d_A.set(&h_A[0], A_size);
    d_B.set(&h_B[0], B_size);
    {}
    for(int i=0;i<30;i++){{
        {} 
    }}
    struct timeval start, end;
    int iter=50;
    double times=0;
    for(int i=0;i<iter;i++){{
        gettimeofday(&start,NULL);
        {}   
        gettimeofday(&end,NULL);
        double time=( end.tv_usec - start.tv_usec )/1000.0  + (end.tv_sec - start.tv_sec)*1000.0;
        times+=time;
    }}
    times=times/(iter*1.0);
    d_C.get(&h_C[0], output_size);
    // Now do the matrix multiplication on the CPU
    normal_matmul(h_A, h_B, h_C_right, M, N, K);
    for (int i = 0; i < output_size; i++)
        if (h_C[i] != h_C_right[i]) {{
             //cout << i << " " << h_C[i] << " " << h_C_right[i] << endl;
            cout << "Error: output not matched" << endl;
            break;
       }}
    cout << "Done" << endl;
    std::ofstream out;
    out.open("result.csv",std::ios::out|std::ios::app);
    out<<{}<<',';
    out<<{}<<',';
    out<<{}<<',';
    out<<{}<<',';
    out<<times<<endl;
    out.close();
    return 0;
}}

'''

_CpuBodyCodeTemplate = '''
__global__ void matrixMultiplication(float *input0, float *input1, float *output0)
{{
    {}
}}
'''

_CpuBodyCodePruneWeightTemplate = '''
__global__ void matrixMultiplication(float *input0, float *input1, float *output0)
{{
    {}
}}
'''