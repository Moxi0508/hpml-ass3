#include <iostream>
#include <cuda.h>
using namespace std;

// kernel definitions same as Q2
__global__ void vecAdd_1thread(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

__global__ void vecAdd_256threads(float *A, float *B, float *C, int N) {
    int i = threadIdx.x;
    int stride = blockDim.x;
    for (int idx = i; idx < N; idx += stride) C[idx] = A[idx] + B[idx];
}

__global__ void vecAdd_multi(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "Usage: ./q3 <scenario: 1|2|3> <K>" << endl;
        return 1;
    }

    int scenario = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = K * 1000000;

    cout << "Scenario = " << scenario << ", Vector size = " << N << endl;

    // Unified memory allocation
    float *A, *B, *C;
    cudaMallocManaged(&A, N*sizeof(float));
    cudaMallocManaged(&B, N*sizeof(float));
    cudaMallocManaged(&C, N*sizeof(float));

    // Initialize
    for (int i=0; i<N; i++) { A[i]=1.0f; B[i]=2.0f; }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    if (scenario == 1) {
        vecAdd_1thread<<<1,1>>>(A,B,C,N);
    } else if (scenario == 2) {
        vecAdd_256threads<<<1,256>>>(A,B,C,N);
    } else if (scenario == 3) {
        int threads = 256;
        int blocks = (N + threads -1)/threads;
        vecAdd_multi<<<blocks, threads>>>(A,B,C,N);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Time elapsed: " << ms/1000.0f << " seconds\n";

    // Free unified memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
