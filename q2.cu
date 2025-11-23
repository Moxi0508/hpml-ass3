#include <iostream>
#include <cuda.h>
using namespace std;

// -------------------------------------------
// 3 kernels for the 3 scenarios
// -------------------------------------------

// Scenario 1: one block, one thread
__global__ void vecAdd_1thread(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// Scenario 2: one block, 256 threads
__global__ void vecAdd_256threads(float *A, float *B, float *C, int N) {
    int i = threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Scenario 3: multiple blocks, total threads = N
__global__ void vecAdd_multi(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// -------------------------------------------
// main program
// -------------------------------------------
int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "Usage: ./q2 <scenario: 1|2|3> <K>" << endl;
        return 1;
    }

    int scenario = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = K * 1000000;

    cout << "Scenario = " << scenario << ", Vector size = " << N << " elements\n";

    // Allocate host memory
    float *hA = (float*)malloc(sizeof(float) * N);
    float *hB = (float*)malloc(sizeof(float) * N);
    float *hC = (float*)malloc(sizeof(float) * N);

    // Initialize
    for (int i = 0; i < N; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(float) * N);
    cudaMalloc(&dB, sizeof(float) * N);
    cudaMalloc(&dC, sizeof(float) * N);

    // Copy to device
    cudaMemcpy(dA, hA, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch appropriate kernel
    if (scenario == 1) {
        vecAdd_1thread<<<1,1>>>(dA, dB, dC, N);

    } else if (scenario == 2) {
        vecAdd_256threads<<<1,256>>>(dA, dB, dC, N);

    } else if (scenario == 3) {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        vecAdd_multi<<<blocks, threads>>>(dA, dB, dC, N);
    }

    // Make sure kernel is finished
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Execution time: " << ms/1000.0f << " seconds\n";

    // Free
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
