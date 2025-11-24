#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// -------------------------------------------
// 3 kernels for the 3 scenarios
// -------------------------------------------

// Scenario 1: one block, one thread
__global__ void vecAdd_1thread(float *A, float *B, float *C, int N) {
    // A single thread loops through all N elements.
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// Scenario 2: one block, 256 threads (CORRECTED)
__global__ void vecAdd_256threads(float *A, float *B, float *C, int N) {
    // Each thread starts at its threadIdx.x
    int i = threadIdx.x;
    // The total number of threads in the block is the stride
    int stride = blockDim.x; 

    // Use a loop to make the 256 threads process all N elements
    for (; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Scenario 3: multiple blocks, total threads approx = N
// This is the standard, high-performance parallel pattern.
__global__ void vecAdd_multi(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // (Optional but more robust) Add a grid-stride loop
    // int stride = gridDim.x * blockDim.x;
    // for (; i < N; i += stride) {
    //     C[i] = A[i] + B[i];
    // }
    // For this assignment, the simple version is sufficient and correct:
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
    long long N = (long long)K * 1000000; // Use long long for N to avoid overflow

    cout << "Scenario = " << scenario << ", Vector size = " << N << " elements\n";

    size_t size_bytes = sizeof(float) * N;

    // Allocate host memory
    float *hA = (float*)malloc(size_bytes);
    float *hB = (float*)malloc(size_bytes);
    float *hC = (float*)malloc(size_bytes);

    // Initialize
    for (long long i = 0; i < N; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    // Allocate device memory
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size_bytes);
    cudaMalloc(&dB, size_bytes);
    cudaMalloc(&dC, size_bytes);

    // Copy to device
    cudaMemcpy(dA, hA, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size_bytes, cudaMemcpyHostToDevice);

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
