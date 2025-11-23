

#include "vecaddKernel.h"

__global__ void AddVectors(const float* A, const float* B, float* C, int ValuesPerThread)
{
    
    const int N = gridDim.x * blockDim.x * ValuesPerThread;


    const int thread_start_index = blockIdx.x * blockDim.x + threadIdx.x;
    

    const int total_threads = gridDim.x * blockDim.x;

    for (int i = thread_start_index; i < N; i += total_threads)
    {
        C[i] = A[i] + B[i];
    }
}
