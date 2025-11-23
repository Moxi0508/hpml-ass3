#include "vecaddKernel.h" 

__global__ void AddVectors(const float* A, const float* B, float* C, int unused_N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int totalSize = gridDim.x * blockDim.x * unused_N;

 
    if (i < totalSize)
    {
        C[i] = A[i] + B[i];
    }
}
