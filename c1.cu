#include <cstdio>
#include <cuda_runtime.h>

#define C 3
#define H 1024
#define W 1024
#define FH 3
#define FW 3
#define K 64
#define P 1

// CUDA error checking
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

// Kernel: simple convolution, no shared memory, no tiling
__global__ void conv_kernel(const double* I0, const double* F, double* O) {
    int k = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x >= W || y >= H || k >= K) return;

    double sum = 0.0;

    for(int c = 0; c < C; ++c) {
        for(int j = 0; j < FH; ++j) {
            for(int i = 0; i < FW; ++i) {
                int I_row = y + j;
                int I_col = x + i;
                // F transpose
                double fval = F[((k*C + c)*FH + (FH-1-j))*FW + (FW-1-i)];
                double ival = I0[(c*(H+2*P) + I_row)*(W+2*P) + I_col];
                sum += fval * ival;
            }
        }
    }
    O[(k*H + y)*W + x] = sum;
}

int main() {
    size_t size_I0 = C*(H+2*P)*(W+2*P)*sizeof(double);
    size_t size_F  = K*C*FH*FW*sizeof(double);
    size_t size_O  = K*H*W*sizeof(double);

    double *h_I0 = new double[C*(H+2*P)*(W+2*P)];
    double *h_F  = new double[K*C*FH*FW];
    double *h_O  = new double[K*H*W];

    // Initialize I0 with padding
    for(int c=0;c<C;++c){
        for(int y=0;y<H+2*P;++y){
            for(int x=0;x<W+2*P;++x){
                if(y==0 || y==H+1 || x==0 || x==W+1)
                    h_I0[(c*(H+2*P)+y)*(W+2*P)+x] = 0.0;
                else
                    h_I0[(c*(H+2*P)+y)*(W+2*P)+x] = c*(x-1 + y-1);
            }
        }
    }

    // Initialize F
    for(int k=0;k<K;++k){
        for(int c=0;c<C;++c){
            for(int j=0;j<FH;++j){
                for(int i=0;i<FW;++i){
                    h_F[((k*C+c)*FH+j)*FW+i] = (c+k)*(i+j);
                }
            }
        }
    }

    // Allocate device memory
    double *d_I0, *d_F, *d_O;
    CUDA_CHECK(cudaMalloc(&d_I0, size_I0));
    CUDA_CHECK(cudaMalloc(&d_F, size_F));
    CUDA_CHECK(cudaMalloc(&d_O, size_O));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_I0, h_I0, size_I0, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice));

    // Kernel launch
    dim3 block(16,16);
    dim3 grid((W+block.x-1)/block.x, (H+block.y-1)/block.y, K);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    conv_kernel<<<grid, block>>>(d_I0, d_F, d_O);
    CUDA_CHECK(cudaEventRecord(stop,0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Kernel execution time: %f ms\n", elapsed_ms);

    // Copy output back
    CUDA_CHECK(cudaMemcpy(h_O, d_O, size_O, cudaMemcpyDeviceToHost));

    // Compute checksum
    double checksum = 0.0;
    for(size_t i=0;i<K*H*W;++i) checksum += h_O[i];
    printf("Checksum: %f\n", checksum);

    // Free memory
    delete[] h_I0;
    delete[] h_F;
    delete[] h_O;
    CUDA_CHECK(cudaFree(d_I0));
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_O));

    return 0;
}
