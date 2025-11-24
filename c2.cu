#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define P 1 // padding
#define BLOCK_SIZE 16

// Indexing macros
#define idxI(c, x, y) ((c)*(W+2*P)*(H+2*P) + (y)*(W+2*P) + (x))
#define idxF(k, c, i, j) ((k)*C*FH*FW + (c)*FH*FW + (i)*FW + (j))
#define idxO(k, x, y) ((k)*W*H + (y)*W + (x))

double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec + tp.tv_usec*1e-6;
}

// CUDA Kernel: Tiled Convolution
__global__ void conv2d_tiled(const double* __restrict__ I0, const double* __restrict__ F, double* O) {
    __shared__ double tile[C][BLOCK_SIZE + 2][BLOCK_SIZE + 2]; // shared tile with padding

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_x = blockIdx.x * BLOCK_SIZE + tx;
    int out_y = blockIdx.y * BLOCK_SIZE + ty;

    double sum[K] = {0.0}; // accumulate for all filters

    for (int c = 0; c < C; ++c) {
        // Load shared memory tile for this channel
        for (int dy = ty; dy < BLOCK_SIZE + 2; dy += BLOCK_SIZE) {
            for (int dx = tx; dx < BLOCK_SIZE + 2; dx += BLOCK_SIZE) {
                int load_x = blockIdx.x * BLOCK_SIZE + dx;
                int load_y = blockIdx.y * BLOCK_SIZE + dy;
                if (load_x < W + 2*P && load_y < H + 2*P) {
                    tile[c][dy][dx] = I0[idxI(c, load_x, load_y)];
                } else {
                    tile[c][dy][dx] = 0.0;
                }
            }
        }

        __syncthreads();

        // Compute convolution for this thread
        if (out_x < W && out_y < H) {
            for (int k = 0; k < K; ++k) {
                double s = 0.0;
                for (int i = 0; i < FH; ++i) {
                    for (int j = 0; j < FW; ++j) {
                        s += F[idxF(k, c, FW-1-j, FH-1-i)] * tile[c][ty+i][tx+j];
                    }
                }
                sum[k] += s;
            }
        }

        __syncthreads();
    }

    // Write results
    if (out_x < W && out_y < H) {
        for (int k = 0; k < K; ++k) {
            O[idxO(k, out_x, out_y)] = sum[k];
        }
    }
}

int main() {
    // Allocate host memory
    size_t size_I0 = C * (W + 2*P) * (H + 2*P) * sizeof(double);
    size_t size_F = K * C * FH * FW * sizeof(double);
    size_t size_O = K * W * H * sizeof(double);

    double* h_I0 = (double*)malloc(size_I0);
    double* h_F = (double*)malloc(size_F);
    double* h_O = (double*)malloc(size_O);

    // Initialize input tensor I0 and filters F
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H + 2*P; ++y)
            for (int x = 0; x < W + 2*P; ++x) {
                if (x >= P && x < W + P && y >= P && y < H + P) {
                    h_I0[idxI(c, x, y)] = c * ((x-P) + (y-P));
                } else {
                    h_I0[idxI(c, x, y)] = 0.0; // padding
                }
            }

    for (int k = 0; k < K; ++k)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < FH; ++i)
                for (int j = 0; j < FW; ++j)
                    h_F[idxF(k, c, i, j)] = (c + k) * (i + j);

    // Allocate device memory
    double *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, size_I0);
    cudaMalloc(&d_F, size_F);
    cudaMalloc(&d_O, size_O);

    // Copy data to device
    cudaMemcpy(d_I0, h_I0, size_I0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, size_F, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((W + BLOCK_SIZE -1)/BLOCK_SIZE, (H + BLOCK_SIZE -1)/BLOCK_SIZE);

    cudaDeviceSynchronize();
    double t0 = seconds();

    conv2d_tiled<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();

    double t1 = seconds();
    double exe = (t1 - t0) * 1000;

    // Copy result back to host
    cudaMemcpy(h_O, d_O, size_O, cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    for (int i = 0; i < K*W*H; ++i) checksum += h_O[i];
    printf("C2: %0.10e,%.3f\n", checksum, exe);

    // Free memory
    free(h_I0);
    free(h_F);
    free(h_O);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);

    return 0;
}
