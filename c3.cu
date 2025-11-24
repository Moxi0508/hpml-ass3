// c3.cu
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#define C 3
#define H 1024
#define W 1024
#define K 64
#define FH 3
#define FW 3
#define P 1  // padding

int main() {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    const int N = 1; // batch size

    // Allocate and initialize input tensor on device
    double *d_input;
    size_t input_bytes = N * C * H * W * sizeof(double);
    cudaMalloc(&d_input, input_bytes);
    std::vector<double> h_input(N * C * H * W);
    for (int c = 0; c < C; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                h_input[c*H*W + i*W + j] = c * (i + j);
    cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice);

    // Allocate and initialize filter tensor on device
    double *d_filter;
    size_t filter_bytes = K * C * FH * FW * sizeof(double);
    cudaMalloc(&d_filter, filter_bytes);
    std::vector<double> h_filter(K * C * FH * FW);
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < FH; ++i)
                for (int j = 0; j < FW; ++j)
                    h_filter[k*C*FH*FW + c*FH*FW + i*FW + j] = (c + k) * (i + j);
    cudaMemcpy(d_filter, h_filter.data(), filter_bytes, cudaMemcpyHostToDevice);

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                               N, C, H, W);

    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW,
                               K, C, FH, FW);

    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc, P, P, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_DOUBLE);

    // Get output dimensions
    int outN, outC, outH, outW;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                          &outN, &outC, &outH, &outW);

    // Allocate output
    double *d_output;
    size_t output_bytes = outN * outC * outH * outW * sizeof(double);
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);

    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
                               outN, outC, outH, outW);

    // Find fastest algorithm
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returnedAlgoCount;
    cudnnFindConvolutionForwardAlgorithm(cudnn,
                                         input_desc,
                                         filter_desc,
                                         conv_desc,
                                         output_desc,
                                         1, // request 1 best algo
                                         &returnedAlgoCount,
                                         &perf);

    // Allocate workspace
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc,
                                            conv_desc, output_desc, perf.algo,
                                            &workspace_bytes);
    if (workspace_bytes > 0) cudaMalloc(&workspace, workspace_bytes);

    // Alpha and beta must be variables
    double alpha = 1.0;
    double beta  = 0.0;

    // Launch convolution and time it
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    cudnnConvolutionForward(cudnn,
                            &alpha,
                            input_desc, d_input,
                            filter_desc, d_filter,
                            conv_desc, perf.algo,
                            workspace, workspace_bytes,
                            &beta,
                            output_desc, d_output);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end-start).count();

    // Copy output back and compute checksum
    std::vector<double> h_output(outN*outC*outH*outW);
    cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (double v : h_output) checksum += v;

    std::cout << "C3: " << checksum << ", "<< std::fixed << std::setprecision(3) << elapsed_ms << "\n";
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
