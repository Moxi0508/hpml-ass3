#include <iostream>
#include <chrono>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: ./vecadd K\n";
        return 1;
    }

    long K = atol(argv[1]);   
    long N = K * 1000000L;    

    std::cout << "Vector size = " << N << " elements\n";

    // 分配内存
    float* A = (float*) malloc(N * sizeof(float));
    float* B = (float*) malloc(N * sizeof(float));
    float* C = (float*) malloc(N * sizeof(float));

    // 初始化向量
    for (long i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time elapsed: " << elapsed.count() << " seconds\n";

    free(A);
    free(B);
    free(C);

    return 0;
}
