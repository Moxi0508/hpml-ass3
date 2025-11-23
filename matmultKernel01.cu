// matmultKernel01.cu
// Final improved kernel for matmult01
// Each thread (in a 16x16 block) computes a 2x2 sub-block of an F x F tile,
// where F = FOOTPRINT_SIZE (compiled with -DFOOTPRINT_SIZE=32 in Makefile).
//
// Compatible with matmultKernel.h:
//   BLOCK_SIZE is defined there (16)
//   FOOTPRINT_SIZE is provided by Makefile (32)
// Kernel signature matches header: __global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

#include "matmultKernel.h"

#ifndef FOOTPRINT_SIZE
  #define FOOTPRINT_SIZE (2 * BLOCK_SIZE)
#endif

// Ensure consistency: we expect FOOTPRINT_SIZE == 2 * BLOCK_SIZE for this scheme.
// (Makefile already compiles matmult01 with -DFOOTPRINT_SIZE=32 and BLOCK_SIZE=16)
static_assert((FOOTPRINT_SIZE == 2 * BLOCK_SIZE), "This kernel expects FOOTPRINT_SIZE == 2*BLOCK_SIZE");

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C) {
  // Thread coords inside thread block (0..BLOCK_SIZE-1)
  const int trow = threadIdx.y;
  const int tcol = threadIdx.x;

  // Block coords in grid (in units of FOOTPRINT tiles)
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int F = FOOTPRINT_SIZE;
  const int Bsize = BLOCK_SIZE; // 16

  // Base pointer to this block's C footprint (top-left of the F x F tile)
  float *Cblock = &C.elements[C.stride * (F * block_row) + (F * block_col)];

  // Each thread computes a 2x2 sub-block at local rows r0/r1 and cols c0/c1 within the footprint
  const int r0 = trow;           // 0..B-1
  const int r1 = trow + Bsize;   // B..F-1
  const int c0 = tcol;           // 0..B-1
  const int c1 = tcol + Bsize;   // B..F-1

  // accumulators for the 2x2 result
  float c00 = 0.0f;
  float c01 = 0.0f;
  float c10 = 0.0f;
  float c11 = 0.0f;

  // Shared memory for full footprint tiles (F x F)
  // Size with F=32 -> 1024 floats -> ~4KB; two tiles => ~8KB: fine.
  __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
  __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

  // Number of F-sized tiles across the width (matrices are square and widths are multiples of F)
  int numTiles = A.width / F;

  for (int m = 0; m < numTiles; ++m) {
    // Pointers to this tile's top-left in global memory
    const float *Asub = &A.elements[A.stride * (F * block_row) + (F * m)];
    const float *Bsub = &B.elements[B.stride * (F * m) + (F * block_col)];

    // ----------------------------------------------------------------
    // Cooperative load: each of the Bsize*Bsize threads loads 4 elements:
    // positions (r0,c0), (r0,c1), (r1,c0), (r1,c1) into shared_A
    // and similarly for shared_B.
    // This fills the entire F x F region because (F*F) / (Bsize*Bsize) = 4.
    // ----------------------------------------------------------------

    // Load A: four elements per thread (coalesced)
    shared_A[r0][c0] = Asub[r0 * A.stride + c0];
    shared_A[r0][c1] = Asub[r0 * A.stride + c1];
    shared_A[r1][c0] = Asub[r1 * A.stride + c0];
    shared_A[r1][c1] = Asub[r1 * A.stride + c1];

    // Load B: four elements per thread (coalesced)
    // Do NOT transpose here; keep B stored as [row][col] so reads in compute stage are consistent.
    shared_B[r0][c0] = Bsub[r0 * B.stride + c0];
    shared_B[r0][c1] = Bsub[r0 * B.stride + c1];
    shared_B[r1][c0] = Bsub[r1 * B.stride + c0];
    shared_B[r1][c1] = Bsub[r1 * B.stride + c1];

    // Wait until whole tile loaded
    __syncthreads();

    // ----------------------------------------------------------------
    // Compute partial products: iterate k across the full F dimension
    // Use pragma unroll to encourage the compiler to unroll the inner loop.
    // For each k we read A[r0,k], A[r1,k] and B[k,c0], B[k,c1].
    // These accesses use contiguous shared memory rows/columns and are consistent
    // with how we loaded shared_A/shared_B above.
    // ----------------------------------------------------------------
#pragma unroll
    for (int k = 0; k < F; ++k) {
      float A_r0_k = shared_A[r0][k];
      float A_r1_k = shared_A[r1][k];

      float B_k_c0 = shared_B[k][c0];
      float B_k_c1 = shared_B[k][c1];

      c00 += A_r0_k * B_k_c0;
      c01 += A_r0_k * B_k_c1;
      c10 += A_r1_k * B_k_c0;
      c11 += A_r1_k * B_k_c1;
    }

    // Wait before next tile load to avoid race on shared memory
    __syncthreads();
  } // end tile loop

  // ----------------------------------------------------------------
  // Write back the 2x2 results to global memory.
  // Compute global coordinates of top-left of this thread's 2x2 block:
  // global_row = block_row*F + r0 (and +1 for r1)
  // global_col = block_col*F + c0 (and +1 for c1)
  // Because threads with consecutive tcol write consecutive columns, writes are coalesced.
  // ----------------------------------------------------------------
  int global_r0 = block_row * F + r0;
  int global_r1 = block_row * F + r1;
  int global_c0 = block_col * F + c0;
  int global_c1 = block_col * F + c1;

  C.elements[global_r0 * C.stride + global_c0] = c00;
  C.elements[global_r0 * C.stride + global_c1] = c01;
  C.elements[global_r1 * C.stride + global_c0] = c10;
  C.elements[global_r1 * C.stride + global_c1] = c11;
}
