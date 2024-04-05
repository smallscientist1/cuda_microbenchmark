#include "utils/do_bench.cuh"
#include <cstdint>
#include <iostream>

#define THREAD_ATOM 1

template <typename T>
__global__ void k(volatile T *__restrict__ d1, volatile T *__restrict__ d2,
                  const int loops, const int ds) {

  for (int i = 0; i < loops; i++)
    for (int j = (threadIdx.x + blockDim.x * blockIdx.x)*THREAD_ATOM; j < ds;
         j += gridDim.x * blockDim.x*THREAD_ATOM)
      if (i & 1){
        #pragma unroll
        for(int k=0;k<THREAD_ATOM;k++)
          d1[j+k] = d2[j+k];
      }
      else{
        #pragma unroll
        for(int k=0;k<THREAD_ATOM;k++)
          d2[j+k] = d1[j+k];
      }
}

const int dsize = 1048576 * 128;
const int iter = 64;
typedef int T;
// typedef uint64_t T;
int main() {

  T *d;
  cudaMalloc(&d, dsize);
  // case 1: 64MB copy, should exceed L2 cache on A100
  // 1048576 * 64/sizeof(T);
  // case 2: 2MB copy, should fit in L2 cache on A100
  // 1048576 * 2/sizeof(T);
  // case 3: 32MB copy, should exceed L2 cache on A100(32*2MB>40)
  // 1048576 * 32/sizeof(T);
  // case 4: 16MB copy, should fit in L2 cache on A100
  // 1048576 * 16/sizeof(T);

  int csize_list[] = {1048576 * 64 / sizeof(T), 1048576 * 2 / sizeof(T),
                      1048576 * 32 / sizeof(T), 1048576 * 16 / sizeof(T)};

  dim3 grid(512, 1, 1), block(256, 1, 1);
  printf("grid dim %d, block dim %d, sizeof type: %lu bytes\n", grid.x, block.x,
         sizeof(T));

  for (int i = 0; i < 4; i++) {
    int csize = csize_list[i];
    float ms =
        do_bench([&]() { k<<<grid, block>>>(d, d + csize, iter, csize); });
    printf("case %d: %lu MB copy, %f ms\n", i, csize * sizeof(T) / 1048576, ms);
    printf("BW: %f GB/s\n", 2.0 * csize * sizeof(T) * iter / ms / 1e6);
  }
}
