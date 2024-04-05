/*
no volatile to use float4,
but the load was optimized.
*/
#include "utils/do_bench.cuh"
#include <cstdint>
#include <iostream>


template <typename T>
__global__ void k(T *__restrict__ d1, T *__restrict__ d2,
                  const int loops, const int ds) {
  for (int i = 0; i < loops; i++)
    for (int j = (threadIdx.x + blockDim.x * blockIdx.x); j < ds;
         j += gridDim.x * blockDim.x)
      if (i & 1){
          d1[j]=d2[j];
      }
      else{
          d2[j]=d1[j];
      }
}

const int dsize = 1048576 * 128;
const int iter = 64;
// typedef float T;
typedef float4 T;
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
