// using ncu to demonstrate that XU can parallel with FMA pipeline
#include "utils/do_bench.cuh"
#include <iostream>

// A100 FP32 64 cores/SM, SFU 16 cores/SM
#define REG_NUM 64
#define REG_NUM_EXP 32
__global__ void k(float* g_a, float* g_b, const int loops){
    float a[REG_NUM], a1[REG_NUM];
    float b[REG_NUM_EXP];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    #pragma unroll
    for(int j=0;j<REG_NUM;j++){
        a[j] = g_a[idx*REG_NUM+j] ;
    }
    #pragma unroll
    for(int j=0;j<REG_NUM_EXP;j++){
        b[j] = g_b[idx*REG_NUM_EXP+j] ;
    }

    for(int i=0;i<loops;i++){
        for(int j=0;j<REG_NUM_EXP;j++){
            b[j] = exp2f(b[j]);
        }
        for(int j=0;j<REG_NUM;j++){
            a1[j] *= a[j];
        }
        for(int j=0;j<REG_NUM;j++){
            a1[j] *= 1.1f;
            // b[j] = exp2f(b[j]);
        }
        for(int j=0;j<REG_NUM;j++){
            a1[j] += a[j];
            // b[j] = exp2f(b[j]);
        }
    }

    #pragma unroll
    for(int j=0;j<REG_NUM;j++){
        g_a[idx*REG_NUM+j] = a1[j];
    }
    #pragma unroll
    for(int j=0;j<REG_NUM_EXP;j++){
        g_b[idx*REG_NUM_EXP+j] = b[j];
    }
}

int main(){
    int loops = 10000;
    dim3 grid(512, 1, 1), block(256, 1, 1);
    float* d_a, *d_b;
    cudaMalloc(&d_a, grid.x * block.x * REG_NUM * sizeof(float));
    cudaMalloc(&d_b, grid.x * block.x * REG_NUM_EXP * sizeof(float));
    float ms = do_bench([&]() { k<<<grid, block>>>(d_a, d_b, loops); });
    printf("done in %f ms\nFMA flops: %f TFLOPs/s\nXU flops: %f TFLOPs/s\n", ms, 3*grid.x*block.x*REG_NUM*float(loops/1e9)/ms, 1*grid.x*block.x*REG_NUM_EXP*float(loops/1e9)/ms);

}
