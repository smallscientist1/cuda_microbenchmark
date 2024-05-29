#include "utils/do_bench.cuh"
#include <iostream>

#include<cute/tensor.hpp>


#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 128

#define Nthreads 256
#define STAGES 2

// b: [experts, N, K]
__global__ void fp8_kernel(uint8_t* b_ptr, int n, int k,int bn_stride, int loops){
    extern __shared__ uint8_t smem[];
    
    using namespace cute;
    cute::Tensor mB = cute::make_tensor(cute::make_gmem_ptr(b_ptr), cute::make_shape(n,k), cute::make_stride(bn_stride,Int<1>{}));
    cute::Tensor gB = cute::local_tile(mB, make_shape(Int<BLOCK_SIZE_N>{},Int<BLOCK_SIZE_K>{}), make_coord(blockIdx.x, _), Step<_1,_1>{}); // (BLK_N, BLK_K,k) (_64,_128,32):(4096,_1,_128)
    cute::Tensor sB = make_tensor(make_smem_ptr(smem), make_shape(Int<BLOCK_SIZE_N>{},Int<BLOCK_SIZE_K>{},Int<STAGES>{}), make_stride(Int<BLOCK_SIZE_K>{},Int<1>{},Int<BLOCK_SIZE_N*BLOCK_SIZE_K>{})); // (BLK_N, BLK_K) (_64,_128,_2):(_128,_1,_8192)

    using GmemTiledCopy = decltype(
        make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, uint8_t>{},
            make_layout(make_shape(Int<Nthreads/(BLOCK_SIZE_K/16)>{},Int<BLOCK_SIZE_K/16>{}), make_stride(Int<BLOCK_SIZE_K/16>{},Int<1>{})),
            make_layout(make_shape(Int<1>{},Int<16>{}))
        )
    );
    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
    Tensor tcopygB = gmem_thr_copy.partition_S(gB); // ((_16,_1),_2,_1,32):((_1,_0),131072,_0,_128)
    Tensor tcopysB = gmem_thr_copy.partition_D(sB); // ((_16,_1),_2,_1,_2):((_1,_0),_4096,_0,_8192)
    // here _,_,_, depends on GmemTiledCopy(不好的封装)
    copy(gmem_tiled_copy, tcopygB(_,_,_,0), tcopysB(_,_,_,0));
    cute::cp_async_fence();
    for(int i=1;i<size<2>(gB);i++){
        copy(gmem_tiled_copy, tcopygB(_,_,_,i), tcopysB(_,_,_,i%STAGES));
        cute::cp_async_fence();
        cute::cp_async_wait<1>();
    }
    cute::cp_async_wait<0>();

    // cute::Tensor tB = make_tensor<uint8_t>(make_shape());
    // Tensor tAgB = local_partition() // where local_partition is a lot like local_tile, except the coordinate slices into the tile-mode (the first mode) of the zipped_divide rather than the rest-mode (the second mode).
    // copy()

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // print(gB.layout());
        // print(sB.layout());
        // print(tcopygB.layout());
        // print(tcopysB.layout());
    }

}

int main(){
    int loops = 10000;
    int n = 12800, k =4096;
    dim3 grid(n/BLOCK_SIZE_N, 1, 1), block(Nthreads, 1, 1);
    int shared_mem = BLOCK_SIZE_N*BLOCK_SIZE_K*sizeof(uint8_t)*STAGES;
    uint8_t *d_b;
    cudaMalloc(&d_b, n*k * sizeof(uint8_t));
    float ms = do_bench([&]() { fp8_kernel<<<grid, block, shared_mem>>>(d_b,n,k,k, loops); });
    printf("done in %f ms\n", ms);
    cudaFree(d_b);
}
