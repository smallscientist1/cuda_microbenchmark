#include "utils/do_bench.cuh"
#include <iostream>

#include<cute/tensor.hpp>
#include<cutlass/numeric_conversion.h>


#define BLOCK_SIZE_M 16
#define BLOCK_SIZE_N 64
#define BLOCK_SIZE_K 128

#define Nthreads 128
#define Nwarps (Nthreads/32)
#define STAGES 3

// b: [experts, N, K]
__global__ void fp8_kernel(cutlass::float_e4m3_t* b_ptr, int n, int k,int bn_stride, int loops){
    extern __shared__ uint8_t smem[];
    
    using namespace cute;
    cute::Tensor mB = cute::make_tensor(cute::make_gmem_ptr(b_ptr), cute::make_shape(n,k), cute::make_stride(bn_stride,Int<1>{}));
    cute::Tensor gB = cute::local_tile(mB, make_shape(Int<BLOCK_SIZE_N>{},Int<BLOCK_SIZE_K>{}), make_coord(blockIdx.x, _), Step<_1,_1>{}); // (BLK_N, BLK_K,k) (_64,_128,32):(4096,_1,_128)
    cute::Tensor sB = make_tensor(make_smem_ptr((cutlass::float_e4m3_t*)smem), make_shape(Int<BLOCK_SIZE_N>{},Int<BLOCK_SIZE_K>{},Int<STAGES>{}), make_stride(Int<BLOCK_SIZE_K>{},Int<1>{},Int<BLOCK_SIZE_N*BLOCK_SIZE_K>{})); // (BLK_N, BLK_K) (_64,_128,_2):(_128,_1,_8192)

    using GmemTiledCopy = decltype(
        make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, cutlass::float_e4m3_t>{},
            make_layout(make_shape(Int<Nthreads/(BLOCK_SIZE_K/16)>{},Int<BLOCK_SIZE_K/16>{}), make_stride(Int<BLOCK_SIZE_K/16>{},Int<1>{})),
            make_layout(make_shape(Int<1>{},Int<16>{}))
        )
    );
    GmemTiledCopy gmem_tiled_copy;
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(threadIdx.x);
    Tensor tcopygB = gmem_thr_copy.partition_S(gB); // ((_16,_1),_2,_1,32):((_1,_0),131072,_0,_128)
    Tensor tcopysB = gmem_thr_copy.partition_D(sB); // ((_16,_1),_2,_1,_2):((_1,_0),_4096,_0,_8192)

    // TODO: sB_fp16 layout，tiledmma definition TO BE checked
    // TODO: sA, sB_fp16 pointer
    cute::Tensor sA = make_tensor(make_smem_ptr((half *)smem), make_shape(Int<BLOCK_SIZE_M>{},Int<BLOCK_SIZE_K>{},Int<STAGES>{}), make_stride(Int<BLOCK_SIZE_K>{},Int<1>{},Int<BLOCK_SIZE_M*BLOCK_SIZE_K>{})); // (BLK_M, BLK_K) (_16,_128,_2):(_128,_1,_2048)
    cute::Tensor sB_fp16 = make_tensor(make_smem_ptr((half *)(smem)), make_shape(Int<BLOCK_SIZE_N>{},Int<BLOCK_SIZE_K>{},Int<STAGES>{}), make_stride(Int<BLOCK_SIZE_K>{},Int<1>{},Int<BLOCK_SIZE_N*BLOCK_SIZE_K>{})); // (BLK_N, BLK_K) (_64,_128,_2):(_128,_1,_8192)


    TiledMMA<MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,Layout<Shape<Int<1>,Int<Nwarps>,_1>>,Layout<Shape<_1,_2,_1>>> tiled_mma;
    Tensor acc_o_fragment = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_SIZE_M>,Int<BLOCK_SIZE_N>>{}); 
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor rA = thr_mma.partition_fragment_A(sA(_,_,0));
    Tensor rB = thr_mma.partition_fragment_B(sB_fp16(_,_,0)); // 

    auto tiled_copy_A = make_tiled_copy_A(Copy_Atom<DefaultCopy, half>{}, tiled_mma);
    auto  thr_copy_A = tiled_copy_A.get_thread_slice(threadIdx.x);
    Tensor tmmarA = thr_copy_A.retile_D(rA); // 
    Tensor tmmasA = thr_copy_A.partition_S(sA); // ((_1,(_2,_2,_2)),_1,_8,_3):((_0,(_1,_1024,_8)),_0,_16,_2048)
    auto tiled_copy_B = make_tiled_copy_B(Copy_Atom<DefaultCopy, half>{}, tiled_mma);
    auto  thr_copy_B = tiled_copy_B.get_thread_slice(threadIdx.x);
    Tensor tmmarB = thr_copy_B.retile_D(rB); // 

    // load sB -> rB_fp8, Convert rB_fp8 -> rB_fp16
    // TODO: store rB_fp16 to sB_fp16
    cute::Tensor tile_sB = local_partition(sB, make_layout(make_shape(Int<Nthreads/(BLOCK_SIZE_K/16)>{},Int<BLOCK_SIZE_K/16>{})), threadIdx.x); // (_4,_16,_3):(_2048,_8,_8192)
    Tensor rB_fp8 = make_tensor<cutlass::float_e4m3_t>(make_shape(size<0>(tile_sB),size<1>(tile_sB)), LayoutRight{}); // row-major
    cute::copy(tile_sB(_,_,0), rB_fp8);
    cutlass::NumericArrayConverter<half, cutlass::float_e4m3_t, size(rB_fp8)> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<cutlass::float_e4m3_t, decltype(size(rB_fp8))::value> *>(rB_fp8.data()));
    Tensor rB_fp16 = make_tensor(make_rmem_ptr<half>(&frag), rB_fp8.layout()); // (_4,_16):(_16,_1)


// end definition-------------------------------------------------------------------------------------------
// start computation-------------------------------------------------------------------------------------------
                                                                    // here _,_,_, depends on GmemTiledCopy(不好的封装)
    // pipelined, stage=STAGES
    // TODO: do convert, mma
    #pragma unroll
    for(int i=0;i<STAGES-1;i++){
        copy(gmem_tiled_copy, tcopygB(_,_,_,i), tcopysB(_,_,_,i));
        cute::cp_async_fence();
    }
    for(int i=STAGES-1;i<size<2>(gB);i++){
        copy(gmem_tiled_copy, tcopygB(_,_,_,i), tcopysB(_,_,_,i%STAGES));
        cute::cp_async_fence();
        cute::cp_async_wait<STAGES-1>();
    }
    cute::cp_async_wait<0>();

                                                               // where local_partition is a lot like local_tile, except the coordinate slices into the tile-mode (the first mode) of the zipped_divide rather than the rest-mode (the second mode).

// end computation-------------------------------------------------------------------------------------------
// for debug-------------------------------------------------------------------------------------------
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // print(gB.layout());
        // print(sB.layout());
        // print(tcopygB.layout());
        // print(tcopysB.layout());
        // print(rB.layout());
        // print(tmmasA.layout());
        // print(tile_sB.layout());
        // print(rB_fp16.layout());
    }

}

int main(){
    int loops = 10000;
    int n = 12800, k =4096;
    dim3 grid(n/BLOCK_SIZE_N, 1, 1), block(Nthreads, 1, 1);
    int shared_mem = BLOCK_SIZE_N*BLOCK_SIZE_K*sizeof(uint8_t)*STAGES;
    uint8_t *d_b;
    cudaMalloc(&d_b, n*k * sizeof(uint8_t));
    float ms = do_bench([&]() { fp8_kernel<<<grid, block, shared_mem>>>((cutlass::float_e4m3_t*)d_b,n,k,k, loops); });
    printf("done in %f ms\n", ms);
    cudaFree(d_b);
}
