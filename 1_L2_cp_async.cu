#include <cute/algorithm/copy.hpp>
#include <cute/tensor.hpp>
#include "utils/do_bench.cuh"
#include <cstdio>

constexpr int iters = 100;

struct Params {
    void *__restrict__ d_q;         // Tile of input
    int q_bh_stride, q_kdim_stride;
    int batch, heads, l;
    int block_kdim;
    int vdim;
};

template<int NUM_THREADS, int ITEMS_PER_THRRADS, typename data_t>
__global__ void L2_kernel(Params params){
    // using data_t = float;
    extern __shared__ char smem[];

    const int bh_id = blockIdx.y;
    const int vdim_id = blockIdx.x;
    // const int h_id = bh_id % params.heads;

    data_t* qptr = reinterpret_cast<data_t*>(params.d_q) + bh_id * params.q_bh_stride;
    using namespace cute;
    constexpr int LOADATOM_PER_THREAD = 16/sizeof(data_t);
    cute::Tensor gQ = cute::make_tensor(cute::make_gmem_ptr(qptr), cute::make_shape(params.l/LOADATOM_PER_THREAD,Int<LOADATOM_PER_THREAD>{}), cute::make_stride(Int<LOADATOM_PER_THREAD>{},Int<1>{}));
    cute::Tensor sQ = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<data_t*>(smem)), cute::make_shape(params.l/LOADATOM_PER_THREAD,Int<LOADATOM_PER_THREAD>{}), cute::make_stride(Int<LOADATOM_PER_THREAD>{},Int<1>{}));
    using GmemCopyQ = decltype(
        make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, data_t>{},
            make_layout(Shape<Int<NUM_THREADS>,Int<1>>{}), // thread Layout
            make_layout(Shape<_1,Int<LOADATOM_PER_THREAD>>{})
        )
    );
    GmemCopyQ gmem_copy_q;
    auto gmem_thr_copy_q = gmem_copy_q.get_thread_slice(threadIdx.x);
    Tensor gQ_partition = gmem_thr_copy_q.partition_S(gQ);
    Tensor sQ_partition = gmem_thr_copy_q.partition_D(sQ);
    for (int i = 0; i < iters; i++) {
        for(int k=0;k<params.block_kdim;k++){
            cute::copy(gmem_copy_q, gQ_partition, sQ_partition);
            gQ_partition.data() = gQ_partition.data() + params.l;
            cute::cp_async_fence();
            cute::cp_async_wait<0>();
            __syncthreads();
        }
        // sQ(threadIdx.x,0) = sQ(threadIdx.x,0) + sQ(threadIdx.x,LOADATOM_PER_THREAD-1);
        // gQ_partition.data() = gQ.data(); // cause strange compile behavior
        gQ_partition.data() = gQ_partition.data() + params.l*(-params.block_kdim);
    }

}

template <
    int                     BLOCK_THREADS_,
    int                     ITEMS_PER_THREAD_,
    typename data_t_>
void L2_launch(Params &params, cudaStream_t stream) {
    int SmemSize = params.l * sizeof(data_t_);

    // printf("smem_size = %d\n", kSmemSize);
    // printf("smem_load_size = %d\n", Ktrait::kSmemLoadSize);
    // printf("smem_store_size = %d\n", Ktrait::kSmemStoreSize);
    // printf("smem_scan_size = %d\n", Ktrait::kSmemScanSize);
    // dim3 grid(params.batch * params.heads, params.vdim/params.block_vdim, params.kdim/params.block_kdim);
    // dim3 grid(params.batch * params.heads, params.vdim/params.block_vdim, 1);
    dim3 grid(params.vdim, params.batch * params.heads, 1);
    // printf("grid = %d %d %d\n", grid.x, grid.y, grid.z);
    auto kernel = &L2_kernel<BLOCK_THREADS_,ITEMS_PER_THREAD_,data_t_>;
    if (SmemSize >= 48 * 1024) {
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemSize);
    }
    kernel<<<grid, BLOCK_THREADS_, SmemSize, stream>>>(params);
}

void L2_cuda(void* d_q, int batch, int heads, int kdim, int vdim, int l, int block_kdim){

    Params params;
    params.d_q = d_q;
    params.batch = batch;
    params.heads = heads;
    params.l = l;
    params.vdim = vdim;
    params.block_kdim = block_kdim;

    params.q_bh_stride = kdim * l;
    params.q_kdim_stride = l*block_kdim;

    L2_launch<128, 16, float>(params, 0);

}

int main(){
    int batch = 4;
    int heads = 32;
    int kdim = 128;
    int vdim = 128;
    int l = 2048;
    int block_kdim = 128;
    float *d_q;
    cudaMalloc(&d_q, batch*heads*kdim*l*sizeof(float));
    L2_cuda(d_q, batch, heads, kdim, vdim, l, block_kdim);
    float ms = do_bench([&]() {
        L2_cuda(d_q, batch, heads, kdim, vdim, l, block_kdim);}, 500, 1000);
    printf("time = %f\n ms", ms);
    printf("bandwidth = %f GB/s\n", 1.0 * batch*heads*kdim*l*vdim*sizeof(float)*iters/ms/1e6);
    return 0;
}