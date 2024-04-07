# L2cache
A100 PCIe 80GB
- partitioned L2 cache: 80 MB (https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821 ), but 40 MB on ncu report(device__attribute_l2_cache_size)
- L2 cache throughput (non compression, 1 sectors per elapsed cycle for load&store ): 32 (Bytes per sector) * 8 (l2 slices per framebuffer partition) * 10 (framebuffer partitions)= 2560 Bytes/cycle. 2560 * 1.41e9 (SM clock) = 3609.6 GB/s
- TODO: cub blockload(ld.global.nc, LDG.E.CONSTANT); CP_ASYNC(cp.async.cg.shared.global, LDGSTS.E.BYPASS.128), 1.5 sectors/cycle: 3840 Bytes/cycle, 3840 * 1.41e9 = 5414.4 GB/s

## L2 cache throughput breakdown
- lts__t_tag_requests: 1 requests/cycle
- lts2xbar: 2 sectors/active cycle (with L2 compression?)
- lts__t_sectors: 3 sectors/active cycle (with L2 compression? 2 sectors load + 1 sectors L2 fabric)
- lts__d_sectors: 4 sectors/active cycle
- xbar2lts: ???


## type int32
```
grid dim 160, block dim 1024
, sizeof type: 4
case 0: 64 MB copy, 7.158033 ms
BW: 1200.041229 GB/s
case 1: 2 MB copy, 0.109687 ms
BW: 2447.290944 GB/s
case 2: 32 MB copy, 3.486331 ms
BW: 1231.944622 GB/s
case 3: 16 MB copy, 0.990820 ms
BW: 2167.379567 GB/s
```

- ncu profile `2MB copy,iters 64` L2 memory chart:
L1 load: 134e6 Bytes ( = 2*64MB), hit rate: 50.14 %
L1 store: 134e6 Bytes ( = 2*64MB), hit rate: 100%
L2 fabric total: 110e6 Bytes (unknown)
L2: Xbar2lts Cycles Active [%]: 46.99 % 
small grid: 0.74 waves per SM

- ncu ptx & sass
`ld.volatile.global.u32` `LDG.E.STRONG.SYS`
`st.volatile.global.u32` `STG.E.STRONG.SYS`


378e6 / 46.99% / 0.11ms = 7310 GB/s
268e6 / 46.99% / 0.11ms = 5185 GB/s

### Questions
`ld.volatile.global` cannot be vectorized?
