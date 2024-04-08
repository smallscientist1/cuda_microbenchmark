# L2cache
RTX4090
- 72 MB Cache
- L2 cache throughput **compute** (non compression, 1 sectors per elapsed cycle for load&store ): 32 (Bytes per sector) * 8 (l2 slices per framebuffer partition) * 6 (framebuffer partitions)= 1536 Bytes/cycle. 1536 * 2.52e9 (SM clock) = 3871 GB/s
- L2 cache throughput **actual** ( 1 sectors/cycle for load&store): 1024 sectors/cycle (32 l2s concurrent, not 48)
- CP_ASYNC(cp.async.cg.shared.global, LDGSTS.E.BYPASS.128), 1.9 sectors/cycle: 1945.6 Bytes/cycle,   4903 GB/s


## TODO


## type int32
```
grid dim 512, block dim 256, sizeof type: 4 bytes
case 0: 64 MB copy, 10.480535 ms
BW: 819.608442 GB/s
case 1: 2 MB copy, 0.082485 ms
BW: 3254.371352 GB/s
case 2: 32 MB copy, 1.852528 ms
BW: 2318.436161 GB/s
case 3: 16 MB copy, 0.589156 ms
BW: 3645.014244 GB/s
```
