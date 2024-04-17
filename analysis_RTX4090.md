# L2cache
RTX4090
- 72 MB Cache
- L2 cache throughput **compute** (non compression, 1 sectors per elapsed cycle for load&store ): 32 (Bytes per sector) * 8 (l2 slices per framebuffer partition) * 6 (framebuffer partitions)= 1536 Bytes/cycle. 1536 * 2.52e9 (SM clock) = 3871 GB/s
- L2 cache throughput **actual** ( 1 sectors/cycle for load&store): 1024 sectors/cycle (32 l2s concurrent, not 48), 2580 GB/s
- load volatile only(ld.volatile.global.u32, LDG.E.STRONG.SYS), 1.5 sectors/cycle:  1536 Bytes/cycle, 3871 GB/s
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

### ncu profile `16MB copy,iters 64` L2 memory chart:
- L1Load: 1074e6 Bytes 35.77%
- L1Store: 1074e6 Bytes 71.54%
- L2: Xbar2lts Cycles Active [%]  88.47%

### peak L2 bandwidth
- load: 1074e6 Bytes / 0.3577 / 0.59ms = 5089 GB/s
- store: 1074e6 Bytes / 0.7154 / 0.59ms = 2544 GB/s

### hypothesis
- 35.77% + 71.54% = 107.31% reach the peak, 71% cycles for store, 36% cycles for load
