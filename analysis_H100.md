# L2 cache
NVIDIA H100 SXM5 80GB HBM3 
- cp.async: 1.86 sectors/cycle, 吞吐量与SM频率无关
## type int32
```
grid dim 512, block dim 256, sizeof type: 4 bytes
case 0: 64 MB copy, 4.707605 ms
BW: 1824.693009 GB/s
case 1: 2 MB copy, 0.067002 ms
BW: 4006.371825 GB/s
case 2: 32 MB copy, 2.337165 ms
BW: 1837.682253 GB/s
case 3: 16 MB copy, 0.511400 ms
BW: 4199.223627 GB/
```

## 1_L2_cp_async
```
time = 203.829147
 msbandwidth = 8428.563534 GB/s
```
ncu的结果为 205 ms(1590MHz，1980Mhz)
- 功耗墙，当有其他任务时，测试时间变为 461 ms。