# L2cache
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

- ncu profile `2MB copy,iters 64`:
L1 load: 134e6 Bytes ( = 2*64MB)
L1 store: 134e6 Bytes ( = 2*64MB)
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
