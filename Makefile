NVCCFLAGS=-arch=sm_80 --use_fast_math -lineinfo -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++17

all: 2_exp_fmul 2_exp_fmul_allpeak

2_exp_fmul: 2_exp_fmul.cu
	nvcc ${NVCCFLAGS} -o 2_exp_fmul 2_exp_fmul.cu

2_exp_fmul_allpeak: 2_exp_fmul_allpeak.cu
	nvcc ${NVCCFLAGS} -o 2_exp_fmul_allpeak 2_exp_fmul_allpeak.cu

clean:
	rm -f 2_exp_fmul 2_exp_fmul_allpeak