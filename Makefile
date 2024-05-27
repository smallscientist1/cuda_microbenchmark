all: 2_exp_fmul

2_exp_fmul: 2_exp_fmul.cu
	nvcc -arch=sm_80 --use_fast_math -lineinfo -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++17 -o 2_exp_fmul 2_exp_fmul.cu

clean:
	rm -f 2_exp_fmul