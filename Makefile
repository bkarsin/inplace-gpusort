#OPTIONS=-std=c++11 -Xcompiler="-Wundef" -O2 -g -Xcompiler="-Werror" -lineinfo  --expt-extended-lambda -use_fast_math -Xptxas="-v" -D_FORCE_INLINES

GENCODE_SM20	:= -gencode arch=compute_20,code=sm_20
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
#GENCODE_SM52	:= -arch compute_52 -code sm_52

#UPDATE THE GENCODE HERE FOR YOUR PARTICULAR HARDWARE

OPTIONS=-std=c++11  -lineinfo -use_fast_math --expt-extended-lambda -Xptxas -dlcm=cg -lcudart -D_FORCE_INLINES $(GENCODE_SM52)

default: bitonicSort

bitonicSort: bitonicSort.o
	nvcc bitonicSort.o $(OPTIONS) -o bitonicSort

bitonicSort.o: bitonicSort.cu buildData.h basecase/squareSort.hxx basecase/sortRowMajor.hxx bitonic.hxx
	nvcc $(OPTIONS) -c bitonicSort.cu

#eval-mgpu: eval-mgpu.cu
#	nvcc eval-mgpu.cu -std=c++11 -O3 -arch=sm_30 -D_FORCE_INLINES --expt-extended-lambda -I "../mgpu/src" -o eval-mgpu

clean:
	rm -f *.o bitonicSort
