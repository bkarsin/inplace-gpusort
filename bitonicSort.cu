/*
Copyright (c) 2017 Ben Karsin

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and
associated documentation files (the "Software"), to
deal in the Software without restriction, including
without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom
the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include<algorithm>
#include"bitonic.hxx"
#include"buildData.h"
//#include"basecase/squareSort.hxx"

//#define DEBUG 1

#define TYPE int


template<typename T>
__global__ void bitonicSort(T* data, int N);

int main(int argc, char** argv) {

  if(argc != 4) {
	printf("usage: bitonic <N> <BLOCKS> <THREADS>\n");
	exit(1);
  }

  cudaEvent_t start, stop;
  float time_elapsed=0.0;
  int N = atoi(argv[1]);
  int BLOCKS = atoi(argv[2]);
  int THREADS = atoi(argv[3]);

 // Create sample sorted lists
  TYPE* h_data = (TYPE*)malloc(N*sizeof(TYPE));

  TYPE* d_data;
  cudaMalloc(&d_data, N*sizeof(TYPE));
  float total_time=0.0;

  srand(time(NULL));

  create_random_list<TYPE>(h_data, N, 0);

  cudaMemcpy(d_data, h_data, N*sizeof(TYPE), cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

//  bitonicSort<TYPE><<<BLOCKS,THREADS>>>(d_data,N);
  bitonicSort<TYPE,cmp>(d_data,N,BLOCKS, THREADS);

  cudaDeviceSynchronize();

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_elapsed, start, stop);

  printf("%lf\n", time_elapsed);

  cudaMemcpy(h_data, d_data, N*sizeof(TYPE), cudaMemcpyDeviceToHost);

#ifdef DEBUG
  bool error=false;
  for(int i=1; i<N; i++) {
    if(h_data[i-1] > h_data[i]) {
      error=true;
      printf("i:%d, %d > %d\n", i,h_data[i-1], h_data[i]);
    }
  }

  if(error)
    printf("NOT SORTED!\n");
  else
    printf("SORTED!\n");
#endif

  cudaFree(d_data);
  free(h_data);
}

