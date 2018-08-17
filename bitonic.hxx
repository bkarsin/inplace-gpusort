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
#include"basecase/squareSort.hxx"

#define TYPE long int

#define ILP 1
#define BUFF 8

template<typename T, fptr_t f>
__global__ void bitonicKernel(T* data, int N) {
	squareSortDevice<T,f>(data, N);
}

template<typename T, fptr_t f>
__forceinline__ __device__ void cmpSwapRev(T* data, int idx1, int idx2) {
  T v1[ILP];
  T v2[ILP];

  T temp[ILP];

#pragma unroll
  for(int i=0; i<ILP; i++) {
    temp[i] = data[idx1+i];
    v2[i] = data[idx2-i];
  }
#pragma unroll
  for(int i=0; i<ILP; i++) {
    v1[i] = temp[i];
    if(f(v2[i],temp[i])) {
      v1[i] = v2[i];
      v2[i] = temp[i];
    }
  }
#pragma unroll
  for(int i=0; i<ILP; i++) {
    data[idx1+i] = v1[i];
    data[idx2-i] = v2[i];
  }
}

template<typename T, fptr_t f>
__forceinline__ __device__ void cmpSwapRegs(T* a, T* b) {
	T temp = *a;
	*a = *b;
	*b = temp;
}

template<typename T, fptr_t f>
__forceinline__ __device__ void cmpSwap(T* data, int idx1, int idx2) {
  T v1[ILP];
  T v2[ILP];

  T temp[ILP];

#pragma unroll
  for(int i=0; i<ILP; i++) {
    temp[i] = data[idx1+i];
    v2[i] = data[idx2+i];
  }
#pragma unroll
  for(int i=0; i<ILP; i++) {
    v1[i] = temp[i];
    if(f(v2[i],temp[i])) {
      v1[i] = v2[i];
      v2[i] = temp[i];
    }
  }
#pragma unroll
  for(int i=0; i<ILP; i++) {
    data[idx1+i] = v1[i];
    data[idx2+i] = v2[i];
  }
}
/*
template<typename T, fptr_t f>
__global__ void swapAllBlock(T* data, int N, int dist) {
  int eltsPerBlock = N/blockDim.x;

  for(int chunk=0; chunk < chunksPerBlock; chunk++) {
    for(int i=threadIdx.x; i<M*dist; i+=blockDim.x) {
      cmpSwap<T,f>(data, (chunkStart+chunk)*M*2*dist + i, (chunkStart+chunk)*M*2*dist +i+(M*dist));
    }
  }
}
*/
template<typename T, fptr_t f>
__global__ void swapAllBlock(T* data, int N, int dist) {
  int totalChunks = N/(M*2*dist);
  int chunksPerBlock = totalChunks/gridDim.x;
  if(threadIdx.x==0 && blockIdx.x==0) {
  printf("chunksPerBlock:%d\n", chunksPerBlock);
  }
  
  if(chunksPerBlock > 0) {
    int chunkStart = blockIdx.x*chunksPerBlock;

    for(int chunk=0; chunk < chunksPerBlock; chunk++) {
      for(int i=threadIdx.x; i<M*dist; i+=blockDim.x) {
        cmpSwap<T,f>(data, (chunkStart+chunk)*M*2*dist + i, ((chunkStart+chunk)*M*2*dist)+i+(M*dist));
      }
    }
  }
  /*
  else {
    int blocksPerChunk = gridDim.x/totalChunks;
    int eltsPerBlock = (M*dist)/blocksPerChunk;
    int blockOffset = (blockIdx.x/blocksPerChunk)*M*2*dist + (blockIdx.x%blocksPerChunk)*M*dist;
    for(int i=threadIdx.x; i<eltsPerBlock; i+=blockDim.x) {
      cmpSwap<T,f>(data, blockOffset+i, blockOffset+(M*dist)+i);
    }
  }
  */
}

template<typename T, fptr_t f>
__global__ void swapAll(T* data, int N, int dist) {
  int globalId = threadIdx.x + blockIdx.x*blockDim.x;
//int i=0;
  for(int i=0; i<N; i+=M*dist*2) {
//    for(int j=globalId*ILP; j<M*dist; j+=gridDim.x*blockDim.x*ILP) {
    int j=globalId;
    if(j < M*dist)
      cmpSwap<T,f>(data, i+j, M*dist+i+j);
    }
//  }
}

template<typename T, fptr_t f>
__global__ void swapAllRevRegs(T* data, int N, int dist) {
	int globalId = threadIdx.x + blockIdx.x*blockDim.x;
	int count=0;
	T buff1[BUFF];
	T buff2[BUFF];
	for(int i=0; i<N; i+=M*dist*2) {
		for(int j=globalId*ILP; j<M*dist; j+=gridDim.x*blockDim.x*ILP) {
			buff1[count] = data[i+j];
			buff2[count] = data[i+(M*dist*2)-j-1];
			count++;
			if(count == BUFF-1) {
				for(int k=0; k<count; k++) {
					cmpSwapRegs<T,f>(buff1+i, buff2+i);
				}
				for(int k=0; k<count; k++) {
					data[i+j - (k*gridDim.x*blockDim.x)] = buff1[i];
					data[i+(M*dist*2)-j-1 + (k*gridDim.x*blockDim.x)] = buff2[i];
				}
				count = 0;
			}
		}
	}
}

template<typename T, fptr_t f>
__global__ void swapAllRev(T* data, int N, int dist) {
  int globalId = threadIdx.x + blockIdx.x*blockDim.x;

  for(int i=0; i<N; i+=M*dist*2) {
    for(int j=globalId*ILP; j<M*dist; j+=gridDim.x*blockDim.x*ILP) {
      cmpSwapRev<T,f>(data, i+j, i+(M*dist*2)-j-1);
//      cmpSwap<T,f>(data, i+j, i+(M*dist*2)-j-1);
    }
  }
}

template<typename T,fptr_t f>
void bitonicSort(T* data, int N, int BLOCKS, int THREADS) {

  int baseBlocks=((N/M)/(THREADS/W));
  int roundDist=1;
  int subDist=1;

// baseBlocks = 16384;
// Sort the base case into blocks of 1024 elements each
  squareSort<T,f><<<baseBlocks,32>>>(data, N);
  cudaDeviceSynchronize();

  int levels = (int)log2((float)(N/M)+1)+1;
//  int levels = 19;
//  printf("levels:%d\n", levels);
  for(int i=1; i<levels; i++) {
	  
    swapAllRev<T,f><<<BLOCKS,THREADS>>>(data,N,roundDist);
//    swapAllRevRegs<T,f><<<BLOCKS,THREADS>>>(data,N,roundDist);
    cudaDeviceSynchronize();
    subDist = roundDist/2;
    for(int j=i-1; j>0; j--) {	
//      swapAllBlock<T,f><<<BLOCKS,THREADS>>>(data,N,subDist);
      swapAll<T,f><<<BLOCKS,THREADS>>>(data,N,subDist);
      cudaDeviceSynchronize();
      subDist /=2;
    }

//    squareSort<T,f><<<BLOCKS,32>>>(data, N);
//    cudaDeviceSynchronize();
    roundDist *=2;
  }

/*
  //2048
  swapAllRev<T,f><<<baseBlocks,THREADS>>>(data, N, 1);
  cudaDeviceSynchronize();
  squareSort<T,f><<<baseBlocks,THREADS>>>(data, N);
  cudaDeviceSynchronize();

  //4096
  swapAllRev<T,f><<<baseBlocks,THREADS>>>(data,N,2);
  cudaDeviceSynchronize();
  swapAll<T,f><<<baseBlocks,THREADS>>>(data,N,1);
  cudaDeviceSynchronize();
  squareSort<T,f><<<baseBlocks,THREADS>>>(data, N);
  cudaDeviceSynchronize();
*/


/* 
  int levels = (int)log2((float)size+1);

  for(int i=1; i<levels; i++) {
	for(int j=i; j>i; j--) {
		swapAll<T,f><<<BLOCKS,THREADS>>>(data, N, j);
	}
	squareSort<T,f><<<baseBlocks,THREADS>>>(data, N);
  }
*/
//  bitonicKernel<T,f><<<baseBlocks,THREADS>>>(data, N);

  cudaDeviceSynchronize();
}
