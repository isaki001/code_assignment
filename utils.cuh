#ifndef IOANNIS_SAKIOTIS_CESIUM_INTERVIEW
#define IOANNIS_SAKIOTIS_CESIUM_INTERVIEW

#include <array>
#include <cstdio>
#include "cuda.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

  inline void
  __cudaCheckError(const char* file, const int line)
  {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr,
              "cudaCheckError() failed at %s:%i : %s\n",
              file,
              line,
              cudaGetErrorString(err));
      print_trace();
      abort();
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
      fprintf(stderr,
              "cudaCheckError() with sync failed at %s:%i : %s\n",
              file,
              line,
              cudaGetErrorString(err));
      print_trace();
      abort();
    }
#endif

    return;
  }

__forceinline__
__device__
size_t
index(size_t r, size_t c, size_t w){
  return r * w + c;
}

template <typename T>
void
cuda_memcpy_to_host(T* dest, T const* src, size_t n_elements)
{
  auto rc =
    cudaMemcpy(dest, src, sizeof(T) * n_elements, cudaMemcpyDeviceToHost);
  if (rc != cudaSuccess){
        std::cout<<"bad copy" << std::endl;
        fprintf(stderr,
              "cudaCheckError() failed : %s\n",
              cudaGetErrorString(rc));
        abort();

  }
}

template <class T>
T*
cuda_malloc(size_t size)
{
  T* temp;
  auto rc = cudaMalloc((void**)&temp, sizeof(T) * size);
  if (rc != cudaSuccess) {
    std::cout << "Could not allocate " << size << " elements of size:" << sizeof(T) << std::endl;
    fprintf(stderr,
              "cudaCheckError() failed : %s\n",
              cudaGetErrorString(rc));
    throw std::bad_alloc();
  }
  return temp;
}

template <typename T>
void
cuda_memcpy_to_device(T* dest, T* src, size_t size)
{
  auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
  if (rc != cudaSuccess) {
    printf("error in cuda_mempcy_to_device with host src\n");
    fprintf(stderr,
              "cudaCheckError() failed : %s\n",
              cudaGetErrorString(rc));
    abort();
  }
}

template <typename T>
void
cuda_memcpy_to_device(T* dest, const T* src, size_t size)
{
  auto rc = cudaMemcpy(dest, src, sizeof(T) * size, cudaMemcpyHostToDevice);
  if (rc != cudaSuccess) {
    fprintf(stderr,
              "cudaCheckError() failed : %s\n",
              cudaGetErrorString(rc));
    printf("error in cuda_mempcy_to_device with host src\n");
    abort();
  }
}

#endif