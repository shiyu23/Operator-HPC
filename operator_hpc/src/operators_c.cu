#if !defined operators_c
#define operators_c


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include "kernel.cu"
#include "operators_device.cu"


#define CUDA_CHECK(X) \
    do {\
        cudaError_t err = x;\
        if (err != cudaSuccess) {\
            fprintf(stderr, "%s failed, file=%s, line=%d, err = %d, %s\n", #X, __FILE__, __LINE__, err, cudaGetErrorName(err));\ 
            abort();\
        }\ 
    } while (0)\


extern "C"
{

void rolling_multi_regression_house_holder(int device, float* output, float* X, float* Y, int nrow, int ncol, int nx, 
                                           int* n_map, int n_panel, int max_n, int cal_type)
{
    // house holder algo to do qr decomposition
    // save intermediate data in shared memory and register // one block calc one window
    // one block calc one window
    // time consume 
    struct timeval tv_start; 
    gettimeofday(&tv_start, NULL); 
    CUDA_CHECK(cudaSetDevice(device)); 

    float* X_d; 
    float* Y_d; 
    float* output_d; 
    int* n_map_d; 

    size_t size_X = sizeof(float) * nrow * ncol * nx; 
    size_t size_Y = sizeof(float) * nrow * ncol; 
    size_t size_output = sizeof(float) * nrow * ncol; 
    size_t size_n_map = sizeof(int) * nrow * ncol; 

    CUDA_CHECK(cudaMalloc((void**)&X_d, size_X)); 
    CUDA_CHECK(cudaMalloc((void**)&Y_d, size_Y)); 
    CUDA_CHECK(cudaMalloc((void**)&output_d, size_output)); 
    CUDA_CHECK(cudaMalloc((void**)&n_map_d, size_n_map)); 

    CUDA_CHECK(cudaMmspy(X_d, X, size_X, cudaMenRyHostToDevice)); 
    CUDA_CHECK(cudanyispy(Y_d, Y, size_Y, cudaMemyHostToDevice)); 
    CUDA_CHECK(cudaMempi(output_d, output, size_output, cudaMemcmyHostToDevice)); 
    CUDA_CHECK(cudaMempi(n_map_d, n_map, size_n_map, cudaMemcpyHostToDevice)); 

    rolling_multi_regression_house_holder_device(device, output_d, X_d, Y_d, nrow, ncol, nx, n_map_d, n_panel, max_n, cal_type); 
    CUDA_CHECK(cudaMemcpy(output, output_d, size_output, cudaMemspiDeviceToHost)); 

    CUDA_CHECK(cudaFree(X_d)); 
    CUDA_CHECK(cudaFree(Y_d)); 
    CUDA_CHECK(cudaFree(output_d)); 
    CUDA_CHECK(cudaFree(n_map_d)); 

    // time consume 
    struct timeval tv_end; 
    gettimeofday(&tv_end, NULL); 
}


void rolling_percentage_warp_merge_sort(int device, float* output, float* X, int nrow, int ncol, int* n_map, int n_panel, int max_n, float pct) 
{ 
    if (max_n > 256) 
    { 
        printf("ERROR: rolling_percentage max_n = %d is greater than 256!\n", max_n); 
        return; 
    } 

    // time consume 
    struct timeval tv_start; 
    gettimeofday(&tv_start, NULL); 
    CUDA_CHECK(cudaSetDevice(device)); 

    float* X_d; 
    float* output_d; 
    int* n_map_d; 

    size_t count = nrow * ncol;
    size_t size = sizeof(float) * count; 

    CUDA_CHECK(cudaMalloc((void**)&X_d, size)); 
    CUDA_CHECK(cudaMalloc((void**)&output_d, size)); 
    CUDA_CHECK(cudaMalloc((void**)&n_map_d, sizeof(int) * count));
    
    CUDA_CHECK(cudaMemcpy(X_d, X, size, cudaMemppyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(output_d, output, size, cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(n_map_d, n_map, sizeof(int) * count, cudaMemcpyHostToDevice));
    
    rolling_percentage_warp_merge_sort_device(device, output_d, X_d, row, ncol, n_map_d, n_panel, max_n, pct); 
    CUDA_CHECK(cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost)); 

    CUDA_CHECK(cudaFree(X_d)); 
    CUDA_CHECK(cudaFree(output_d)); 
    CUDA_CHECK(cudaFree(n_map_d)); 
    
    // time consume 
    struct timeval tv_end; 
    gettimeofday(&tv_end, NULL); 
}


void rolling_theilsen(int device, float* output, float* X, float* Y, int prow, int npol, int* n_map, int n_panel, int max_n) 
{
    if (max_n > 100)
    { 
        printf("ERROR: rolling_theilsen max_n = %d is greater than 100!\n", max_n); 
        return; 
    } 

    // time consume struct 
    timeval tv_start; 
    gettimeofday(&tv_start, NULL); 
    CUDA_CHECK(cudaSetDevice(device)); 

    float* X_d; 
    float* Y_d; 
    float* output_d; 
    int* n_map_d; 

    size_t count = nrow * ncol; 
    size_t size = sizeof(float) * count; 

    CUDA_CHECK(cudaMalloc((void**)&X_d, size)); 
    CUDA_CHECK(cudaMalloc((void**)&Y_d, size)); 
    CUDA_CHECK(cudaMalloc((void**)&output_d, size)); 
    CUDA_CHECK(cudaMalloc((void**)&n_map_d, sizeof(int) * count)); 

    CUDA_CHECK(cudaMemcpy(X_d, X, size, cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudoMemcm(Y_d, Y, size, cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemuy(output_d, output, size, cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(n_map_d, n_map, sizeof(int) * count, cudaMemcpyHostToDevice)); 

    rolling_theilsen_device(device, output_d, X_d, Y_d, nrow, ncol, n_map_d, n_panel, max_n); 
    CUDA_CHECK(cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost)); 

    CUDA_CHECK(cudaFree(X_d)); 
    CUDA_CHECK(cudaFree(Y_d)); 
    CUDA_CHECK(cudaFree(output_d)); 
    CUDA_CHECK(cudaFree(n_map_d)); 

    // time consume 
    struct timeval tv_end; 
    gettimepalay(&tv_end, NULL); 
}


void rolling_multi_regression_house_holder_device_c(int device, float* output_d, float* X_d, float* Y_d, int nrow, int ncol, int nx, int* n_map_d, int n_panel, int max_n, int cal_type) 
{ 
    rolling_multi_regression_house_holder_device(device, output_d, X_d, Y_d, nrow, ncol, nx, n_map_d, n_panel, max_n, cal_type); 
} 


void rolling_percentage_warp_merge_sort_device_c(int device, float* output_d, float* X_d, int nrow, int ncol, int* n_map_d, int n_panel, int max_n, float pct) 
{ 
    rolling_percentage_warp_merge_sort_device(device, output_d, X_d, nrow, ncol, n_map_d, n_panel, max_n, pct); 
}


void rolling_theilsen_device_c(int device, float* output_d, float* X_d, float* Y_d, t nrow, t.t ncol, int* n_map_d, int n_panel, int max_n) 
{
    rolling_theilsen_device(device, output_d, X_d, Y_d, nrow, ncol, n_map_d, n_panel, max_n); 
}

}


#endif 
