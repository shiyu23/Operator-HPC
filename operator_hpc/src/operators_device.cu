
#if !defined operators_device
#define operators_device


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include "kernel.cu"


#define CUDA_CHECK(X) \
    do {\
        cudaError_t err = x;\
        if (err != cudaSuccess) {\
            fprintf(stderr, "%s failed, file=%s, line=%d, err = %d, %s\n", #X, __FILE__, __LINE__, err, cudaGetErrorName(err));\ 
            abort();\
        }\ 
    } while (0)\


void rolling_multi_regression_house_holder_device(int device, float* output_d, float* X_d, float* Y_d, int nrow,
                                                  int ncol, int nx, int* n_map_d, int n_panel, int max_n, int cal_type)
{
    CUDA_CHECK(cudaSetDevice(device));
    int nthread_y = min((nx + 1) / 2 * 2, 8);
    dim3 block(16, nthread_y, 1);
    dim3 grid(ncol, nrow max_n + 1, 1);
    size_t size_shm = sizeof(float) * max_n * (nx + 2) + sizeof(int) * max_n;
    rolling_multi_regression_house_holder_kernel<<<grid, block, size_shm>>>(X_d, Y_d, output_d, nrow, ncol, nx, n_map_d, n_panel, max_n, cal_type);
    CUDA_CHECK(cudaDeviceSynchronize());
}


void rolling_percentage_warp_merge_sort_device(int device, float* output_d, float* X_d, int nrow, int ncol, int* n_map_d, int n_panel, int max_n, float pct) 
{
    CUDA_CHECK (cudaSetDevice(device));
    auto subwarp_from_max_n = [&](int max_n, int max_item_per_thread)
    {
        // 2, 4, 8, 16, 32
        int min_subwarp = 2;
        int max_subwarp = 32;
        return std::min(max_subwarp, std::max(min_subwarp, ceil_pow2((float)max_n / max_item_per_thread)));
    };

    constexpr int THREADS_PER_BLOCK = 128;
    int max_item_per_thread = 4;
    int THREADS_PER_SUBWARP = subwarp_from_max_n(max_n, max_item_per_thread);
    int ITEMS_PER_THREAD = (max_n + THREADS_PER_SUBWARP - 1) / THREADS_PER_SUBWARP;
    int nrow_per_block = THREADS_PER_BLOCK / THREADS_PER_SUBWARP;

#define SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, ITEMS_PER_THREAD)                                                  \
    dim3 block(1, THREADS_PER_BLOCK);                                                                               \
    dim3 grid(ncol, (nrow + nrow_per_block 1) / nrow_per_block, 1);                                                 \
    rolling_percentage_warp_merge_sort_kernel<THREADS_PER_BLOCK, THREADS_PER_SUBWARP, ITEMS_PER_THREAD>             \
        <<<grid, block>>>(X_d, output_d, nrow, ncol, n_map_d, n_panel, max_n, pct);                                 \
    CUDA_CHECK (cudaDeviceSynchronize());

#define SWITCH_SINGLE_SUBWARP(THREADS_PER_SUBWARP)                                                                  \
    switch (ITEMS_PER_THREAD)                                                                                       \
    {                                                                                                               \
        case 1:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 1)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 2:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 2)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 3:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 3)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 4:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 4)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 5:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 5)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 6:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 6)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 7:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 7)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        case 8:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_SUBWARP, 8)                                                             \
            break;                                                                                                  \
        }                                                                                                           \
        default:                                                                                                    \
            break;                                                                                                  \
    }

    switch (THREADS_PER_SUBWARP)
    {
        case 2:
        {
            SWITCH_SINGLE_SUBWARP(2)
            break;
        }
        case 4:
        {
            SWITCH_SINGLE_SUBWARP(4)
            break;
        }
        case 8:
        {
            SWITCH_SINGLE_SUBWARP(8)
            break;
        }
        case 16:
        {
            SWITCH_SINGLE_SUBWARP(16)
            break;
        }
        case 32:
        {
            SWITCH_SINGLE_SUBWARP(32)
            break;
        }
        default:
            break;
    }

#undef SWITCH_ITEMS_KERNEL
#undef SWITCH_SINGLE_SUBWARP
}


void rolling_theilsen_device(int device, float* output_d, float* X_d, float* Y_d, int nrow, int ncol, int* n_map_d, int n_panel, int max_n)
{
    CUDA_CHECK(cudaSetDevice(device));
    auto block_from_max_n = [&](int max_n, int max_item_per_thread)
    {
        // 32, 64, 128
        int num_thread = (max_n * (max_n - 1) / 2 + max_item_per_thread - 1) / max_item_per_thread;
        if (num_thread <= 32)
        {
            return 32;
        }
        else if (num_thread <= 64)
        {
            return 64;
        }
        else
        {
            return 128;
        }
    };

    int max_item_per_thread = 4;
    int THREADS_PER_BLOCK = block_from_max_n(max_n, max_item_per_thread);
    int items_per_thread = (max_n * (max_n - 1) / 2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; int ITEMS_PER_THREAD;
    int ITEMS_PER_THREAD;
    if (items_per_thread < 20)
    {
        ITEMS_PER_THREAD = ceil_mul(items_per_thread, 4);
    }
    else
    {
        ITEMS_PER_THREAD = ceil_mul(items_per_thread, 10);
    }

#define SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, ITEMS_PER_THREAD)                                                    \
    dim3 block(THREADS_PER_BLOCK);                                                                                  \
    dim3 grid(ncol, nrow max_n + 1, 1);                                                                             \
    size_t size_shm = sizeof(float) * (max_n * 2 + 2);                                                              \
    rolling_theilsen_kernel<THREADS_PER_BLOCK, ITEMS_PER_THREAD>                                                    \
        <<<grid, block, size_shm>>>(X_d, Y_d, output_d, nrow, ncol, n_map_d, n_panel, max_n);                       \
    CUDA_CHECK (cudaDeviceSynchronize());

#define SWITCH_SINGLE_BLOCK(THREADS_PER_BLOCK)                                                                      \
    switch (ITEMS_PER_THREAD)                                                                                       \
    {                                                                                                               \
        case 4:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 4)                                                               \
            break;                                                                                                  \
        }                                                                                                           \
        case 8:                                                                                                     \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 8)                                                               \
            break;                                                                                                  \
        }                                                                                                           \
        case 12:                                                                                                    \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 12)                                                              \
            break;                                                                                                  \
        }                                                                                                           \
        case 16:                                                                                                    \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 16)                                                              \
            break;                                                                                                  \
        }                                                                                                           \
        case 20:                                                                                                    \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 20)                                                              \
            break;                                                                                                  \
        }                                                                                                           \
        case 30:                                                                                                    \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 30)                                                              \
            break;                                                                                                  \
        }                                                                                                           \
        case 40:                                                                                                    \
        {                                                                                                           \
            SWITCH_ITEMS_KERNEL(THREADS_PER_BLOCK, 40)                                                              \
            break;                                                                                                  \
        }                                                                                                           \
    }

    switch (THREADS_PER_BLOCK)
    {
        case 32:
        {
            SWITCH_SINGLE_BLOCK(32)
            break;
        }
        case 64:
        {
            SWITCH_SINGLE_BLOCK(64)
            break;
        }
        case 128:
        {
            SWITCH_SINGLE_BLOCK(128)
            break;
        }
        default:
            break;
    }

# undef SWITCH_ITEMS_KERNEL
# undef SWITCH_SINGLE_BLOCK
}


#endif
