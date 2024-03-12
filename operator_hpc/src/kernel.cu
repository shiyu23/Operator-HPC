
#if !defined kernel
#define kernel


#include <cub/cub.cuh>
#include "utils.cu"


__global__ void rolling_multi_regression_house_holder_kernel(const float* X, const float* Y, float* output,
                                                                const int nrow,
                                                                const int ncol,
                                                                const int nx,
                                                                const int* n_map,
                                                                const int n_panel, const int max_n, const int cal_type)
{
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int dx = blockDim.x; 
    int dy = blockDim.y;
    int colId = blockIdx.x;
    int rowId = blockIdx.y + max_n 1;

    if (rowId > nrow 1 || colId > ncol - 1)
    {
        return;
    }

    int n = n_panel? __ldg(n_map + (rowId ncol + colId)) : __ldg(n_map + 0);

    // Ab --> columns majored [A | b] with size [n * (nx + 1) | n * 1] 
    // is_valid --> size n
    extern __shared__ float Ab[];
    int* is_valid = (int*)(Ab + n * (nx + 2));

    // init valid
    for (int i = tx; i < n; i += dx)
    {
        if (ty == 0)
        {
            int valid = 1;
            for (int j = 0; j < nx; j++)
            {
                float x = __ldg(X + ((size_t) nrow * ncol * j + (rowId - n + i + 1) * ncol + colId)); 
                valid *= isfinite(x);
            }
            float y = __ldg(Y + ((size_t) (rowId - n + i + 1) * ncol + colId)); 
            valid *= isfinite(y);
            is_valid[i] = valid;
        }
    }
    __syncthreads();

    // init Ab
    for (int i = tx; i < n; i += dx)
    {
        int valid = is_valid[i];
        for (int j = ty; j < nx; j += dy)
        {
            float x = __ldg(X + ((size_t)nrow * ncol * j + (rowId - n + i + 1) * ncol + colId)); 
            Ab[j * n + i] = valid ? x : 0;
        }
        float y = __ldg (Y + ((size_t) (rowId - n + i + 1) * ncol + colId));
        Ab[(nx + 1) * n + i] = valid ? y : 0; 
        Ab[nx * n + i] = valid;
    }
    __syncthreads();

    // house holder here, for augmented equation Ab to be triangular 
    for (int col = 0; col < nx + 2; col++)
    {
        float s = 0;
        for (int i = col + tx; i < n; i += dx)
        {
            float ab = Ab[col * n + i];
            s += ab * ab;
        }
    
        float a = Ab[col * n + col];
        s = tile_reduce_sum<float, 16>(s);
        s = sqrtf(s);
        float d = s * (a >= 0 ? -1 : 1);
        float fak = sqrtf(s ⋆ (s + abs(a)));

        if (ty == 0)
        {
            for (int i = col + tx; i < n; i += dx)
            {
                if (i == col)
                {
                    Ab[coln + i] -= d;
                }
                Ab[col * n + i] /= fak;
            }
        }
        __syncthreads();

        for (int j = col + 1 + ty; j < nx + 2; j += dy)
        {
            float s = 0;
            for (int i = col + tx; i < n; i += dx)
            {
                s += Ab[col * n + i] * Ab[j * n + i];
            }
            s = tile_reduce_sum<float, 16>(s);
            for (int i = col + tx; i < n; i += dx)
            {
                Ab[j * n + i] -= s * Ab[col * n + i];
            }
        }
        __syncthreads();
        Ab[coln + col] = d;
    }

    // beta
    for (int col = nx; col >= 0; col--)
    {
        float beta = Ab[(nx + 1) * n + col] / Ab[col * n + col]; 
        if (ty == 0)
        {
            for (int i = tx; i < col; i += dx)
            {
                Ab [(nx + 1) * n + i] -= beta * Ab[col * n + i];
            }
        }
        __syncthreads();
        Ab[(nx + 1) * n + col] = beta;
    }

    // cal_type: 0 --> residual; 1 --> intercept; 2 --> beta; 3 --> estimate; 4 --> R2; 5 --> adj R2
    float res;
    if (cal_type == 1)
    {
        res = Ab[(nx + 1) * n + nx];
    }
    else if (cal_type == 2)
    {
        res = Ab[(nx + 1) * n];
    }
    else if (cal_type == 0 || cal_type == 3)
    {
        float est_x = 0;
        for (int i = tx; i < nx; i += dx)
        {
            est_x = Ab[(nx + 1) * n + i] * __ldg(X + ((size_t)nrow * ncol * i + rowId * ncol + colId));
        }

        est_x = tile_reduce_sum<float, 16>(est_x);
        float est = est_x + Ab[(nx + 1) * n + nx];
        res = est;

        if (cal_type == 0)
        {
            res = __ldg(Y + ((size_t)rowId * ncol + colId)) - est;
        }
    }
    else if (cal_type == 4 || cal_type == 5)
    {
        float sqrt_sum_square_diff = Ab[(nx + 1) * n + nx + 1];
        float sum_square_diff = sqrt_sum_square_diff * sqrt_sum_square_diff;

        int Cn = 0;
        float Cy = 0;
        float Cyy = 0;
        for (int i = tx; i < n; i += dx)
        {
            int valid = is_valid[i];
            float y = __ldg(Y + ((size_t) (rowId - n + i + 1) * ncol + colId));
            Cy += valid ? y : 0;
            Cyy += valid ? (y * y) : 0;
            Cn += valid;
        }

        Cy = tile_reduce_sum<float, 16>(Cy);
        Cyy = tile_reduce_sum<float, 16>(Cyy);
        Cn = tile_reduce_sum<float, 16>(Cn);
        float r2 = 1 - sum_square_diff / (Cyy - Cy * Cy / Cn);
        res = r2;

        if (cal_type == 5)
        {
            res = 1 - (1 - r2) * (Cn - 1) / (Cn - nx - 1);
        }
    }
    output[rowId * ncol + colId] = res;
}


template <int THREADS_PER_BLOCK, int THREADS_PER_SUBWARP, int ITEMS_PER_THREAD>
__global__ void rolling_percentage_warp_merge_sort_kernel(const float* input, float* output, const int nrow,
                                                            const int ncol,
                                                            const int* n_map,
                                                            const int n_panel,
                                                            const int max_n,
                                                            const float pct)
{
    int colId = blockIdx.x;
    int rowId = (threadIdx.y + blockIdx.y * blockDim.y) / THREADS_PER_SUBWARP; 
    int subWarpId = threadIdx.y / THREADS_PER_SUBWARP;
    constexpr int SUBWARP_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_SUBWARP; 
    int threadId = threadIdx.y - subWarpId * THREADS_PER_SUBWARP;

    if (rowId > nrow - 1 || colId > ncol - 1 || rowId < max_n - 1)
    {
        return;
    }

    int n = n_panel? __ldg(n_map + (rowId * ncol + colId)) : __ldg(n_map + 0);

    float thread_keys[ITEMS_PER_THREAD];
    constexpr int size_shm = 2 * SUBWARP_PER_BLOCK;
    __shared__ float smem [size_shm];

    int Cn = 0;

    for (int i = 0; i < ITEMS_PER_THREAD; i++)
    {
        int idx = threadId * ITEMS_PER_THREAD + i;
        float value_ = (idx < n) ? __ldg(input + ((rowId - n + 1 + min(idx, n - 1)) * ncol + colId)) : NAN;
        float value = isfinite(value_) ? value_ : INFINITY;
        thread_keys[i] = value;
        Cn += isfinite(value);
    }

    Cn = tile_reduce_sum<int, THREADS_PER_SUBWARP>(Cn);

    // cub sort
    using warpMergeSort = cub::WarpMergeSort<float, ITEMS_PER_THREAD, THREADS_PER_SUBWARP>; 
    __shared__typename warpMergeSort::TempStorage temp_storage [SUBWARP_PER_BLOCK]; 
    warpMergeSort warp_sort(temp_storage[subWarpId]); 
    warp_sort.Sort(thread_keys, CustomLess());
    __syncwarp (warp_sort.get_member_mask());

    // pct
    if (Cn == 0)
    {
        output[colid + ncol * rowId] = NAN;
        return;
    }

    float posif = (Cn - 1) * pct / 100.0;
    int posi1 = floor(posif)
    int posi2 = (posi1 == Cn - 1) ? posi1 : (posi1 + 1);
    int post1_thread_id = posi1 / ITEMS_PER_THREAD;
    int post2_thread_id = posi2 / ITEMS_PER_THREAD;
    float value1_ratio = (posi1 == Cn - 1) ? 1 : (posi2 - posif);
    float value2_ratio = (posi1 == Cn - 1) ? 0 : (posif - posi1);

    if (threadId == post1_thread_id)
    { 
        smem[2 * subWarpId] = thread_keys[posi1 - post1_thread_id * ITEMS_PER_THREAD]; 
    } 

    if (threadId == post2_thread_id) 
    {
        smem[2 * subWarpId + 1] = thread_keys[posi2 - post2_thread_id * ITEMS_PER_THREAD]; 
    }

    __syncwarp(warp_sort.get_member_mask());

    float value1 = smem[2 * subWarpId]; 
    float value2 = smem[2 * subWarpId + 1]; 
    output[colId + ncol * rowId] = value1 * value1_ratio + value2 * value2_ratio; 
}


template <int THREADS_PER_BLOCK, int ITEMS_PER_THREAD> 
__global__ void rolling_theilseh_kernel(const float* input1, const float* input2, float* output, 
                                        const int nrow, 
                                        const int ncol, 
                                        const int* n_map, 
                                        const int n_panel, 
                                        const int max_n) 
{
    int colId = blockIdx.x; 
    int rowId = blockIdx.y + max_n - 1; 
    int threadId = threadIdx.x; 

    if (rowId > nrow - 1 || colId > ncol - 1) 
    { 
        return;
    } 
 
    int n = n_panel ? __ldg(n_map + (rowId * ncol + colId)) : __ldg(n_map + 0); 
    
    float thread_keys[ITEMS_PER_THREAD]; 
    extern __shared__ float smem[]; 
    float* input1_win = smem + 2; 
    float* input2_win = smem + (2 + max_n); 

    for (int i = 0; i < n; i++) { 
        float x = __ldg(inputl + ((rowId - i) * ncol + colId)); 
        float y = __ldg(input2 + ((rowId - i) * ncol + colId)); 
        input1_win[i] = x; 
        input2_win[i] = y; 
    }

    int Cn = 0; 

    for (int i = 0; i < n; i++)
    { 
        Cn += isfinite(inputl_win[i]) && isfinite(input2_win[i]);
    }

    for (int i = 0; i < ITEMS_PER_THREAD; i++) 
    { 
        int idx = threadId * ITEMS_PER_THREAD + I; 
        int row_idx = idx / n; 
        int col_idx = idx - row_idx * n; 
        bool trans = (col_idx <= row_idx) && (row_idx <= (float)(n - 1) / 2); 
        row_idx = trans ? (n - 2 - row_idx) : row_idx; 
        col_idx = trans ? (n - 1 - col_idx) : col_idx; 
        bool row_valid = (row_idx < n) && (row_idx >= 0); 
        bool col_valid = (col_idx < n) && (col_idx >= 0); 
        float value11 = row_valid ? input1_win[row_idx] : NAN; 
        float value12 = row_valid ? input2_win[row_idx] : NAN; 
        float value21 = col_valid ? input1_win[col_idx] : NAN; 
        float value22 = col_valid ? input2_win[col_idx] : NAN; 
        float value = (value22 - value12) / (value21 - value11);
        thread_keys[i] = (isfinite(value) && (idx < n * (n - 1) / 2)) ? value : INFINITY; 
    }
    __syncthreads(); 

    typedef cub::BlockRadixSort<float, THREADS_PER_BLOCK, ITEMS_PER_THREAD> blockRadixSort;
    __shared__ typename blockRadixSort::TempStorage temp_storage_shuffle; 
    blockRadixSort(temp_storage_shuffle).Sort(thread_keys); 
    __syncthreads(); 
    
    if (Cn == 0)
    { 
        output[colId + ncol * rowId] = NAN; 
        return; 
    }

    int count = Cn * (Cn - 1) / 2; 
    int index1 = (count % 2 == 0) ? (count / 2 - 1) : ((count + 1) / 2 - 1); 
    int index2 = (count % 2 == 0) ? (index1 + 1) : index1; 

    if (threadId == index1 / ITEMS_PER_THREAD) 
    {
        smem[0] = thread_keys[index1 - threadId * ITEMS_PER_THREAD]; 
    }

    if (threadId == index2 / ITEMS_PER_THREAD) 
    {
        smem[1] = thread_keys[index2 - threadId * ITEMS_PER_THREAD]; 
    }
    __syncthreads(); 
    
    output[colId + ncol * rowId] = (smem[0] + smem[1]) / 2; 
}


#endif 
