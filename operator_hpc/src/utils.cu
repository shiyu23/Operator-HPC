#if !defined utils
#define utils 


template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val)
{
    for (int i = 1; i < 32; i <<= 1)
    {
        val += __shfl_xor_sync(__activemask(), val, i);
    }
    return val;
}


template <typename T, int SIZE_OF_TILE>
__device__ __forceinline__ T tile_reduce_sum(T val)
{
    for (int i = 1; i < SIZE_OF_TILE; i <<= 1)
    {
        val += __shfl_xor_sync(__activemask(), val, i);
    }
    return val;
}


struct CustomLess
{
    template <typename T>
    __device__ bool operator() (const T &lhs, const T &rhs)
    {
        return lhs < rhs;
    }
};


template <typename T>
__inline__ int ceil_mul(const T value, const int mul)
{
    return (int)std::ceil((double)value / mul) * mul;
}


template <typename T>
__inline__ int ceil_pow2(const T value)
{
    return 1 << (int)std::ceil(log2((double)value));
}


#endif
