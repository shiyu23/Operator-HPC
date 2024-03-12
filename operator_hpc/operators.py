
import os 
import ctypes
import cupy as cp
from functools import wraps

script_path = os.path.dirname(__file__)
so_path = os.path.join(script_path, "./operator_hpc_compiled.so") 
op = ctypes.cdll.LoadLibrary(so_path)


def format_cupy_array(arr, force_type=cp.float32):
    if cp.isfortran (arr):
        with cp.cuda.Device(arr.device.id):
            arr = cp.array(arr, order='C')
    if not arr.dtype == force_type:
        with cp.cuda.Device(arr.device.id):
            arr = arr.astype (force_type)
    return arr


def pre_func_check(arr_index, n_idx=None):
    def check_func(func):
        @wraps (func)
        def wrapper (*args, **kwargs):
            args = list(args)
            for i in arr_index:
                if isinstance(args[i], (list, tuple)):
                    args[i] = type (args[i])([format_cupy_array(arr) for arr in args[i]])
                else:
                    args[i] = format_cupy_array(args[i])
            if n_idx is not None:
                if isinstance(args[n_idx], (int, float)):
                    with cp.cuda.Device(args[arr_index[0]].device.id):
                        args[n_idx] = cp.full((1,), int(args[n_idx]), dtype=cp.int32)
                    kwargs['n_panel'] = 0
                    kwargs['max_n'] = int(args[n_idx])
                else:
                    args[n_idx] = format_cupy_array(args[n_idx], force_type=cp.int32) 
                    kwargs['n_panel'] = 1
                    with cp.cuda.Device(args[n_idx].device.id):
                        kwargs['max_n'] = cp.max(args[n_idx]).item()
            res = func(*tuple(args), **kwargs)
            return res
        return wrapper
    return check_func


@pre_func_check(arr_index=[0], n_idx=1)
def rolling_percentage(X, n, n_panel=0, max_n=1, pct=50.0):
    if max_n > 256:
        raise Exception(f'ERROR: rolling_percentage max_n = {max_n} is greater than 256!')
    
    device = X.device.id
    with cp.cuda.Device(device):
        res = cp.full(X.shape, cp.nan, dtype=cp.float32)
    
    op.rolling_percentage_warp_merge_sort_device_c(ctypes.c_int(device),
                                                   ctypes.c_void_p(res.data.ptr), 
                                                   ctypes.c_void_p(X.data.ptr), 
                                                   ctypes.c_int(X.shape[0]),
                                                   ctypes.c_int(X.shape[1]),
                                                   ctypes.c_void_p(n.data.ptr),
                                                   ctypes.c_int(n_panel),
                                                   ctypes.c_int(max_n),
                                                   ctypes.c_float(pct))
    return res


@pre_func_check(arr_index=[0, 1], n_idx=2)
def rolling_multi_regression(Y, X, n, nx, n_panel=0, max_n=1, rettype=0): 
    device = Y.device.id
    with cp.cuda.Device (device):
        X = cp.vstack(X)
        res = cp.full(Y.shape, cp.nan, dtype=cp.float32)
    
    op.rolling_multi_regression_house_holder_device_c(ctypes.c_int(device),
                                                      ctypes.c_void_p(res.data.ptr), ctypes.c_void_p(X.data.ptr),
                                                      ctypes.c_void_p(Y.data.ptr),
                                                      ctypes.c_int(Y.shape[0]),
                                                      ctypes.c_int(Y.shape[1]),
                                                      ctypes.c_int(nx),
                                                      ctypes.c_void_p(n.data.ptr),
                                                      ctypes.c_int(n_panel), ctypes.c_int(max_n),
                                                      ctypes.c_int(rettype))
    return res


@pre_func_check(arr_index=[0, 1], n_idx=2)
def TS_THEILSEN(Y, X, n, n_panel=0, max_n=1):
    if max_n > 100:
        raise Exception(f'ERROR: rolling_theilsen max_n = {max_n} is greater than 100!')
    
    device = Y.device.id
    with cp.cuda.Device(device):
        res = cp.full(Y.shape, cp.nan, dtype=cp.float32)
    
    op.rolling_theilsen_device_c(ctypes.c_int(device),
                                 ctypes.c_void_p(res.data.ptr),
                                 ctypes.c_void_p(X.data.ptr),
                                 ctypes.c_void_p(Y.data.ptr),
                                 ctypes.c_int(Y.shape[0]),
                                 ctypes.c_int(Y.shape[1]),
                                 ctypes.c_void_p(n.data.ptr),
                                 ctypes.c_int(n_panel),
                                 ctypes.c_int(max_n))
    return res
