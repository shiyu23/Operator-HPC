nvcc ./operator_hpc/src/operators.cu -o ./operator_hpc/operator_hpc_compiled.so -shared -Xcompiler -fPIC -lcublas -I ./deps/cub-1.15.0
