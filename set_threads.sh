#!/usr/bin/env bash

# Detect number of CPUs (fallback to 1 if nproc fails)
num_procs=$(nproc 2>/dev/null || echo 1)

# Decide the limit: if num_procs < 128, use num_procs. Else use 128.
if [ "$num_procs" -lt 128 ]; then
    thread_limit=$num_procs
else
    thread_limit=128
fi

export OPENBLAS_NUM_THREADS=$thread_limit
export OMP_NUM_THREADS=$thread_limit
export BLAS_NUM_THREADS=$thread_limit

echo "Detected $num_procs CPUs; setting thread limit = $thread_limit"