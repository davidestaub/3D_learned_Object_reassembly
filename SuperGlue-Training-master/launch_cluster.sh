#!/bin/bash

#output="-oo $(pwd)/out.txt"
cores="20"
memory="4000"  # MB per core
scratch="0"  # MB per core
gpus="1"
clock="4:00"  # time limit: 4:00, 24:00, or 120:00
gpu_memory="10000"  # minimum GPU memory
gpu_model="any"
#gpu_model="TeslaV100_SXM2_32GB"
gpu="'select[gpu_mtotal0>=$gpu_memory]'"  # ,gpu_model0==$gpu_model
#gpu="volta"
warn="-wt 10 -wa INT"  # interrupt signal 10 min before timeout

cmd="bsub
    -n $cores
    -W $clock $output
    $warn
    -R 'select[gpu_mtotal0>=$gpu_memory] rusage[mem=$memory,scratch=$scratch,ngpus_excl_p=$gpus]'
    $*"
echo $cmd
eval $cmd

# https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs
