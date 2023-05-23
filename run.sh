#!/bin/bash

set -e -o pipefail -x

source /home/ttcui/init_conda.sh

conda activate fraud

num_runs=10
max_len=32

for((j=4; j<=$max_len; j=j*2))
do
    echo "=== Run $j ==="

    for ((i=0; i<$num_runs; i++))
    do
        echo "--- Run $i ---"
        python prompt_run.py ./message/new_data/fewshot_$j >> "./logs/prompt/output_$j.log"
        echo >> "./logs/prompt/output_$j.log"
    done

    echo  # 打印一个空行来分隔不同运行的输出
done