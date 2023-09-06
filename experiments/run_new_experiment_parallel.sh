#!/bin/bash

j_val=$1
rep=0

task(){
   sleep 0.5; echo "Hello From Bash Script";
   python3 -m experiments.new_experiment --J=$j_val --repnum=$rep
}

# initialize a semaphore with a given number of tokens
open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
    local x
    # this read waits until there is something to read
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
    )&
}

N=4
open_sem $N
for rep in {0..19}; do
    run_with_lock task $j_val $rep
done 

