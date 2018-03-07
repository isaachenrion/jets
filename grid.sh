

function split() {
    #statements
    str=$1
    delimiter=$2
    s=$str$delimiter
    array=();
    while [[ $s ]]; do
        new=( "${s%%"$delimiter"*}" );
        echo $new;
        array+=$new;
        s=${s#*"$delimiter"};
        #echo $s;
        #echo ${array[2]}
    done;
    echo "${array[@]}"
}

S=" --slurm --gpu=0 --slurm_array_job_id=4790119 --slurm_array_task_id=9 -a=[dm,phy,eye] --data_dir=/scratch/ih692/data"
#S="5,6,7"
array=($S)

for i in ${array[@]}
do
    kv=(${i//=/ })
    k=${kv[0]}
    v=${kv[1]}
    echo "$k $v"
done

function join { local IFS="$1"; shift; echo "$*"; }
#A=( 1 a 2 b c d e 3 )
#echo $A
OUT=$(join "\\" "${array[@]}" )

echo $OUT
