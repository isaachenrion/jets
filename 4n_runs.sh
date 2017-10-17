
while [ "$1" != "" ]; do
    case $1 in
        -m | --model_type )     shift
                                model_type=$1
                                shift;;
        -n | --n_copies )       shift
                                N=$1
                                shift;;
    esac
done
let 'total = N * 4'
printf "Launching %s jobs with model type %s, base seed is %s\n" $model_type $total

for SEED {1 .. $N}
do
        ./four_runs.sh -m $model_type -n $N
        disown %1
        sleep 5
done
printf 'Successfully launched all the jobs\n'
