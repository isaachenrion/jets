
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
printf "Launching %s jobs with model type %s" $total $model_type

for i {1 .. $N}
do
        let 'SEED = i * 10000'
        ./four_runs.sh -m $model_type -s $SEED
        disown %1
        sleep 5
done
printf 'Successfully launched all the jobs\n'
