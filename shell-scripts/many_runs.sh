
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
printf "Launching %s jobs with model type %s\n" $total $model_type
COUNTER=0
while [  $COUNTER -lt $N ];
do
  let 'SEED = COUNTER * 10000'
  ./four_runs.sh -m $model_type -s $SEED
  sleep 2
  let COUNTER=COUNTER+1
done

printf 'Successfully launched all the jobs\n'
