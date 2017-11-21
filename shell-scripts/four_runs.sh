
while [ "$1" != "" ]; do
    case $1 in
        -m | --model_type )     shift
                                model_type=$1
                                shift;;
    esac
done
###-s | --seed )           shift
###                        base_seed=$1
###                        shift;;
printf "Launching a job on each GPU with model type %s\n" $model_type
batch_size=100
step_size=0.001
decay=0.94
epochs=50
iters=1
for VARIABLE in 0 1 2 3
do
        let 'seed = VARIABLE + base_seed'
        python train.py -i $iters -b $batch_size --step_size $step_size --decay $decay -e $epochs -g $VARIABLE -s -m $model_type &
        disown %1
        sleep 30

done
printf 'Successfully started all jobs\n'
