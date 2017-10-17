
while [ "$1" != "" ]; do
    case $1 in
        -m | --model_type )     shift
                                model_type=$1
                                shift;;
        -s | --seed )           shift
                                base_seed=$1
                                shift;;
    esac
done
printf "Launching a job on each GPU with model type %s, base seed is %s\n" $model_type $base_seed
batch_size=64
step_size=0.001
decay=0.912
n_epochs=25
for VARIABLE in 0 1 2 3
do
        let 'seed = VARIABLE + base_seed'
        python train.py -b $batch_size --step_size $step_size --decay $decay -e $n_epochs --leaves -g $VARIABLE -s -m $model_type --seed $seed &
        disown %1
        sleep 10

done
printf 'Successfully started all jobs\n'
