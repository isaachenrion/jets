
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
        python train.py -p -s -m 3 -i 1 -g 0 & disown %1
        sleep 20

done
printf 'Successfully started all jobs\n'
