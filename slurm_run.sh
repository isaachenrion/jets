
while [ "$1" != "" ]; do
    case $1 in
        -m | --model_type )     shift
                                model_type=$1
                                shift;;
        -n | --n_copies )       shift
                                N=$1
                                shift;;
        -d | --data_dir )       shift
                                DATA_DIR=$1
                                shift;;
    esac
done
batch_size=64
step_size=0.001
decay=0.912
n_epochs=25
COUNTER=0
printf 'N = %s, Model = %s, DATA_DIR = %s\n' $N $model_type $DATA_DIR
while [  $COUNTER -lt $N ];
do
  let 'SEED = COUNTER * 10000'
  printf 'Running with seed = %s\n' $SEED
  python train.py --debug --data_dir $DATA_DIR -b $batch_size --step_size $step_size --decay $decay -e $n_epochs -s -m $model_type --seed $seed &
  disown %1
  sleep 1
  let COUNTER=COUNTER+1
done
