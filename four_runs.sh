
while [ "$1" != "" ]; do
    case $1 in
        -m | --model_type )     shift
                                model_type=$1
                                ;;
    esac
    shift
done

for VARIABLE in 0 1 2 3
do
        python train.py -g $VARIABLE -s -m $model_type --seed $VARIABLE &
        disown %1
        sleep 5

done
