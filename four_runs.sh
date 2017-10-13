
while [ "$1" != "" ]; do
    case $1 in
        -i | --iterations )     shift
                                iterations=$1
                                ;;
        -m | --model_type )     model_type=$1
                                ;;
        -s | --seed )           seed=$1
                                ;;
    esac
    shift
done

for VARIABLE in 0 1 2 3
do
        python train.py -g $VARIABLE -s -i $iterations -m $model_type --seed ($VARIABLE*1000 + $seed) &
        disown %1
        sleep 5

done
