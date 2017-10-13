for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do
        python train.py -g 2 -s -m 1 --seed $VARIABLE &
        disown %1
        sleep 5
        python train.py -g 3 -s -m 2 --seed $VARIABLE &
        disown %1
        sleep 5
done
