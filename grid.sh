commands="$(python _grid.py "$@")"
IFS=$'\n'
commands=($commands)
for cmd in ${commands[@]}
do
    echo $cmd
    #bash run.sh $cmd
done
