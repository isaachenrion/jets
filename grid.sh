commands="$(python _grid.py "$@")"
IFS=$'\n'
commands=($commands)
for cmd in ${commands[@]}
do
    echo "$cmd"
    echo ""
done

for cmd in ${commands[@]}
do
    bash run.sh $cmd
    #echo bash run.sh $cmd
done
