commands="$(python ../src/scripts/_grid.py "$@")"
IFS=$'\n' read -r -d '' -a arr < <(printf '%s\0' "$commands")
IFS=$'\n'
#commands=("$commands")
#echo ${commands[@]}
for cmd in ${commands[@]}
do
    #echo ""
    #echo $cmd
    #cmd=( "$cmd" )
    bash run.sh $cmd
done
