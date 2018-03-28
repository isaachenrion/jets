commands="$(python _grid.py "$@")"
IFS=$'\n' read -r -d '' -a arr < <(printf '%s\0' "$commands")
IFS=$'\n'

for cmd in ${commands[@]}
do
    bash run-eval.sh $cmd
done
