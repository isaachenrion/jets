commands="$(python $SRCDIR/src/scripts/_grid.py "$@")"
IFS=$'\n' read -r -d '' -a arr < <(printf '%s\0' "$commands")
IFS=$'\n'
for cmd in ${commands[@]}
do
    bash $SRCDIR/sh/train.sh $cmd
done
