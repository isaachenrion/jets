with open('relnn.txt', 'r') as f:
    lines = f.readlines()
    newlines = []
    for l in lines:
        if 'took' not in l:
            newlines.append(l)
with open('newrelnn.txt', 'w') as f:
    for l in newlines:
        f.write(l)
