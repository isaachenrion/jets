import os
import pickle
import numpy as np
import sys

def create_numpy_arrays(jets):
    batch_outers = []

    for j, jet in enumerate(jets):
        k = 1000
        if j % k == 0:
            print('Processing jets {}-{}'.format(j+1, j+k))
        tree = np.copy(jet["tree"])
        inners = []   # Inner nodes at level i
        outers = []   # Outer nodes at level i

        queue = [(jet["root_id"], -1)]

        while len(queue) > 0:
            node, parent = queue.pop(0)
            # Inner node
            if tree[node, 0] != -1:
                inners.append(node)
                queue.append((tree[node, 0], node))
                queue.append((tree[node, 1], node))

            # Outer node
            else:
                outers.append(node)
        batch_outers.append(outers)
    jet_contents = [jet['content'] for jet in jets]
    leaves = [np.stack([jet[i] for i in outers], 0) for jet, outers in zip(jet_contents, batch_outers)]
    return leaves

def main(train=True):
    if train:
        in_path = 'dij-kt_antikt-kt-train.pickle'
        out_path = 'dij-kt_antikt-kt-train'
    else:
        in_path = 'dij-kt_antikt-kt-test.pickle'
        out_path = 'dij-kt_antikt-kt-test'

    with open(os.path.join('data', in_path), mode="rb") as f:
        X, y, dij = pickle.load(f, encoding='latin-1')
        y = np.array(y)
        print('Loaded pickles')
        #print(len(X))
        #print(len(y))
        #print(len(dij))
        #import ipdb; ipdb.set_trace()

    #X = create_numpy_arrays(X)
    #print('Converted to numpy arrays, length = {}'.format(len(X)))
    chunks = 10
    offset = 0

    chunk_size = int(len(X) / chunks)
    for i in range(chunks):
        with open(os.path.join('data', '{}-{}.pickle'.format(out_path, i+1)), mode='wb') as f:
            pickle.dump((X[offset:offset+chunk_size], y[offset:offset+chunk_size], dij[offset:offset+chunk_size]), f)
        offset += chunk_size
        print('Created pickle {}'.format(i+1))
    #print(sum(x.nbytes for x in X))
    ##print(sum(x.nbytes for x in y))
    #print(sum(x.nbytes for x in dij))
    #import ipdb; ipdb.set_trace()


    print('Stored in new pickles')

if __name__ == '__main__':
    main()
