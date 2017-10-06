def main():

    d = [10, 20, 50, 35, 5, 100]
    n = len(d)
    A = [None] * (n+1)
    def max_theft(d, A, j):
        if j == 0:
            A[0] = 0
        elif j == 1:
            A[1] = d[0]
        else:
            if A[j] is None:
                if A[j-1] is None:
                	A[j-1] = max_theft(d, A, j-1);
                if A[j-2] is None:
                	A[j-2]= max_theft(d, A, j-2);
                A[j]= max(A[j-1], A[j-2] + d[j-1])
                print('Max theft using cars from car 1 to car {}: ${}'.format(j, A[j]))
        return A[j]

    max_theft(d, A, n)

    # postprocessing
    L = []
    i = n
    while i >= 1:
        if A[i] == A[i-1]:
            i = i - 1
        else:
            L.append(i)
            i = i - 2
    print('Best cars to rob are as follows:')
    for j in reversed(L):
        print('Car {} containing ${}'.format(j, d[j-1]))

if __name__ == '__main__':
    main()
