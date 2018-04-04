import time
import torch
import numpy
torch.set_default_tensor_type("torch.FloatTensor")

w = 5000
h = 40000
is_cuda = torch.cuda.is_available()
start = time.time()

a = torch.rand(w,h)
b = torch.rand(h,w)
a_np = a.numpy()
b_np = b.numpy()
if is_cuda:
    a_cu = a.cuda()
    b_cu = b.cuda()

allocation = time.time()
print("Allocation ", allocation - start)

c = a.mm(b)
th_blas = time.time()
print("Torch Blas ", th_blas - allocation)

c = a_np.dot(b_np)
np_blas = time.time()
print("Numpy Blas ", np_blas - th_blas)

if is_cuda:
    c = a_cu.mm(b_cu)
    torch.cuda.synchronize()
    cu_blas = time.time()
    print("Torch cuBlas ", cu_blas - np_blas)

print("Final", time.time() - start)
