import time
import torch

linear = torch.nn.Linear(100,200)
linear.cuda()

for k in range(15):
  time.sleep(1)
  print('{} seconds'.format(k+1))
