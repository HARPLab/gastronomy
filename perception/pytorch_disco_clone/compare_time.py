import torch
import numpy as np
import time

numbers = 16*16*16*32

val = np.random.randn(1000,numbers)
init_time = time.time()
val =  np.dot(val,val.transpose())
print(time.time() - init_time)
print("time taken by numpy")
val = torch.from_numpy(val).cuda()
init_time = time.time()
val =  torch.matmul(val,val.transpose(1,0))
print(time.time() - init_time)
print("time taken by cuda")
