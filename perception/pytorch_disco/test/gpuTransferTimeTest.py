import torch
import numpy as np 
import time

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')

a=torch.from_numpy(np.random.randint(0,10,(100,24,32,16,16,16)))
b=torch.from_numpy(np.random.randint(0,10,(100,24,32,16,16,16)))
a=a.to(cuda1)
# b=b.to(cuda0)
print("checking time")
start = time.time()
a=a.to(cuda0)
end1 = time.time()
print(end1 - start)
a=a.to(cuda1)
b=b.to(cuda1)
end2 = time.time()
print(end2 - end1)
a=a.to(cuda0)
end3 = time.time()
print(end3 - end2)

end = time.time()
print(end - start)