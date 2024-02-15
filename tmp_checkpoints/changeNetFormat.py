import numpy as np
import torch

name = "full_2000"

a = torch.load(name + ".pt")["actor_architecture_state_dict"]

l = []
for k in a.keys():
    l += a[k].T.flatten().tolist()

f = open(name + ".txt", "w")
f.write(str(l)[1:-1])
f.close()

l = {}
for k in a.keys():
    l[k] = a[k].cpu().numpy()
np.save(name + ".npy", l)
