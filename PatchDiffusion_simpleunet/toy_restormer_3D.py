import numpy as np
import torch
import training.networks as net

restormer3d=net.Restormer3D()

intensor= torch.zeros((1,3,64,64,64))

out=restormer3d(intensor)
print (out.shape)
