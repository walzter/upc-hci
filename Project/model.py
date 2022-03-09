# imports 
import torch 
from torch import nn, optim


'''
Preprocessing -> scaling to tanh activation [-1, 1]
Mini-batch SGD size 128 
Weight_init = normal, std = 0.02
LeakyReLu -> leak to 0.2 
Adam; lr = 0.0002, beta_1 = 0.5
'''

## encoder 
'''

input -> 100 x 100 
layer 1: 4x4x1024 
layer 2: 8x8x512 stride 2, filters 5 
layer 3: 16x16x256; stride 2, filters 5 
layer 4: 32x32x128; stride 2, filters 5
layer 5: 64x64x3; stride 2 filters 5 

'''

class DC_Encoder(nn.Module) -> None: 
   """Implementation of the DCGAN encoder for generation of audio samples """
   
   # inits 
   def __init__(self, sample): 
       super(DC_Encoder, self).__init__()
       self.sample = sample
       self.type = 'rap'


## decoder 
'''
same as encoder but inverted ! 
'''


