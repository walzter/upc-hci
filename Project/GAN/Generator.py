import torch
from torch import nn

class Generator(nn.Module):
    """
    Basic Generator according to the GAN paper by Goodfellow et al. 
    
    There need to be 3 upsamplings done: 128 -> 256 -> 512 
    In between each upsampling there is going to be a BatchNorm with 0.8
    And an activation which is LeakyReLU with m=0.2. 
    
    FIRST UPSAMPLING: 
    -----------------
        Linear (Noise_dimension, 128, bias=False)
        BatchNorm(128, 0.8)
        LeakyReLU(0.2)
    SECOND UPSAMPLING:
    -----------------
        Linear (128, 256, bias=False)
        BatchNorm(256, 0.8)
        LeakyReLU(0.2)
    THIRD UPSAMPLING:
    -----------------
        Linear (256, 512, bias=False)
        BatchNorm(512, 0.8)
        LeakyReLU(0.2)
    LAST UPSAMPLE:
    -------------
        Linear (512, FINAL_OUTPUT_SHAPE, bias=False )
        Tanh()
    
    
    """
    
    def __init__(self, config_dict):
        super(Generator, self).__init__()
        self.block = nn.Sequential(
            ## First Upsampling block 
            nn.Linear(config_dict['NOISE_DIM'], 128, bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## Second Upsampling block 
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## Third Upsampling block 
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## Final Upsampling 
            nn.Linear(512, config_dict['OUTPUT_SHAPE'], bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        """Forward pass of the model"""
        return self.block(x)    