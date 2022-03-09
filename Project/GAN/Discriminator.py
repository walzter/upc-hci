import torch
from torch import nn

class Discriminator(nn.Module):
    """
    Basic Discriminator according to the GAN paper by Goodfellow et al. 
    
    Opposite to what the Generator does 
    
    DOWNSAMPLE ONE: 
    --------------
    Linear(output_shape, 1024)
    LeakyReLU(0.2)
    
    DOWNSAMPLE TWO: 
    --------------
    Linear(1024, 512)
    LeakyReLU(0.2)
    
    DOWNSAMPLE THREE: 
    --------------
    Linear(512, 256)
    LeakyReLU(0.2)
    
    DOWNSAMPLE FINAL: 
    --------------
    Linear(256, 1)
    Sigmoid()
    """
    def __init__(self, config_dict):
        super(Discriminator, self).__init__()
        self.block = nn.Sequential(
            ## Output -> 1024
            nn.Linear(config_dict['OUTPUT_SHAPE'],1024),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## 1024 -> 512
            nn.Linear(1024, 512),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## 512 -> 256
            nn.Linear(512, 256),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## 256 -> 1
            nn.Linear(256, 1),
            nn.LeakyReLU(config_dict['LEAKY_RELU_SLOPE']),
            ## sigmoid
            nn.Sigmoid()
        )
    def forward(self, x):
        """Forward pass of the model"""
        return self.block(x)