import math
import numpy as np
class CONFIG():

    def __init__(self,CLFTOKEN = True):
        self.CLFTOKEN = CLFTOKEN
        
        self.B = 2
        # Define Lenght of the Time-series
        # Define Number of the ROI
        self.T_prim , self.V = 1200 , 116
        # Define P & S Hyper-prameters that used to define numebrs of windows (1 <=P , S <= T_prim)
        self.P = 50
        self.S = 30
        
        # Define T (number of windows)
        self.T = math.floor((self.T_prim - self.P)/  self.S  ) + 1 
        ## Define Convulotion HyperParameters
        self.K_E = self.T * 2  # Define K_E (Use in Convolution Part of the Temporal attention)
        self.KERNEL_1 = (1,self.P // 10)   # Kernel size for the first Conv 
        self.STRIDE_1 = (1,2)  # Stride for both Conv1 & Conv2
        self.P_1 = math.floor((self.P - self.KERNEL_1[1])/  self.STRIDE_1[1]  ) + 1  # Output of the first Conv
        self.KERNEL_2 = (1 , self.P_1 // 10)
        self.STRIDE_2 = (1,1)
        self.P_2 =  math.floor((self.P_1 - self.KERNEL_2[1])/  self.STRIDE_2[1]  ) + 1

        
        self.K_S = self.P_2 * 2  # (Use in the Query & Key weight matrix)
        self.K_F = self.P_2 * 2  # (Number of Outchannel of GCN)
        self.NUM_H1 = 1  # Num Head in temporal Attention
        self.NUM_H2 = 2  # Num Head in laster transformer layers
        self.HIDDEN_DIM = 128 # Embed dim for the attention 
        if self.CLFTOKEN : 
            self.V = self.V + 1
        self.DROP_OUT = 0.0
        self.NUM_LAYER = 10
        self.NUM_CLASS = 2



