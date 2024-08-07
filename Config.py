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
        self.S = 50
        # Define T (number of windows)
        self.T = math.floor((self.T_prim - self.P)/  self.S  )
        # Define K_E (Use in Convolution Part of the Temporal attention)
        self.K_E = self.P * 2
        self.K_S = self.K_E * 2  # (Use in the Query & Key weight matrix)
        self.K_F = self.K_E  # (Number of Outchannel of GCN)
        self.NUM_H = 1 # (Number of Head in Temporal Transformer )
        self.HIDDEN_DIM = 32 # Embed dim for the attention 
        if self.CLFTOKEN : 
            self.V = self.V + 1
        self.DROP_OUT = 0.1
        self.NUM_LAYER = 1
        self.NUM_CLASS = 2