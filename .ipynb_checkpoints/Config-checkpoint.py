import math
import numpy as np
class Config():
    BATCH_SIZE = 2
    # Define Lenght of the Time-series
    # Define Number of the ROI
    T_prim , V = np.loadtxt("../Data/AAL/100408_timeseries.txt").shape
    # Define P & S Hyper-prameters that used to define numebrs of windows (1 <=P , S <= T_prim)
    P = 20
    S = 100
    # Define T (number of windows)
    T = math.floor(( T_prim - 2*( P - 1 ) - 1) / ( S + 1 ) )
    # Define K_E
    K_E = P * 2