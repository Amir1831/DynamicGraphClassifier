from Config import CONFIG
import torch
import numpy as np
Config = CONFIG()
def SplitWindows(Series):
    """
    Args (T_prim * V):
    Series : Take one Subject Fmri TimeSries.
    Return (P * V * T):
    Series of Splited windows that can be interpreted every window as a graph. 
    """
    B , T_prim , V = Series.shape
    # print(Series.shape)
    BATCH = []
    for b in range(B):
        l = []
        for t in range(Config.T):
            try:
                l.append(Series[b][t*Config.S:t*Config.S + Config.P,:])
            except:
                print("Not Fit")
        BATCH.append(torch.stack(l))
    # print(np.array(BATCH).shape)
    return torch.stack(BATCH).transpose(2,3)