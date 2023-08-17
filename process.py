import numpy as np
def pred_transform(x):
    result=10 ** x
    result/=np.pi
    return result