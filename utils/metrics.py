import copy
import numpy as np

def norm_heatmap(heatmap_, nan, mode=0):
    # mode: 0 -> "[-1,1]"
    #       1 -> "[0, 1]"
    heatmap = copy.deepcopy(heatmap_)
    heatmap_wo_nan = heatmap[~nan]

    if heatmap_wo_nan.max() - heatmap_wo_nan.min() == 0:
        print(f"heatmap max == min == {heatmap_wo_nan.max()}")
        return heatmap_wo_nan
    
    heatmap_wo_nan = (heatmap_wo_nan - heatmap_wo_nan.min()) / (heatmap_wo_nan.max() - heatmap_wo_nan.min())

    if mode == 0:
        heatmap_wo_nan  = heatmap_wo_nan * 2 - 1 
    heatmap[~nan] = heatmap_wo_nan

    # # sigmoid numpy
    # heatmap = 1 / (1 + np.exp(-heatmap_))
    # if mode == 0:
    #     heatmap = (heatmap - 0.5) * 2
    # elif mode == 1:
    #     heatmap = heatmap - 0.5
    # else:
    #     raise ValueError(f"Invalid mode: {mode}")
    return heatmap


def compute_cnr(gtmask_, heatmap_, nan):
    """
    Compute contrast-to-noise ratio (CNR) between ground truth mask and heatmap.
    For CNR, let A and A_ denote the interior and exterior of the bounding box, respectively.
    CNR = |meanA - meanA_| / pow((varA_ + varA), 0.5)
    
    Inputs:
        - gtmask_ (np.ndarray): shape=(H, W), ground truth mask
        - heatmap_ (np.ndarray): shape=(H, W), heatmap
        - nan (np.ndarray): shape=(H, W), True if the pixel is NaN

    Returns:
        - CNR (float): contrast-to-noise ratio
    """
    heatmap = norm_heatmap(heatmap_, nan)
    heatmap_wo_nan = heatmap[~nan]
    gtmask_wo_nan = gtmask_[~nan]
    # assert (gtmask_wo_nan == 1).sum() > 0, 'gtmask_wo_nan == 1 is empty'
    A = heatmap_wo_nan[gtmask_wo_nan == 1]
    A_ = heatmap_wo_nan[gtmask_wo_nan == 0]
    meanA = A.mean()
    meanA_ = A_.mean()
    varA = A.var()
    varA_ = A_.var()
    if varA + varA_ == 0:
        CNR = 0
    else:
        CNR = (meanA - meanA_) / pow((varA + varA_), 0.5)
    return CNR