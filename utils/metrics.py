"""
@file metrics.py

Holds a variety of metric computing functions for time-series forecasting models, including
Valid Prediction Time (VPT), Valid Prediction Distance (VPD), etc.
"""
import numpy as np
from scipy.stats import spearmanr
from skimage.filters import threshold_otsu
from scipy.optimize import linear_sum_assignment


def vpt(gt, preds, epsilon=0.020, **kwargs):
    """
    Computes the Valid Prediction Time metric, as proposed in https://openreview.net/pdf?id=qBl8hnwR0px
    VPT = argmin_t [MSE(gt, pred) > epsilon]
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    :param epsilon: threshold for valid prediction
    """
    # Ensure on CPU and numpy
    if not isinstance(gt, np.ndarray):
        gt = gt.cpu().numpy()
        preds = preds.cpu().numpy()

    # Get dimensions
    _, timesteps, height, width = gt.shape

    # Get pixel_level MSE at each timestep
    mse = (gt - preds) ** 2
    mse = np.sum(mse, axis=(2, 3)) / (height * width)

    # Get VPT
    vpts = []
    for m in mse:
        # Get all indices below the given epsilon
        indices = np.where(m < epsilon)[0] + 1

        # If there are none below, then add 0
        if len(indices) == 0:
            vpts.append(0)
            continue

        # Append last in list
        vpts.append(indices[-1])

    # Return VPT mean over the total timesteps
    return vpts, np.mean(vpts) / timesteps, np.std(vpts) / timesteps


def thresholding(preds, gt):
    """
    Thresholding function that converts gt and preds into binary images
    Activated prediction pixels are found via Otsu's thresholding function
    """
    N, T = gt.shape[0], gt.shape[1]
    res = np.zeros_like(preds)

    # For each sample and timestep, get Otsu's threshold and binarize gt and pred
    for n in range(N):
        for t in range(T):
            img = preds[n, t]
            otsu_th = np.max([0.32, threshold_otsu(img)])
            res[n, t] = (img > otsu_th).astype(np.float32)
            gt[n, t] = (gt[n, t] > 0.55).astype(np.float32)
    return res, gt


def dst(gt, preds, **kwargs):
    """
    Computes a Euclidean distance metric between the center of the ball in ground truth and prediction
    Activated pixels in the predicted are computed via Otsu's thresholding function
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    """
    # Ensure on CPU and numpy
    if not isinstance(gt, np.ndarray):
        gt = gt.cpu().numpy()
        preds = preds.cpu().numpy()

    # # Convert to grayscale if not already
    # if gt.shape[2] > 1:
    #     gt = 0.2989 * gt[:, :, 0, :, :] + 0.5870 * gt[:, :, 1, :, :] +  0.1140 * gt[:, :, 2, :, :]
    #     preds = 0.2989 * preds[:, :, 0, :, :] + 0.5870 * preds[:, :, 1, :, :] +  0.1140 * preds[:, :, 2, :, :]

    # # Regardless, cut out that axis
    # gt = gt[:, :, 0]
    # preds = preds[:, :, 0]

    # Get shapes
    num_samples, timesteps, height, width = gt.shape

    # Apply Otsu thresholding function on output
    preds, gt = thresholding(preds, gt)

    # Loop over each sample and timestep to get the distance metric
    results = np.zeros([num_samples, timesteps])
    for n in range(num_samples):
        for t in range(timesteps):
            # Get all active predicted pixels
            a = preds[n, t]
            b = gt[n, t]
            pos_a = np.where(a == 1)
            pos_b = np.where(b == 1)

            # If there are in gt, add 0
            if pos_b[0].shape[0] == 0:
                results[n, t] = 0
                continue

            # Get gt center
            center_b = np.array([pos_b[0].mean(), pos_b[1].mean()])

            # Get center of predictions
            if pos_a[0].shape[0] != 0:
                center_a = np.array([pos_a[0].mean(), pos_a[1].mean()])
            # If no pixels above threshold, add the highest possible error in image space
            else:
                # results[n, t] = np.sqrt(np.sum(np.array([height, width]) ** 2))
                center_a = [0, 0]
                # continue

            # Get distance metric
            dist = np.sum((center_a - center_b) ** 2)
            dist = np.sqrt(dist)

            # Add to result
            results[n, t] = dist

    return results, np.mean(results), np.std(results)


def vpd(target, output, epsilon=10, **kwargs):
    """
    Computes the Valid Prediction Time metric, as proposed in https://openreview.net/forum?id=7C9aRX2nBf2
    VPD = argmin_t [DST(gt, pred) > epsilon]
    :param gt: ground truth sequences
    :param preds: model predicted sequences
    :param epsilon: threshold for valid prediction
    """
    # Ensure on CPU and numpy
    if not isinstance(output, np.ndarray):
        output = output.cpu().numpy()
        target = target.cpu().numpy()

    # # Convert to grayscale if not already
    # if output.shape[2] > 1:
    #     output = 0.2989 * output[:, :, 0, :, :] + 0.5870 * output[:, :, 1, :, :] +  0.1140 * output[:, :, 2, :, :]
    #     target = 0.2989 * target[:, :, 0, :, :] + 0.5870 * target[:, :, 1, :, :] +  0.1140 * target[:, :, 2, :, :]

    #     # Regardless, cut out that axis
    #     output = output[:, :, 0]
    #     target = target[:, :, 0]

    # Get shapes
    num_samples, timesteps, height, width = target.shape
    
    # Apply Otsu thresholding function on output
    preds, gt = thresholding(output, target)

    # Loop over each sample and timestep to get the distance metric
    dsts = np.zeros([num_samples, timesteps])
    for n in range(num_samples):
        for t in range(timesteps):
            # Get all active predicted pixels
            a = preds[n, t]
            b = gt[n, t]
            pos_a = np.where(a == 1)
            pos_b = np.where(b == 1)

            # If there are in gt, add 0
            if pos_b[0].shape[0] == 0:
                dsts[n, t] = 0
                continue

            # Get gt center
            center_b = np.array([pos_b[0].mean(), pos_b[1].mean()])

            # Get center of predictions
            if pos_a[0].shape[0] != 0:
                center_a = np.array([pos_a[0].mean(), pos_a[1].mean()])
            # If no pixels above threshold, add the highest possible error in image space
            else:
                dsts[n, t] = np.sqrt(np.sum(np.array([height, width]) ** 2))
                continue

            # Get distance metric
            dist = np.sum((center_a - center_b) ** 2)
            dist = np.sqrt(dist)

            # Add to result
            dsts[n, t] = dist

    # Get the VPD calculation
    B, T = dsts.shape
    vpdist = np.zeros(B)
    for i in range(B):
        idx = np.where(dsts[i, :] >= epsilon)[0]
        if idx.shape[0] > 0:
            vpdist[i] = np.min(idx)
        else:
            vpdist[i] = T

    # Return VPT mean over the total timesteps
    return vpdist, np.mean(vpdist) / T, np.std(vpdist) / T


def recon_mse(output, target, **kwargs):
    """ Gets the mean of the per-pixel MSE for the given length of timesteps used for training """
    full_pixel_mses = (output[:, :kwargs['cfg'].generation_length] - target[:, :kwargs['cfg'].generation_length]) ** 2
    sequence_pixel_mse = np.mean(full_pixel_mses, axis=(1, 2, 3))
    return sequence_pixel_mse, np.mean(sequence_pixel_mse), np.std(sequence_pixel_mse)


def mean_corr_coef(x, y):
    # Reshape to correct dim
    x = x.reshape([x.shape[0] * x.shape[1], -1])
    y = y.reshape([y.shape[0] * y.shape[1], -1])
    
    # Ensure on CPU and numpy
    if not isinstance(x, np.ndarray):
        x = x.cpu().numpy()
        y = y.cpu().numpy()
    
    print(x.shape, y.shape)
    d = x.shape[1]
    cc = spearmanr(x, y)[0][:d, d:]
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    print(score)
    return score
