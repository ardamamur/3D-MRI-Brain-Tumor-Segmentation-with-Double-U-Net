# Note that this is the soft-dice version: union term has squared sub-terms

import numpy as np

def dice_loss_one_term(true: np.ndarray, pred: np.ndarray):
    batch_size = true.shape[0]
    intersect = true.reshape(batch_size, -1) * pred.reshape(batch_size, -1)
    union = true.reshape(batch_size, -1)**2 + pred.reshape(batch_size, -1)**2 + 1e-8
    return 2*intersect.sum()/union.sum()

def dice_loss_first_equivalent(true, pred):
    intersect = true * pred
    union = true**2 + pred**2 + 1e-8
    return 2*intersect.sum()/union.sum()

def dice_loss_batch_average(true: np.ndarray, pred: np.ndarray):
    batch_size = true.shape[0]
    intersect = true.reshape(batch_size, -1) * pred.reshape(batch_size, -1)
    union = true.reshape(batch_size, -1)**2 + pred.reshape(batch_size, -1)**2 + 1e-8
    res = 2*intersect.sum(-1)/union.sum(-1)
    return np.mean(res)

def dice_loss_batch_channel_average(true: np.ndarray, pred: np.ndarray):
    size = true.shape[:2]
    intersect = true.reshape(*size, -1) * pred.reshape(*size, -1)
    union = true.reshape(*size, -1)**2 + pred.reshape(*size, -1)**2 + 1e-8
    res = 2*intersect.sum(-1)/union.sum(-1)
    return np.mean(res)

def main():
    np.random(32, 3, 15,15,15)

    true = np.random.randint(0,2,(32, 3, 54,54,54))
    pred = np.random.randint(0,2,(32, 3, 54,54,54))

    print(f"Dice Loss as One fraction: {dice_loss_one_term(pred, true)}")
    print(f"Dice Loss as Average over Batch: {dice_loss_batch_average(pred, true)}")
    print(f"Dice Loss as Average per channel and batch: {dice_loss_batch_channel_average(pred, true)}")


if __name__ == "__main__":
    main()