from sklearn.metrics import r2_score
import torch

def out_of_bound_error(x):
    lower_bound_loss = (0 - x).clamp_min(0)
    upper_bound_loss = (x - 1).clamp_min(0)
    return lower_bound_loss.sum() + upper_bound_loss.sum()

def mask_error(x, m):
    return (x * m).clamp_min(0).sum() 

def absolute_val_error(x, y):
    diff = (x - y)
    return diff.abs().sum()

def squared_val_error(x, y):
    diff = (x - y) ** 2
    return diff.sum()

def huber_val_loss(x, y):
    return torch.nn.functional.huber_loss(x, y, delta=0.05)

def LOSS_FN(pred, inp, true, padding=128, eps=1e-8):
    # x = torch.sigmoid(pred[:, :, padding:-padding, padding:-padding])
    x = pred[:, :, padding:-padding, padding:-padding]
    m = inp[:, 0:1, padding:-padding, padding:-padding]
    y = true[:, :, padding:-padding, padding:-padding]
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    n = x.numel()
    # n = x.shape[0]
    if n == 0: n = eps

    # # error = squared_val_error(x, y) + out_of_bound_error(x) + mask_error(x, m)
    # # error = squared_val_error(x, y)

    error = squared_val_error(x, y) + mask_error(x, m)
    return (error / n).clamp_min(0)

    # error = huber_val_loss(x, y) + (mask_error(x, m) / n)
    # return error.clamp_min(0)

def LOSS_EVAL(pred, inp, true, padding=128, eps=1e-8):
    # x = torch.sigmoid(pred[:, :, padding:-padding, padding:-padding])
    x = pred[:, :, padding:-padding, padding:-padding]
    m = inp[:, 0:1, padding:-padding, padding:-padding]
    y = true[:, :, padding:-padding, padding:-padding]
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    n = x.numel()
    # n = x.shape[0]
    if n == 0: n = eps
    mae = absolute_val_error(x, y) / n
    mse = squared_val_error(x, y) / n

    return f'MAE:{mae.cpu().item():2f}, MSE: {mse.cpu().item():2f}'

def LOSS_FN_H(pred, inp, true, padding=128, eps=1e-8):
    x = pred
    # m = inp[:, 0:1, :, :]
    y = true
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    n = x.numel()
    # n = x.shape[0]
    if n == 0: n = eps

    # # error = squared_val_error(x, y) + out_of_bound_error(x) + mask_error(x, m)
    # # error = squared_val_error(x, y)

    error = squared_val_error(x, y)
    return (error / n).clamp_min(0)

    # error = huber_val_loss(x, y) + (mask_error(x, m) / n)
    # return error.clamp_min(0)

def LOSS_EVAL_H(pred, inp, true, padding=128, eps=1e-8):
    # x = torch.sigmoid(pred[:, :, padding:-padding, padding:-padding])
    x = pred
    # m = inp[:, 0:1, :, :]
    y = true
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    n = x.numel()
    # n = x.shape[0]
    if n == 0: n = eps
    mae = absolute_val_error(x, y) / n
    mse = squared_val_error(x, y) / n
    return f'MAE:{mae.cpu().item():2f}, MSE: {mse.cpu().item():2f}'
