import torch
import numpy as np


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)


def camera_loss(camera_out):
    device = camera_out.get_device()
    if device == -1:
        device = torch.device('cpu')
    m = torch.reshape(camera_out, [-1, 2, 3])
    m_sq = m @ torch.permute(m, [0, 2, 1])
    batch_trace = torch.vmap(torch.trace)(m_sq)
    loss_mat = (2 / batch_trace).reshape([-1, 1, 1]) * m_sq - torch.eye(2).to(device)
    loss = torch.abs(loss_mat).sum(dim=[1, 2])
    loss = loss.mean()
    return loss


def weighted_pose_2d_loss(y_true, y_pred, num_keypoint):
    # the custom loss functions weights joints separately
    # it's possible to completely ignore joint detections by setting the respective entries to zero
    device = y_pred.get_device()
    if device == -1:
        device = torch.device('cpu')
    # diff = torch.abs(y_true - y_pred).float()
    diff = torch.pow(y_true - y_pred, 2).float()
    # weighting the joints
    weights_t = torch.tensor(
        np.array([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        # np.ones(num_keypoint * 2)
    ).float().to(device)
    weights = torch.tile(weights_t.reshape([1, num_keypoint * 2]), [y_pred.shape[0], 1])
    tmp = weights * diff
    loss = tmp.sum(dim=1) / (num_keypoint * 2)
    loss = loss.mean()
    return loss
