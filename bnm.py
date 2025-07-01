import torch


def batch_norm_maximization_loss(t_confidence):

    if t_confidence.ndim != 2:
        raise ValueError("t_confidence should be a 2D tensor of shape (batch_size, num_classes)")
    if not torch.is_floating_point(t_confidence):
        raise TypeError("t_confidence should be a float tensor")

    with torch.no_grad():
        U, S, V = torch.svd(t_confidence)
        nuclear_norm = torch.sum(S)
        N_B = t_confidence.size(0)
        loss_bnm = -nuclear_norm / N_B
    return loss_bnm
