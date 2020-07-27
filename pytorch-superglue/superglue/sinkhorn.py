import torch


def sinkhorn_pytorch(a, b, M, reg, stopThr=1e-3, numItermax=100):

    # init data
    dim_a = a.shape[1]
    dim_b = b.shape[1]

    batch_size = b.shape[0]

    u = torch.ones((batch_size, dim_a), requires_grad=False).cuda() / dim_a
    v = torch.ones((batch_size, dim_b), requires_grad=False).cuda() / dim_b
    K = torch.exp(-M / reg)

    Kp = (1 / a).unsqueeze(2) * K
    cpt = 0
    err = 1

    while err > stopThr and cpt < numItermax:
        KtransposeU = (K * u.unsqueeze(2)).sum(dim=1) # has shape K.shape[1]
        v = b / KtransposeU
        u = 1. / (Kp*v.unsqueeze(1)).sum(dim=2)
        cpt = cpt + 1
    return u.unsqueeze(2) * K * v.unsqueeze(1)

