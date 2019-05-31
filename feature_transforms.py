import torch


def WCT(alpha, fc, fs, fs1=None, beta=None):

    # content image whitening
    fc = fc.double()
    c_channels, c_width, c_height = fc.size(0), fc.size(1), fc.size(2)
    fcv = fc.view(c_channels, -1)  # c x (h x w)

    c_mean = torch.mean(fcv, 1) # perform mean for each row
    c_mean = c_mean.unsqueeze(1).expand_as(fcv) # add dim and replicate mean on rows
    fcv = fcv - c_mean # subtract mean element-wise

    c_covm = torch.mm(fcv, fcv.t()).div((c_width * c_height) - 1)  # construct covariance matrix
    c_u, c_e, c_v = torch.svd(c_covm, some=False) # singular value decomposition

    k_c = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)

    w_step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    w_step2 = torch.mm(w_step1, (c_v[:, 0:k_c].t()))
    whitened = torch.mm(w_step2, fcv)

    # style image coloring
    fs = fs.double()
    s_channels, s_width, s_heigth = fs.size(0), fs.size(1), fs.size(2)
    fsv = fs.view(s_channels, -1)

    s_mean = torch.mean(fsv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(fsv)
    fsv = fsv - s_mean

    s_covm = torch.mm(fsv, fsv.t()).div((s_width * s_heigth) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = c_channels # same number of channels ad content features
    for i in range(c_channels):
        if s_e[i] < 0.00001:
            s_k = i
            break
    s_d = (s_e[0:s_k]).pow(0.5)

    c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
    c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
    colored = torch.mm(c_step2, whitened)

    cs0_features = colored + s_mean.resize_as_(colored)
    cs0_features = cs0_features.view_as(fc)

    # additional style coloring
    if beta:
        fs = fs1
        fs = fs.double()
        s_channels, s_width, s_heigth = fs.size(0), fs.size(1), fs.size(2)
        fsv = fs.view(s_channels, -1)

        s_mean = torch.mean(fsv, 1)
        s_mean = s_mean.unsqueeze(1).expand_as(fsv)
        fsv = fsv - s_mean

        s_covm = torch.mm(fsv, fsv.t()).div((s_width * s_heigth) - 1)
        s_u, s_e, s_v = torch.svd(s_covm, some=False)

        s_k = c_channels
        for i in range(c_channels):
            if s_e[i] < 0.00001:
                s_k = i
                break
        s_d = (s_e[0:s_k]).pow(0.5)

        c_step1 = torch.mm(s_v[:, 0:s_k], torch.diag(s_d))
        c_step2 = torch.mm(c_step1, s_v[:, 0:s_k].t())
        colored = torch.mm(c_step2, whitened)

        cs1_features = colored + s_mean.resize_as_(colored)
        cs1_features = cs1_features.view_as(fc)

        target_features = beta * cs0_features + (1.0 - beta) * cs1_features
    else:
        target_features = cs0_features

    fccs = alpha * target_features + (1.0 - alpha) * fc
    return fccs.float()


def WCT_mask(fc, fs):
    fc = fc.double()
    fc_sizes = fc.size()
    c_mean = torch.mean(fc, 1)
    c_mean = c_mean.unsqueeze(1).expand_as(fc)
    fc -= c_mean

    c_covm = torch.mm(fc, fc.t()).div(fc_sizes[1] - 1)
    c_u, c_e, c_v = torch.svd(c_covm, some=False)

    k_c = fc_sizes[0]
    for i in range(fc_sizes[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)
    whitened = torch.mm(torch.mm(torch.mm(c_v[:, 0:k_c], torch.diag(c_d)), (c_v[:, 0:k_c].t())), fc)

    fs = fs.double()
    fs_sizes = fs.size()
    fsv = fs.view(fs_sizes[0], fs_sizes[1] * fs_sizes[2])
    s_mean = torch.mean(fsv, 1)
    s_mean = s_mean.unsqueeze(1).expand_as(fsv)
    fsv -= s_mean

    s_covm = torch.mm(fsv, fsv.t()).div((fs_sizes[1] * fs_sizes[2]) - 1)
    s_u, s_e, s_v = torch.svd(s_covm, some=False)

    s_k = fs_sizes[0]
    for i in range(fs_sizes[0]):
        if s_e[i] < 0.00001:
            s_k = i
            break
    s_d = (s_e[0:s_k]).pow(0.5)
    fccs = torch.mm(torch.mm(torch.mm(s_v[:, 0:s_k], torch.diag(s_d)), s_v[:, 0:s_k].t()), whitened)

    fccs += s_mean.resize_as_(fccs)
    return fccs.float()
