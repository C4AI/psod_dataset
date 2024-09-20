import torch
from torch.distributions import kl_divergence as kl


def elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv):
    """Input:
        qz_m        - latent means [N,2q]
        qz_logv     - latent logvars [N,2q]
        X           - input images [L,N,T,nc,d,d]
        Xrec        - reconstructions [L,N,T,nc,d,d]
    Returns:
        likelihood
        kl terms
    """
    # KL reg
    q = model.model.vae.encoder.q_dist(s0_mu, s0_logv, v0_mu, v0_logv)
    kl_z0 = kl(q, model.model.vae.prior).sum(-1)  # N

    # Reconstruction log-likelihood
    lhood = model.model.vae.decoder.log_prob(X, Xrec)  # L,N,T,d,nc,nc

    return lhood.mean(), kl_z0.mean()


def contrastive_loss(C):
    """
    C - invariant embeddings [N,T,q] or [L,N,T,q]
    """
    C = C.mean(0) if C.ndim == 4 else C
    C = C / C.pow(2).sum(-1, keepdim=True).sqrt()  # N,Tinv,q
    N_, T_, q_ = C.shape
    C = C.reshape(N_ * T_, q_)  # NT,q
    Z = (C.unsqueeze(0) * C.unsqueeze(1)).sum(-1)  # NT, NT
    idx = torch.meshgrid(torch.arange(T_), torch.arange(T_))
    idxset0 = torch.cat([idx[0].reshape(-1) + n * T_ for n in range(N_)])
    idxset1 = torch.cat([idx[1].reshape(-1) + n * T_ for n in range(N_)])
    pos = Z[idxset0, idxset1].sum()
    return -pos


def compute_mse(model, data, T_train, L=1, task=None):

    T_start = 0
    T_max = 0
    T = data.shape[1]
    # run model
    Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = model(data, L, T)

    dict_mse = {}
    while T_max < T:
        if task == "rot_mnist":
            T_max += T_train
            mse = torch.mean((Xrec[:, :, T_start:T_max] - data[:, T_start:T_max]) ** 2)
            dict_mse[str(T_max)] = mse
            T_start += T_train
            T_max += T_train
        else:
            T_max += T_train
            mse = torch.mean((Xrec[:, :, :T_max] - data[:, :T_max]) ** 2)
            dict_mse[str(T_max)] = mse
    return dict_mse


def compute_loss(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv):
    """
    Compute loss for optimization
    @param model: mo/node
    @param data: true observation sequence
    @param L: number of MC samples
    @return: loss, nll, regularizing_kl, inducing_kl
    """

    num_observations = len(X)

    # compute loss

    lhood, kl_z0 = elbo(model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv)

    lhood = lhood * num_observations
    kl_z0 = kl_z0 * num_observations
    loss = -lhood + kl_z0
    mse = torch.tensor(
        [torch.mean((Xrec[ts_id] - X[ts_id]) ** 2) for ts_id in range(len(X))]
    ).mean()
    return loss, -lhood, kl_z0, mse
