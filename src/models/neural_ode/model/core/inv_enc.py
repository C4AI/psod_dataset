import torch
import torch.nn as nn
from model.core.vae import EncoderRNN


class INV_ENC(nn.Module):
    def __init__(
        self,
        modulator_dim=0,
        content_dim=0,
        rnn_hidden=10,
        T_inv=10,
        data_dim=1,
        device="cpu",
    ):
        super(INV_ENC, self).__init__()
        self.modulator_dim = modulator_dim
        self.content_dim = content_dim
        self.inv_encoder = InvariantEncoderRNN(
            data_dim,
            T_inv=T_inv,
            rnn_hidden=rnn_hidden,
            enc_out_dim=modulator_dim + content_dim,
            out_distr="dirac",
        ).to(device)

    def kl(self):
        return torch.zeros(1) * 0.0

    def forward(self, X, L=1):
        """
        X is [N,T,nc,d,d] or [N,T,q]
        returns [L,N,T,q]
        """
        c = self.inv_encoder(X)  # N,Tinv,q or N,ns,q
        return c.repeat([L, 1, 1, 1])  # L,N,T,q


class InvariantEncoderRNN(EncoderRNN):
    def __init__(
        self, input_dim, T_inv=None, rnn_hidden=10, enc_out_dim=16, out_distr="dirac"
    ):
        super(InvariantEncoderRNN, self).__init__(
            input_dim,
            rnn_hidden=rnn_hidden,
            enc_out_dim=enc_out_dim,
            out_distr=out_distr,
        )
        self.T_inv = T_inv

    def forward(self, X, ns=5):
        [N, T, d] = X.shape
        T_inv = T // 2 if self.T_inv is None else self.T_inv
        T_inv = min(T_inv, T)
        X = X.repeat([ns, 1, 1])
        t0s = torch.randint(0, T - T_inv + 1, [ns * N])
        X = torch.stack(
            [X[n, t0 : t0 + T_inv] for n, t0 in enumerate(t0s)]
        )  # ns*N,T_inv,d
        X_out = super().forward(X)  # ns*N,enc_out_dim
        return X_out.reshape(ns, N, self.enc_out_dim).permute(
            1, 0, 2
        )  # N,ns,enc_out_dim
