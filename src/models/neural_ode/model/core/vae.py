import torch
import torch.nn as nn
from torch.distributions import Normal
from torchsummary import summary
from model.core.gru_encoder import GRUEncoder
from model.core.mlp import MLP
import numpy as np

EPSILON = 1e-5


class VAE(nn.Module):

    def __init__(
        self,
        dec_H=100,
        rnn_hidden=10,
        dec_act="relu",
        ode_latent_dim=8,
        content_dim=0,
        device="cpu",
        order=1,
        enc_H=50,
        data_dim=1,
    ):
        super(VAE, self).__init__()

        ### build encoder
        self.prior = Normal(
            torch.zeros(ode_latent_dim).to(device),
            torch.ones(ode_latent_dim).to(device),
        )
        self.ode_latent_dim = ode_latent_dim
        self.order = order

        lhood_distribution = "normal"

        if rnn_hidden == -1:
            self.encoder = IdentityEncoder()
            self.decoder = IdentityDecoder(data_dim)
            if order == 2:
                self.encoder_v = IdentityEncoder()
        else:
            self.encoder = EncoderRNN(
                data_dim,
                rnn_hidden=rnn_hidden,
                enc_out_dim=ode_latent_dim,
                out_distr="normal",
                H=enc_H,
            ).to(device)
            self.decoder = Decoder(
                ode_latent_dim + content_dim,
                H=dec_H,
                distribution=lhood_distribution,
                dec_out_dim=data_dim,
                act=dec_act,
            ).to(device)
            if order == 2:
                self.encoder_v = EncoderRNN(
                    data_dim,
                    rnn_hidden=rnn_hidden,
                    enc_out_dim=ode_latent_dim,
                    out_distr="normal",
                    H=enc_H,
                ).to(device)
                self.prior = Normal(
                    torch.zeros(ode_latent_dim * order).to(device),
                    torch.ones(ode_latent_dim * order).to(device),
                )

    def reset_parameters(self):
        modules = [self.encoder, self.decoder]
        if self.order == 2:
            modules += [self.encoder_v]
        for module in modules:
            for layer in module.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def print_summary(self):
        """Print the summary of both the models: encoder and decoder"""
        summary(self.encoder, (1, *(28, 28)))
        summary(self.decoder, (1, self.ode_latent_dim))
        if self.order == 2:
            summary(self.encoder_v, (1, *(28, 28)))

    def save(self, encoder_path=None, decoder_path=None):
        """Save the VAE model. Both encoder and decoder and saved in different files."""
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def test(self, x):
        """Test the VAE model on data x. First x is encoded using encoder model, a sample is produced from then latent
        distribution and then it is passed through the decoder model."""
        self.encoder.eval()
        self.decoder.eval()
        enc_m, enc_log_var = self.encoder(x)
        z = self.encoder.sample(enc_m, enc_log_var)
        y = self.decoder(z)
        return y


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp = nn.Softplus()

    @property
    def device(self):
        return self.sp.device

    def sample(self, mu, std, L=1):
        """mu,std  - [N,q]
        returns - [L,N,q] if L>1 else [N,q]"""
        if std is None:
            return mu
        eps = (
            torch.randn([L, *std.shape]).to(mu.device).to(mu.dtype).squeeze(0)
        )  # [N,q] or [L,N,q]
        return mu + std * eps

    def q_dist(self, mu_s, std_s, mu_v=None, std_v=None):
        if mu_v is not None:
            means = torch.cat((mu_s, mu_v), dim=-1)
            stds = torch.cat((std_s, std_v), dim=-1)
        else:
            means = mu_s
            stds = std_s

        return Normal(means, stds)  # N,q

    @property
    def device(self):
        return self.sp.device


class EncoderRNN(AbstractEncoder):
    def __init__(
        self, input_dim, rnn_hidden=10, enc_out_dim=16, out_distr="normal", H=50
    ):
        super(EncoderRNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.out_distr = out_distr
        enc_out_dim = enc_out_dim + enc_out_dim * (out_distr == "normal")
        self.gru = GRUEncoder(enc_out_dim, input_dim, rnn_output_size=rnn_hidden, H=H)

    def forward(self, x):
        outputs = self.gru(x)
        if self.out_distr == "normal":
            z0_mu, z0_log_sig = (
                outputs[
                    :,
                    : self.enc_out_dim,
                ],
                outputs[:, self.enc_out_dim :],
            )
            z0_log_sig = self.sp(z0_log_sig)
            return z0_mu, z0_log_sig
        return outputs


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def sample(self, mu, std, L=1):
        return torch.stack([mu] * L) if L > 1 else mu

    def q_dist(self, mu_s, std_s, mu_v=None, std_v=None):
        return Normal(mu_s, torch.ones_like(mu_s))  # N,q

    def __call__(self, x):
        return x[:, 0], None

    def __repr__(self) -> str:
        return "Identity encoder"


class IdentityDecoder(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.data_dim = data_dim

    def __call__(self, z, dims):
        return z[..., : self.data_dim]

    def log_prob(self, X, Xhat, L=1):
        XL = X.repeat([L] + [1] * X.ndim)  # L,N,T,nc,d,d or L,N,T,d
        assert XL.numel() == Xhat.numel()
        Xhat = Xhat.reshape(XL.shape)
        return torch.distributions.Normal(XL, torch.ones_like(XL)).log_prob(Xhat)

    def __repr__(self) -> str:
        return "Identity decoder"


class Decoder(nn.Module):
    def __init__(
        self,
        dec_inp_dim,
        H=100,
        distribution="bernoulli",
        dec_out_dim=1,
        act="relu",
    ):
        super(Decoder, self).__init__()
        self.distribution = distribution
        self.net = MLP(dec_inp_dim, dec_out_dim, L=2, H=H, act=act)
        self.out_logsig = torch.nn.Parameter(torch.zeros(dec_out_dim) * 0.0)
        self.sp = nn.Softplus()

    def forward(self, z, dims):
        inp = z.contiguous().view([np.prod(list(z.shape[:-1])), z.shape[-1]])  # L*N*T,q
        Xrec = self.net(inp)
        return Xrec.view(dims)  # L,N,T,...

    @property
    def device(self):
        return next(self.parameters()).device

    def log_prob(self, X, Xhat):
        """
        x - input [N,T,nc,d,d]   or [N,T,d]
        z - preds [L,N,T,nc,d,d] or [L,N,T,d]
        """
        std = self.sp(self.out_logsig)
        log_p = torch.cat(
            [
                torch.distributions.Normal(X[ts_id], std).log_prob(Xhat[ts_id])
                for ts_id in range(len(X))
            ],
        )

        return log_p
