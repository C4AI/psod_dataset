from collections import defaultdict
import torch
import torch.nn as nn

from processing.loader import (
    MultivariateFeaturesSample,
    MultivariateTimestampsSample,
)


class MoNODE(nn.Module):
    def __init__(
        self, model, flow, vae, order, dt, inv_enc=None, aug=False, nobj=1, Tin=5
    ) -> None:
        super().__init__()

        self.flow = flow
        self.vae = vae
        self.inv_enc = inv_enc
        self.dt = dt
        self.order = order
        self.aug = aug
        self.Nobj = nobj
        self.model = model
        self.Tin = Tin

    @property
    def device(self):
        return self.flow.device

    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    @property
    def is_inv(self):
        return self.inv_enc is not None

    def build_decoding(self, ztL, dims, c=None):
        """
        Given a mean of the latent space decode the input back into the original space.

        @param ztL: latent variable (L,N,T,q)
        @param inv_z: invaraiant latent variable (L,N,q)
        @param dims: dimensionality of the original variable
        @param c: invariant code (L,N,q)
        @return Xrec: reconstructed in original data space (L,N,T,nc,d,d)
        """

        if self.order == 1:
            stL = ztL
        elif self.order == 2:
            q = ztL.shape[-1] // 2
            stL = ztL[:, :, :, :q]  # L,N,T,q Only the position is decoded

        if c is not None:
            cT = torch.stack([c] * ztL.shape[2], -2)  # L,N,T,q
            stL = torch.cat([stL, cT], -1)  # L,N,T,2q

        Xrec = self.vae.decoder(stL, dims)  # L,N,T,...
        return Xrec

    def sample_trajectories(self, z0L, ts, L=1):
        """
        @param z0L - initial latent encoding of shape [L,N,nobj,q]
        """
        # sample L trajectories
        ts = self.dt * ts

        ztL = torch.stack([self.flow(z0, ts) for z0 in z0L])  # [L,N,T,nobj,q]
        return ztL  # L,N,T,Nobj,q)

    def sample_augmented_trajectories(self, z0L, cL, ts, L=1):
        """
        @param z0L - initial latent encoding of shape [L,N, Nobj,q]
        @param cL - invariant code  [L,N,q]
        """
        ts = self.dt * ts

        ztL = [
            self.flow(z0, ts, zc) for z0, zc in zip(z0L, cL)
        ]  # sample L trajectories
        return torch.stack(ztL)  # L,N,T,nobj,2q

    def forward(self, X, ts: torch.tensor = torch.arange(0, 1, 0.1)):

        try:
            self.inv_enc.last_layer_gp.build_cache()
        except:
            pass

        N = X.shape[0]
        T = ts.shape[0]
        L = 1  # one trajectory

        # condition on
        in_data = X
        # if self.model == 'node':
        s0_mu, s0_logv = self.vae.encoder(in_data)  # N,q
        z0 = self.vae.encoder.sample(s0_mu, s0_logv, L=L)  # N,q or L,N,q
        z0 = z0.unsqueeze(0) if z0.ndim == 2 else z0  # L,N,q

        # if multiple object separate latent vector (but shared dynamics)
        q = z0.shape[-1]
        z0 = z0.reshape(L, N, self.Nobj, q // self.Nobj)  # L,N,nobj,q_

        v0_mu, v0_logv = None, None
        if self.order == 2:
            v0_mu, v0_logv = self.vae.encoder_v(in_data)
            v0 = self.vae.encoder_v.sample(v0_mu, v0_logv, L=L)  # N,q or L,N,q
            v0 = v0.unsqueeze(0) if v0.ndim == 2 else v0

            # if multiple object separate latent vector (but shared dynamics)
            q = v0.shape[-1]
            v0 = v0.reshape(L, N, self.Nobj, q // self.Nobj)  # L,N,nobj,q_
            if self.model == "hbnode":
                z0 = torch.concat([z0, v0], dim=2)  # L, N, 2, q
            else:
                z0 = torch.concat([z0, v0], dim=-1)  # L, N, 1, 2q

        # encode content (invariance), pass whole sequence length
        if self.is_inv:
            InvMatrix = self.inv_enc(X, L=L)  # embeddings [L,N,T,q] or [L,N,ns,q]
            inv_var = InvMatrix.mean(2)  # time-invariant code [L,N,q]
            c, m = (
                inv_var[:, :, : self.inv_enc.content_dim],
                inv_var[:, :, self.inv_enc.content_dim :],
            )
        else:
            InvMatrix, c, m = None, None, None

        # sample trajectories
        if self.aug:
            mL = m.reshape((L, N, self.Nobj, -1))  # L,N,Nobj,q
            ztL = self.sample_augmented_trajectories(z0, mL, ts, L)  # L,N,T,Nobj, 2q
            ztL = ztL.reshape(L, N, T, -1)  # L,T,N, nobj*2q
            Xrec = self.build_decoding(ztL, [L, N, T, *X.shape[2:]], c)
        else:
            ztL = self.sample_trajectories(z0, ts, L)  # L,T,N,nobj,q
            ztL = ztL.reshape(L, N, T, -1)  # L,T,N, nobj*q
            Xrec = self.build_decoding(ztL, [L, N, T, *X.shape[2:]], c)

        return (
            Xrec.squeeze(0),
            ztL.squeeze(0),
            (s0_mu, s0_logv),
            (v0_mu, v0_logv),
            InvMatrix.squeeze(0),
            c.squeeze(0),
            m.squeeze(0),
        )


class Node_Forecaster(nn.Module):
    def __init__(
        self,
        flow=nn.Module,
        vae=nn.Module,
        order: int = 1,
        dt: float = 0.1,
        inv_enc=nn.Module,
        aug: bool = False,
        nobj: int = 1,
        Tin: int = 5,
    ) -> None:
        super().__init__()
        self.model = MoNODE(
            model="node",
            flow=flow,
            vae=vae,
            order=order,
            dt=dt,
            inv_enc=inv_enc,
            aug=aug,
            nobj=nobj,
            Tin=Tin,
        )

    def move_time_reference(
        self, timestamps: MultivariateTimestampsSample, t_inferences: torch.Tensor
    ):
        """
        Move the reference of the timestamps to the time of the inferences.
        """
        result = defaultdict(list)
        for ts_name, ts in timestamps.items():
            for i in range(len(ts)):
                if ts[i].size(0) == 0:
                    result[ts_name].append(torch.empty(0, device=ts[i].device))
                    continue

                result[ts_name].append(ts[i] - t_inferences[i])
        return result

    def forward(
        self,
        context_timestamps: MultivariateTimestampsSample,
        context_features: MultivariateFeaturesSample,
        target_timestamps: MultivariateTimestampsSample,
        t_inferences: torch.Tensor,
        ts_name: str,
    ):

        context_timestamps = {
            ts_name: context_timestamps[ts_name] for ts_name in [ts_name]
        }
        context_features = {ts_name: context_features[ts_name] for ts_name in [ts_name]}

        t_start = [
            min(
                [
                    ts[ts_id][0].item() if ts[ts_id].size(0) > 0 else torch.inf
                    for ts_name, ts in context_timestamps.items()
                ]
            )
            for ts_id in range(len(t_inferences))
        ]

        t_start = [t if t != torch.inf else 0 for t in t_start]

        context_timestamps_ = self.move_time_reference(context_timestamps, t_start)
        target_timestamps_ = self.move_time_reference(target_timestamps, t_start)

        inputs = {
            ts_name: nn.utils.rnn.pad_sequence(
                [
                    (
                        c[:-1]
                        if c.size(0) > 0
                        else torch.empty(0, c.size(1), device=c.device)
                    )
                    for c in context_elem
                ],
                batch_first=True,
            )
            for ts_name, context_elem in context_features.items()
        }

        full_timestamps = {
            ts_name: [
                torch.cat(
                    [
                        context_timestamps_[ts_name][ts_id],
                        target_timestamps_[ts_name][ts_id],
                    ]
                )
                for ts_id in range(len(t_inferences))
            ]
            for ts_name in context_timestamps_.keys()
        }

        time_series_union_timestamps = {
            ts_name: torch.cat(full_timestamps[ts_name]).unique().sort()[0]
            for ts_name in full_timestamps.keys()
        }

        union_timestamps = (
            torch.cat([ts for ts in time_series_union_timestamps.values()])
            .unique()
            .sort()[0]
        )

        Xrec, ztL, (s0_mu, s0_logv), (v0_mu, v0_logv), C, c, m = self.model(
            inputs[ts_name], union_timestamps
        )

        forecasting_original_timestamps = {
            ts_name: [
                Xrec[ts_id][
                    torch.searchsorted(
                        union_timestamps, full_timestamps[ts_name][ts_id]
                    )
                ]
                for ts_id in range(len(t_inferences))
            ]
        }

        return (
            forecasting_original_timestamps[ts_name],
            s0_mu,
            s0_logv,
            v0_mu,
            v0_logv,
        )
