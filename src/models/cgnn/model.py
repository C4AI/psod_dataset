from collections import defaultdict
import math
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F


adjoint = True
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

import sys

sys.path.append("src")
from processing.loader import (
    MultivariateFeaturesSample,
    MultivariateTimestampsSample,
)


class PositionalEncoding(nn.Module):
    """Encoder that applies positional based encoding.

    Encoder that considers data temporal position in the time series' tensor to provide
    a encoding based on harmonic functions.

    Attributes:
        hidden_size (int): size of hidden representation
        dropout (nn.Module): dropout layer
        div_term (torch.Tensor): tensor with exponential based values, used to encode timestamps


    """

    def __init__(self, time_encoding_size: int, dropout: float, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = time_encoding_size
        self.div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2).float()
            * (-math.log(10000.0) / self.hidden_size)
        )

    def forward(self, position):
        """Encoder's forward procedure.

        Encodes time information based on temporal position. In the encoding's
        vector, for even positions employs sin function over position*div_term and for
        odds positions uses cos function. Then, applies dropout layer to resulting tensor.

        Args:
            torch.tensor[int]: temporal positions for encoding

        Returns:
            torch.tensor[float]: temporal encoding
        """

        pe = torch.empty(*position.shape, self.hidden_size, device=position.device)
        pe[..., 0::2] = torch.sin(
            position[..., None] * self.div_term.to(position.device)
        )
        pe[..., 1::2] = torch.cos(
            position[..., None] * self.div_term.to(position.device)
        )
        return self.dropout(pe)


def get_act(act="relu"):
    if act == "relu":
        return nn.ReLU()
    elif act == "elu":
        return nn.ELU()
    elif act == "celu":
        return nn.CELU()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sin":
        return torch.sin
    elif act == "linear":
        return nn.Identity()
    elif act == "softplus":
        return nn.modules.activation.Softplus()
    elif act == "swish":
        return lambda x: x * torch.sigmoid(x)
    elif act == "lipswish":
        return lambda x: 0.909 * torch.nn.functional.silu(x)
    else:
        return None


class MLP(nn.Module):
    def __init__(
        self, n_in, n_out, L=2, H=100, act="relu", skip_con=False, dropout_rate=0.2
    ):
        super().__init__()
        layers_ins = [n_in] + L * [H]
        layers_outs = L * [H] + [n_out]
        self.H = H
        self.L = L
        self.n_in = n_in
        self.n_out = n_out
        self.layers = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.acts = nn.ModuleList([])
        for i, (n_in, n_out) in enumerate(zip(layers_ins, layers_outs)):
            self.layers.append(nn.Linear(n_in, n_out))
            self.norms.append(
                nn.Sequential(
                    nn.BatchNorm1d(n_in),
                )
            )
            self.acts.append(
                get_act(act) if i < L else get_act("linear")
            )  # no act. in final layer
        self.skip_con = skip_con
        self.reset_parameters()

    @property
    def device(self):
        return next(self.layers[0].parameters()).device

    @property
    def type(self):
        return "MLP"

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def kl(self):
        return torch.zeros(1).to(self.device)

    def forward(self, x):
        for i, (act, layer, norm) in enumerate(zip(self.acts, self.layers, self.norms)):
            x = x.swapaxes(1, 2)
            h = norm(x)
            h = h.swapaxes(1, 2)
            h = layer(h)
            h = act(h)
            x = x + h if self.skip_con and 0 < i and i < self.n_hid_layers else h
        return x

    def draw_f(self):
        return self


class ODEFuncW(nn.Module):

    # currently requires in_features = out_features
    def __init__(
        self, in_features: int, out_features: int, alpha: float, adj: torch.tensor
    ):
        super(ODEFuncW, self).__init__()
        self.adj = adj
        self.x0 = None
        self.nfe = 0
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.alpha_train = nn.Parameter(self.alpha * torch.ones(adj.shape[1]))

        self.w = nn.Parameter(torch.eye(in_features))
        self.d = nn.Parameter(torch.zeros(in_features) + 1)

    def forward(self, t, x):
        self.nfe += 1

        alph = F.sigmoid(self.alpha_train).unsqueeze(dim=1)

        adj = self.adj.repeat(x.shape[0], 1, 1)

        ax = torch.matmul(adj, x)

        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.matmul(x, w)

        f = alph * 0.5 * (ax - x) + xw - x + self.x0

        return f


class ODEblockW(nn.Module):
    def __init__(
        self, odefunc, solver="rk4", atol=1e-6, rtol=1e-6, use_adjoint="adjoint"
    ):
        super(ODEblockW, self).__init__()
        self.odefunc = odefunc
        self.nfe = 0
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.use_adjoint = use_adjoint

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, H_0, ts):
        self.nfe += 1

        H_t = odeint(
            self.odefunc, H_0, ts, atol=self.atol, rtol=self.rtol, method=self.solver
        )

        return H_t


class CGNN(nn.Module):
    def __init__(
        self,
        *,
        input_sizes: dict[str, int],
        num_layers_rnns: int,
        hidden_units: int,
        time_encoder: nn.Module | None = None,
        context_time_series: list[str],
        target_time_series: list[str],
        propagate_time_series: list[str],
        alpha: float = 0.95,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        self.time_encoder = time_encoder
        self.device = device
        self.dtype = dtype
        self.context_time_series = context_time_series
        self.target_time_series = target_time_series
        self.propagate_time_series = propagate_time_series

        self.canonical_order = sorted(input_sizes.keys())

        input_sizes = {
            ts_name: input_sizes[ts_name] for ts_name in self.context_time_series
        }

        embed_input = input_sizes
        if self.time_encoder is not None:
            embed_input = {
                ts_name: int(embed_s + self.time_encoder.hidden_size)
                for ts_name, embed_s in embed_input.items()
            }

        self.embeds = nn.ModuleDict(
            {
                ts_name: nn.Linear(embed_s, hidden_units)
                for ts_name, embed_s in embed_input.items()
            }
        )
        self.rnns = nn.ModuleDict(
            {
                ts_name: nn.GRU(
                    hidden_units,
                    hidden_units,
                    num_layers=num_layers_rnns,
                    batch_first=True,
                )
                for ts_name in input_sizes
            }
        )

        if len(self.propagate_time_series):
            adj = torch.tensor([[0, 1], [1, 0]]).to(self.device).to(dtype)
        else:
            adj = None

        self.flow = ODEblockW(ODEFuncW(2 * hidden_units, 2 * hidden_units, alpha, adj))

        self.decoders = nn.ModuleDict(
            {
                ts_name: MLP(
                    n_in=2 * hidden_units,
                    n_out=input_sizes[ts_name],
                    L=2,
                    H=hidden_units,
                    act="relu",
                    skip_con=False,
                )
                for ts_name in target_time_series
            }
        )

    def rnn_encode(
        self,
        context_timestamps: MultivariateTimestampsSample,
        context: MultivariateFeaturesSample,
        run_backward: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:

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
            for ts_name, context_elem in context.items()
        }

        if self.time_encoder is not None:
            context_padded_timestamps = {
                ts_name: nn.utils.rnn.pad_sequence(
                    [
                        (c[1:] if c.size(0) > 0 else torch.empty(0, device=c.device))
                        for c in c_ts_elem
                    ],
                    batch_first=True,
                )
                for ts_name, c_ts_elem in context_timestamps.items()
            }
            encoded_timestamps = {
                ts_name: self.time_encoder(context_pad_elem)
                for ts_name, context_pad_elem in context_padded_timestamps.items()
            }
            inputs = {
                ts_name: torch.cat([ip_element, encoded_timestamps[ts_name]], dim=-1)
                for ts_name, ip_element in inputs.items()
            }

        for ts_name, inp in inputs.items():
            if inp.isnan().any():
                print("here")

        inputs = {ts_name: self.embeds[ts_name](inp) for ts_name, inp in inputs.items()}

        if run_backward:
            common_results = {
                ts_name: self.rnns[ts_name](torch.flip(inputs, [1]))[0][:, -1, :]
                for ts_name, inputs in inputs.items()
            }
        else:
            common_results = {
                ts_name: self.rnns[ts_name](inputs)[0][:, -1, :]
                for ts_name, inputs in inputs.items()
            }
        return common_results

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
    ) -> MultivariateFeaturesSample:

        context_timestamps = {
            ts_name: context_timestamps[ts_name] for ts_name in self.context_time_series
        }
        context_features = {
            ts_name: context_features[ts_name] for ts_name in self.context_time_series
        }

        # change the reference of the timestamps to the initial context window timestamp
        t_start = [
            min(
                [
                    ts[ts_id][0].item() if ts[ts_id].size(0) > 0 else float("inf")
                    for ts_name, ts in context_timestamps.items()
                ]
            )
            for ts_id in range(len(t_inferences))
        ]

        if float("inf") in t_start:
            raise ValueError("Empty context window")

        context_timestamps_ = self.move_time_reference(context_timestamps, t_start)
        target_timestamps_ = self.move_time_reference(target_timestamps, t_start)

        h_0 = self.rnn_encode(
            context_timestamps=context_timestamps_,
            context=context_features,
        )

        H_0 = torch.cat(
            [
                torch.cat(
                    [
                        ts.unsqueeze(1),
                        torch.zeros(ts.unsqueeze(1).shape).to(self.device),
                    ],
                    dim=-1,
                )
                for ts in h_0.values()
            ],
            dim=1,
        )

        if H_0.isnan().any():
            raise ValueError("NaN values in the initial hidden state")

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
        union_timestamps_dt = 0.0001 * union_timestamps

        self.flow.set_x0(H_0)
        H_t = self.flow(H_0, union_timestamps_dt)

        if H_t.isnan().any():
            raise ValueError("NaN values in the hidden state")

        H_t = H_t.permute(1, 0, 2, 3)

        hidden_trajectories = {
            ts_name: H_t[:, :, ts_id, :]
            for ts_id, ts_name in enumerate(self.target_time_series)
        }

        forecasting = {
            ts_name: self.decoders[ts_name](hidden_trajectories[ts_name])
            for ts_name in self.target_time_series
        }

        for ts_name in self.target_time_series:
            if forecasting[ts_name].isnan().any():
                raise ValueError("NaN values in the forecasting")

        forecasting_original_timestamps = {
            ts_name: [
                forecasting[ts_name][ts_id][
                    torch.searchsorted(
                        union_timestamps, full_timestamps[ts_name][ts_id]
                    )
                ]
                for ts_id in range(len(t_inferences))
            ]
            for ts_name in forecasting.keys()
        }

        return forecasting_original_timestamps
