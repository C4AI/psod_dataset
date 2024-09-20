from collections import defaultdict
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

import sys

sys.path.append("src")
from processing.loader import (
    MultivariateFeaturesSample,
    MultivariateTimestampsSample,
)


@dataclass
class ModelConfig:
    num_layers_rnn: int
    num_layers_gnn: int
    num_heads_gnn:int
    time_encoding_size: int
    hidden_units: int

    def to_dict(self):
        return {
            "num_layers_rnn": self.num_layers_rnn,
            "num_layers_gnn": self.num_layers_gnn,
            "num_heads_gnn":self.num_heads_gnn,
            "time_encoding_size": self.time_encoding_size,
            "hidden_units": self.hidden_units,
        }


class PositionalEncoding(nn.Module):
    """Encoder that applies positional based encoding.

    Encoder that considers data temporal position in the time series' tensor to provide
    a encoding based on harmonic functions.

    Attributes:
        hidden_size (int): size of hidden representation
        dropout (nn.Module): dropout layer
        div_term (torch.Tensor): tensor with exponential based values, used to encode timestamps


    """

    def __init__(self, time_encoding_size: int, dropout: float = 0, **kwargs):
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


def create_fully_connected_edge_index(n_nodes: int) -> torch.Tensor:
    # Create the source and target indices
    src_indices = torch.repeat_interleave(torch.arange(n_nodes), n_nodes)
    tgt_indices = torch.tile(torch.arange(n_nodes), (n_nodes,))

    # Stack the indices to create the edge_index tensor
    edge_index = torch.stack([src_indices, tgt_indices], dim=0)

    return edge_index


def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
    index of agreement
    Willmott (1981, 1982)

    Args:
        s: simulated
        o: observed

    Returns:
        ia: index of agreement
    """
    o_bar = torch.mean(o, dim=0)
    ia = 1 - (torch.sum((o - s) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(s - o_bar) + torch.abs(o - o_bar)) ** 2,
            dim=0,
        )
    )

    return ia


def create_fully_connected_bipartite_edge_index(
    n_nodes: tuple[int, int]
) -> torch.Tensor:
    # Create the source and target indices

    n_nodes_src, n_nodes_tgt = n_nodes

    edge_index = torch.stack(
        [torch.arange(n_nodes_src), torch.arange(n_nodes_tgt)], dim=0
    )

    return edge_index


def create_fully_connected_hetero_edge_index(
    n_nodes: dict[str, int]
) -> dict[tuple[str, str, str], torch.Tensor]:

    edge_index_dict = {
        (
            src_name,
            f"{src_name}_{tgt_name}",
            tgt_name,
        ): create_fully_connected_bipartite_edge_index(
            (n_nodes[src_name], n_nodes[tgt_name])
        )
        for src_name in n_nodes
        for tgt_name in n_nodes
    }

    return edge_index_dict


class GNN(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int,num_heads:int):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                pyg_nn.GATConv(hidden_channels, hidden_channels,heads=num_heads, add_self_loops=False,concat=False)
                for _ in range(num_layers)
            ]
        )

        self.projections = nn.ModuleList(
            [pyg_nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )

        self.skip_connections = nn.ModuleList(
            [pyg_nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        out = x
        for layer in range(self.num_layers):
            out = self.projections[layer](
                self.convs[layer](out, edge_index).relu()
            ) + self.skip_connections[layer](out)

        return out


class HeteroGNN(nn.Module):
    def __init__(
        self,
        *,
        hidden_units: int,
        num_layers: int,
        num_heads:int,
        node_types: list[str],
    ) -> None:
        super().__init__()

        homo_gnn = GNN(hidden_units, num_layers,num_heads)

        self.gnn = pyg_nn.to_hetero(
            homo_gnn,
            (
                node_types,
                [
                    (src, f"{src}_{tgt}", tgt)
                    for src in node_types
                    for tgt in node_types
                ],
            ),
        )

        self.node_types = node_types

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        out_dict = self.gnn(x_dict, edge_index_dict)

        return out_dict


class GapAheadAMTSRegressor(nn.Module):
    def __init__(
        self,
        *,
        input_sizes: dict[str, int],
        num_layers_rnns: int,
        hidden_units: int,
        num_layers_gnn: int = 0,
        num_heads_gnn:int=0,
        time_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.time_encoder = time_encoder

        self.canonical_order = sorted(input_sizes.keys())

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

        if num_layers_gnn > 0:
            self.gnn = HeteroGNN(
                hidden_units=hidden_units,
                num_layers=num_layers_gnn,
                num_heads=num_heads_gnn,
                node_types=self.canonical_order,
            )
        else:
            self.gnn = None

        self.projections = nn.ModuleDict(
            {
                ts_name: nn.Linear(hidden_units, input_s)
                for ts_name, input_s in input_sizes.items()
            }
        )

    def rnn_encode(
        self,
        context_timestamps: MultivariateTimestampsSample,
        context: MultivariateFeaturesSample,
        first_targets: dict[str, MultivariateTimestampsSample],
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

        inputs = {ts_name: self.embeds[ts_name](inp) for ts_name, inp in inputs.items()}

        common_results = {
            ts_name: self.rnns[ts_name](inputs) for ts_name, inputs in inputs.items()
        }
        common_results = {
            ts_name: torch.cat(
                [
                    out[[i], context[ts_name][i][:-1].shape[0] - 1]
                    for i in range(len(inputs[ts_name]))
                ]
            )
            for ts_name, (out, _) in common_results.items()
        }

        branched_results: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

        for tgt_name, first_targets_elem in first_targets.items():
            for src_name, frst_target_tensor_list in first_targets_elem.items():
                if self.time_encoder is not None:

                    frst_target_tensor = self.time_encoder(
                        torch.cat(
                            [
                                (
                                    first_tgt_tensor.view(1)
                                    if len(first_tgt_tensor.shape) == 0
                                    else torch.zeros(1, device=first_tgt_tensor.device)
                                )
                                for first_tgt_tensor in frst_target_tensor_list
                            ],
                            dim=0,
                        )
                    ).unsqueeze(1)

                last_measurements = nn.utils.rnn.pad_sequence(
                    [
                        (
                            c[[-1]]
                            if c.size(0) > 0
                            else torch.empty(0, c.size(1), device=c.device)
                        )
                        for c in context[src_name]
                    ],
                    batch_first=True,
                )

                if self.time_encoder is not None:
                    inputs = torch.cat([last_measurements, frst_target_tensor], dim=-1)
                else:
                    inputs = last_measurements

                inputs = self.embeds[src_name](inputs)

                h_0 = common_results[src_name].unsqueeze(0)

                out, _ = self.rnns[src_name](
                    inputs,
                    h_0,
                )

                branched_results[tgt_name][src_name] = out[:, 0]

        return branched_results

    def rnn_ar_decode(
        self,
        h_0: dict[str, torch.Tensor],
        target_timestamps: MultivariateTimestampsSample,
    ) -> MultivariateFeaturesSample:

        max_target_timestamps = {
            ts_name: max(
                [
                    target_timestamp.size(0)
                    for target_timestamp in target_timestamps_elem
                ]
            )
            for ts_name, target_timestamps_elem in target_timestamps.items()
        }

        if self.time_encoder is not None:
            target_padded_timestamps = {
                ts_name: self.time_encoder(
                    nn.utils.rnn.pad_sequence(target_timestamps_elem, batch_first=True)
                )
                for ts_name, target_timestamps_elem in target_timestamps.items()
            }

        out = {
            ts_name: torch.empty(
                h_0_elem.shape[0],
                max_target_timestamps[ts_name],
                self.projections[ts_name].out_features,
                device=h_0_elem.device,
            )
            for ts_name, h_0_elem in h_0.items()
        }
        h_t_minus = {
            ts_name: h_0_elem.unsqueeze(0) for ts_name, h_0_elem in h_0.items()
        }
        z_t_hat_minus = {
            ts_name: self.projections[ts_name](h_0_elem)
            for ts_name, h_0_elem in h_0.items()
        }

        for ts_name, z_t_hat_minus_elem in z_t_hat_minus.items():
            out[ts_name][:, 0] = z_t_hat_minus_elem

        for ts_name in h_0.keys():
            for i in range(1, max_target_timestamps[ts_name]):
                inp = z_t_hat_minus[ts_name]
                if self.time_encoder is not None:
                    target_timestamp = target_padded_timestamps[ts_name][:, i]
                    inp = torch.cat([inp, target_timestamp], dim=-1)

                inp = self.embeds[ts_name](inp)
                _, h_t = self.rnns[ts_name](inp.unsqueeze(1), h_t_minus[ts_name])
                z_t_hat = self.projections[ts_name](h_t.squeeze(0))
                out[ts_name][:, i] = z_t_hat
                h_t_minus[ts_name] = h_t
                z_t_hat_minus[ts_name] = z_t_hat

        out = {
            ts_name: [
                out[ts_name][i, : len(target_timestamps_elem[i])]
                for i in range(len(target_timestamps_elem))
            ]
            for ts_name, target_timestamps_elem in target_timestamps.items()
            if ts_name in out
        }

        return out

    def gnn_propagate(
        self, h_0: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:

        if self.gnn is None:
            return {
                ts_name: h_0_minus_dict[ts_name]
                for ts_name, h_0_minus_dict in h_0.items()
            }

        h_0_out: dict[str, torch.Tensor] = {}
        for ts_name, h_0_elem in h_0.items():
            src_x_dict = {
                source_name: (
                    src_features
                    if src_features.size(0) > 0
                    else torch.zeros(1, device=src_features.device)
                )
                for source_name, src_features in h_0_elem.items()
            }
            edge_index_dict = create_fully_connected_hetero_edge_index(
                n_nodes={
                    source_name: src_features.size(0)
                    for source_name, src_features in src_x_dict.items()
                }
            )
            edge_index_dict = {
                edge_index: edge.to(src_x_dict[edge_index[0]].device)
                for edge_index, edge in edge_index_dict.items()
            }
            out = self.gnn(x_dict=src_x_dict, edge_index_dict=edge_index_dict)
            h_0_out[ts_name] = out[ts_name]

        return h_0_out

    def get_first_targets(
        self,
        context_timestamps: MultivariateTimestampsSample,
        target_timestamps: MultivariateTimestampsSample,
    ) -> dict[str, MultivariateTimestampsSample]:
        """
        Finds the first target timestamps for each context source.

        Args:
            context_timestamps (MultivariateWindowSample): context timestamps
            target_timestamps (MultivariateWindowSample): target timestamps

        Returns:
            torch.Tensor: first_targets
        """

        all_first_targets: dict[str, MultivariateTimestampsSample] = {}

        for target_name, tgt_ts_sample in target_timestamps.items():

            if len(tgt_ts_sample) == 0 or not any(
                [t.size(0) > 0 for t in tgt_ts_sample]
            ):
                raise ValueError(f"Empty batch of target timestamps for {target_name}")

            first_targets = defaultdict(list)

            for batch_pos, tgt_ts in enumerate(tgt_ts_sample):
                if tgt_ts.size(0) == 0:
                    for source_name in context_timestamps.keys():
                        first_targets[source_name].append(
                            torch.empty(0, device=tgt_ts.device)
                        )
                    continue

                for source_name in context_timestamps.keys():

                    first_targets[source_name].append(tgt_ts[0])

            all_first_targets[target_name] = first_targets

        return all_first_targets

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

        context_timestamps_ = self.move_time_reference(context_timestamps, t_inferences)
        target_timestamps_ = self.move_time_reference(target_timestamps, t_inferences)

        first_targets = self.get_first_targets(
            context_timestamps=context_timestamps_,
            target_timestamps=target_timestamps_,
        )

        h_0 = self.rnn_encode(
            context_timestamps=context_timestamps_,
            context=context_features,
            first_targets=first_targets,
        )

        h_0 = self.gnn_propagate(h_0)

        out = self.rnn_ar_decode(h_0=h_0, target_timestamps=target_timestamps_)

        return out
