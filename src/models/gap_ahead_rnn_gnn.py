import math
import torch
import torch.nn as nn
import uniplot
import polars as pl


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

    return ia.mean()


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


class SimpleARRNN(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        num_layers: int,
        output_size: int,
        hidden_units: int,
        activation_function: nn.Module | None = None,
        time_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.activation_function = activation_function

        self.time_encoder = time_encoder

        embed_input = input_size
        if self.time_encoder is not None:
            embed_input += self.time_encoder.hidden_size

        self.embed = nn.Linear(embed_input, hidden_units)
        self.rnn = nn.GRU(
            hidden_units,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
        )

        self.projection = nn.Linear(hidden_units, output_size)

    def encode(
        self, context: list[torch.Tensor], context_timestamps: list[torch.Tensor]
    ) -> torch.Tensor:

        input = nn.utils.rnn.pad_sequence(context, batch_first=True)
        if self.time_encoder is not None:
            context_padded_timestamps = nn.utils.rnn.pad_sequence(
                context_timestamps, batch_first=True
            )
            encoded_timestamps = self.time_encoder(context_padded_timestamps)
            input = torch.cat([input, encoded_timestamps], dim=-1)

        input = self.embed(input)

        out, _ = self.rnn(input)

        out = torch.cat([out[[i], len(context[i]) - 1] for i in range(len(context))])

        return out

    def decode(
        self, h_0: torch.Tensor, target_timestamps: list[torch.Tensor]
    ) -> list[torch.Tensor]:

        max_target_timestamps = max(
            [target_timestamp.size(0) for target_timestamp in target_timestamps]
        )
        if self.time_encoder is not None:
            target_padded_timestamps = nn.utils.rnn.pad_sequence(
                target_timestamps, batch_first=True
            )

        out = torch.empty(
            h_0.shape[0],
            max_target_timestamps + 1,
            self.projection.out_features,
            device=h_0.device,
        )
        h_t_minus = h_0.unsqueeze(0)
        z_t_minus = self.projection(h_0)
        out[:, 0] = z_t_minus
        for i in range(max_target_timestamps):
            inp = z_t_minus
            if self.time_encoder is not None:
                target_timestamp = target_padded_timestamps[:, i]
                target_timestamp = self.time_encoder(target_timestamp)
                inp = torch.cat([inp, target_timestamp], dim=-1)

            inp = self.embed(inp)
            _, h_t = self.rnn(inp.unsqueeze(1), h_t_minus)
            z_t = self.projection(h_t.squeeze(0))
            out[:, i + 1] = z_t
            h_t_minus = h_t
            z_t_minus = z_t

        out = [
            out[i, : len(target_timestamps[i]) + 1]
            for i in range(len(target_timestamps))
        ]

        return out

    def forward(
        self,
        context: list[torch.Tensor],
        context_timestamps: list[torch.Tensor],
        target_timestamps: list[torch.Tensor],
    ) -> list[torch.Tensor]:

        rolled_context_timestamps = []
        rolled_target_timestamps = []
        for i in range(len(context)):
            context_ts = context_timestamps[i].roll(-1)
            context_ts[-1] = target_timestamps[i][0]
            rolled_context_timestamps.append(context_ts)
            rolled_target_timestamps.append(target_timestamps[i].roll(-1)[:-1])

        h_0 = self.encode(context=context, context_timestamps=rolled_context_timestamps)
        out = self.decode(h_0=h_0, target_timestamps=rolled_target_timestamps)

        return out


def prepare_batch(
    batch_size: int,
    lbound: float,
    ubound: float,
    context_size: int,
    forecast_size: int,
    ts: torch.Tensor,
    ys: torch.Tensor,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    t_inf = torch.empty(batch_size, 1, device=ts.device).uniform_(
        lbound + context_size, ubound - forecast_size
    )
    lbounds = t_inf - context_size
    ubounds = t_inf + forecast_size

    lindices = torch.searchsorted(ts, lbounds)
    tinf_indices = torch.searchsorted(ts, t_inf)
    uindices = torch.searchsorted(ts, ubounds)
    context = []
    context_timestamps = []
    target = []
    target_timestamps = []
    for i, (lb, ti, ub) in enumerate(zip(lindices, tinf_indices, uindices)):
        if ts[lb:ti].shape[0] == 0 or ts[ti:ub].shape[0] == 0:
            continue
        context.append(ys[lb:ti])
        context_timestamps.append(ts[lb:ti] - t_inf[i])

        target.append(ys[ti:ub])
        target_timestamps.append(ts[ti:ub] - t_inf[i])

    return context, context_timestamps, target, target_timestamps


# Example usage
n_samples = 100000
n_features = 1
n_layers = 1
hidden_units = 100
epochs = 100
lbound = 0.0
ubound = 10000.0
batch_size = 64
context_size = 100
content_increase_step = 50
forecast_size = 30
forecast_increase_step = 30
n_iter = 10000
time_encoding_size = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# ts, _ = torch.empty(n_samples, device=device).uniform_(lbound, ubound).sort()
# ys = torch.sin(ts).unsqueeze(-1)

df = pl.read_parquet("data/santos/current_praticagem.parquet")
ts = torch.tensor(
    (df["datetime"] - df["datetime"].min()).dt.total_minutes().to_numpy(),
    device=device,
    dtype=torch.float32,
)
ys = torch.tensor(
    df["cross_shore_current"].to_numpy(), device=device, dtype=torch.float32
).unsqueeze(-1)

lbound = ts.min().item()
ubound = ts.max().item()

# 80% train, 20% test
train_size = int(0.8 * n_samples)
train_ts = ts[:train_size]
train_ys = ys[:train_size]
train_lbound, train_ubound = lbound, ubound * 0.8

test_ts = ts[train_size:]
test_ys = ys[train_size:]
test_lbound, test_ubound = ubound * 0.8, ubound

model = SimpleARRNN(
    input_size=test_ys.shape[-1],
    num_layers=n_layers,
    output_size=n_features,
    hidden_units=hidden_units,
    activation_function=None,
    time_encoder=PositionalEncoding(time_encoding_size=time_encoding_size, dropout=0.1),
    # time_encoder=None,
)
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    for it in range(n_iter):
        context, context_timestamps, target, target_timestamps = prepare_batch(
            batch_size=batch_size,
            lbound=train_lbound,
            ubound=train_ubound,
            context_size=context_size,
            forecast_size=forecast_size,
            ts=train_ts,
            ys=train_ys,
        )

        model.zero_grad()
        forecast = model(
            context=context,
            target_timestamps=target_timestamps,
            context_timestamps=context_timestamps,
        )
        losses = torch.stack(
            [
                1.0 - index_agreement_torch(f, t.to(device))
                for f, t in zip(forecast, target)
            ]
        )
        loss = losses.mean()
        loss.backward()
        optimizer.step()
        if it % 10 == 0:
            print(f"Iteration: {it}, Loss: {loss}, Forecast Size: {forecast_size}")

    for _ in range(1):
        context, context_timestamps, target, target_timestamps = prepare_batch(
            batch_size=batch_size,
            lbound=test_lbound,
            ubound=test_ubound,
            context_size=context_size,
            forecast_size=forecast_size,
            ts=test_ts,
            ys=test_ys,
        )

        forecast = model(
            context=context,
            target_timestamps=target_timestamps,
            context_timestamps=context_timestamps,
        )
        losses = torch.stack(
            [index_agreement_torch(f, t) for f, t in zip(forecast, target)]
        )
        loss = losses.mean()
        forecast_size += forecast_increase_step
        context_size += content_increase_step

        uniplot.plot(
            xs=[target_timestamps[0].cpu().numpy(), target_timestamps[0].cpu().numpy()],
            ys=[
                forecast[0].squeeze().cpu().detach().numpy(),
                target[0].squeeze().cpu().numpy(),
            ],
            color=True,
            legend_labels=["Forecast", "Target"],
        )
