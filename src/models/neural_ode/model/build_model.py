from model.core.mlp import MLP
from model.core.flow import Flow
from model.core.vae import VAE
from model.core.inv_enc import INV_ENC
from model.core.model import Node_Forecaster


def build_model(
    modulator_dim: int,
    Nobj: int,
    ode_latent_dim: int,
    order: int,
    de_L: int,
    de_H: int,
    dec_act: str,
    rnn_hidden: int,
    dec_H: int,
    enc_H: int,
    content_dim: int,
    T_in: int,
    T_inv: int,
    solver: str,
    dt: float,
    use_adjoint: str,
    device: str,
    dtype: str,
    data_dim: int,
    **params,
):
    """
    Builds a model object of monode.MoNODE based on training sequence

    @param args: model setup arguments
    @param device: device on which to store the model (cpu or gpu)
    @param dtype: dtype of the tensors
    @param params: dict of data properties (see config.yml)
    @return: an object of MoNODE class
    """
    # define training set-up
    aug = modulator_dim > 0
    Nobj = Nobj

    D_in = ode_latent_dim
    D_out = int(ode_latent_dim / order)

    if aug:  # augmented dynamics
        D_in += modulator_dim

    de = MLP(D_in, D_out, L=de_L, H=de_H, act="softplus")

    flow = Flow(diffeq=de, order=order, solver=solver, use_adjoint=use_adjoint)

    # encoder & decoder
    vae = VAE(
        ode_latent_dim=ode_latent_dim // order,
        dec_act=dec_act,
        rnn_hidden=rnn_hidden,
        dec_H=dec_H,
        enc_H=enc_H,
        content_dim=content_dim,
        order=order,
        device=device,
        data_dim=data_dim,
    ).to(dtype)

    # time-invariant network
    if modulator_dim > 0 or content_dim > 0:
        inv_enc = INV_ENC(
            modulator_dim=modulator_dim,
            content_dim=content_dim,
            rnn_hidden=10,
            T_inv=T_inv,
            data_dim=data_dim,
            device=device,
        ).to(dtype)
    else:
        inv_enc = None

    # full model
    monode = Node_Forecaster(
        flow=flow,
        vae=vae,
        inv_enc=inv_enc,
        order=order,
        dt=dt,
        aug=aug,
        nobj=Nobj,
        Tin=T_in,
    )

    return monode
