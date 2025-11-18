"""
Neural Network Layers for use in GMM models
"""

import torch
import torch.nn as nn
from pyro.nn.module import PyroModule


class DecoderX(PyroModule):
    """
    This is the decoder network for X (loc and scale), given Z.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int,
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # parameters
        self.data_dim = data_dim
        # setup the linear transformations used
        self.fc_in = PyroModule[nn.Linear](z_dim, hidden_dim)
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, data_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, data_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

        if use_cuda:
            self.cuda()

    def forward(
        self,
        z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.fc_in(z)
        # apply nonlinearity
        hidden = self.softplus(hidden)
        # location of reconstructed data
        x_loc = self.fc_loc(hidden)
        x_scale = self.softplus(self.fc_scale(hidden))
        return x_loc, x_scale


class EncoderZ(PyroModule):
    """
    This is the encoder network for Z (loc and scale), given X.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int,
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # parameters
        self.data_dim = data_dim
        # setup the n linear transformations used
        self.fc_in = PyroModule[nn.Linear](data_dim, hidden_dim)
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, z_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

        if use_cuda:
            self.cuda()

    def forward(
        self, 
        x: torch.Tensor, 
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the expression vector x
        hidden = self.fc_in(x)
        hidden = self.softplus(hidden)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc_loc(hidden)
        z_scale = self.softplus(self.fc_scale(hidden))
        return z_loc, z_scale



class CondEncoderZ(PyroModule):
    """
    This is the encoder network for Z (loc and scale), given X.
    This is a conditional encoder that also takes time as input.
    """
    def __init__(
        self, 
        z_dim: int, 
        hidden_dim: int, 
        data_dim: int,
        time_scaling: float = 1.0,
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        # parameters
        self.data_dim = data_dim
        # setup the n linear transformations used
        self.fc_in = PyroModule[nn.Linear](data_dim + 1, hidden_dim)
        self.fc_loc = PyroModule[nn.Linear](hidden_dim, z_dim)
        self.fc_scale = PyroModule[nn.Linear](hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        # time scaling factor
        self.time_scaling = time_scaling

        if use_cuda:
            self.cuda()

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # define the forward computation on the expression vector x, 
        # concatenated with time t
        t_scaled = t.unsqueeze(-1) * self.time_scaling
        hidden = self.fc_in(torch.cat([x, t_scaled], dim=-1))
        hidden = self.softplus(hidden)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc_loc(hidden)
        z_scale = self.softplus(self.fc_scale(hidden))
        return z_loc, z_scale
