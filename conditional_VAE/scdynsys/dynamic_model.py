import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.infer.autoguide.initialization import init_to_mean
from pyro import poutine
import functools

from typing import Callable



def build_Q_mat(Qoffdiag: torch.Tensor) -> torch.Tensor:
    """
    Construct a stochastic matrix from a matrix of entry-rates Q_ij 
    for i != j.
    
    Parameters
    ----------
        Qoffdiag: torch.Tensor
            off-diagonal elements of stochastic matrix
    
    Returns
    -------
        Q: a square, stochastic matrix (continuous time) Q
    """
    n = Qoffdiag.shape[-1]
    Q = torch.zeros((*Qoffdiag.shape[:-2], n, n), device=Qoffdiag.device)
    torch.diagonal(Q, dim1=-2, dim2=-1)[..., :] = -Qoffdiag.sum(axis=-2)
    Q[..., :-1, :] = Q[..., :-1, :] + torch.triu(Qoffdiag, diagonal=1)
    Q[..., 1:, :] = Q[..., 1:, :] + torch.tril(Qoffdiag)
    return Q




def dynamic_model(
    time: torch.Tensor,
    X0: torch.Tensor,
    Qoffdiag: torch.Tensor
) -> torch.Tensor:
    """
    Dynamical model allowing for differentiation. 
    Parameters are constant.

    We have to be careful to get the batch dimension right.
    Pyro adds a batch dimension for time to the parameters,
    and might also add more batch dims to the left for parallel
    particles. However, Predictive does not add a batch dim for time.
    """
    batch_shape = X0.shape[:-1]
    batch_dim = functools.reduce(lambda a, b: a*b, batch_shape, 1)

    Q = build_Q_mat(Qoffdiag)
    X0_almost_flat = X0.reshape(batch_dim, 1, X0.shape[-1])
    Q_flat = Q.reshape(batch_dim, *Q.shape[-2:])

    # solve the ODEs explicitly
    At = torch.einsum("...i,...jk->...ijk", time, Q_flat)
    Xt = torch.einsum("...ij,...j->...i", torch.matrix_exp(At), X0_almost_flat)

    # reshape the output with the correct batch dims
    sol_shape = Xt.shape[-2:]
    if len(batch_shape) > 0 and batch_shape[-1] == 1:
        Xt = Xt.reshape(*batch_shape[:-1], *sol_shape)
    else:
        Xt = Xt.reshape(*batch_shape, *sol_shape)

    return Xt




class DynamicModel(PyroModule):
    def __init__(
        self, 
        num_clus: int, 
        init_fn: Callable = init_to_mean,
        init_scale: float = 0.1,
        use_cuda: bool = False
    ) -> None:
        super().__init__()
        
        self.num_clus = num_clus
        
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
           
        self.auto_guide = AutoMultivariateNormal(
            self, 
            init_loc_fn=init_fn, 
            init_scale=init_scale
        )
        
        if use_cuda:
            self.cuda()
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # sample initial population sizes (cf GMM weights in the static model)

        X0 = pyro.sample(
            "X0",
            dist.Dirichlet(torch.full((self.num_clus,), 0.5, device=self.device))
        )

        # and sample the off-diagonal elements of Q
        shp_Q = (*X0.shape[:-1], self.num_clus-1, self.num_clus)
        rate_Q = 10*torch.ones(shp_Q, device=self.device)
        Qoffdiag = pyro.sample("Qoffdiag", dist.Exponential(rate_Q).to_event(2))
        
        Xt = dynamic_model(time, X0, Qoffdiag)
        return Xt
    
    
    def guide(self, time: torch.Tensor) -> torch.Tensor:       
        guide_trace = poutine.trace(self.auto_guide).get_trace(time)
        logweights = poutine.block(poutine.replay(self, guide_trace))(time)
                
        return logweights