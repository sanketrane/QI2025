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
    for i != j. This method constructs a continuous-time
    rate matrix Q from its off-diagonal elements.
    The diagonal elements are set such that each row sums to zero.
    Hence they are negative and represent the exit rates from each state.
    
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
    Dynamical model without net loss, and only differentiation.
    Solves the ODE dX/dt = Q X with initial condition X(0) = X0,
    where Q is constructed from its off-diagonal elements Qoffdiag.

    This linear ODE can be solved explicitly using the matrix exponential:
        X(t) = exp(t Q) X0

    where exp(t Q) is the matrix exponential of t Q.

    The tedious part is to get the batch dimensions right.

    We have to be careful: Pyro adds a batch dimension for 
    time to the parameters, and might also add more batch dims 
    to the left for parallel particles. 
    However, the Predictive class does NOT add a batch dim for time.

    Parameters
    ----------
        time: torch.Tensor
            time points at which to evaluate the solution
        X0: torch.Tensor
            initial condition(s) for the ODE
        Qoffdiag: torch.Tensor
            off-diagonal elements of the rate matrix Q

    Returns
    -------
        Xt: torch.Tensor
            solution of the ODE at the specified time points
    """
    batch_shape = X0.shape[:-1]
    batch_dim = functools.reduce(lambda a, b: a*b, batch_shape, 1)

    Q = build_Q_mat(Qoffdiag)
    X0_almost_flat = X0.reshape(batch_dim, 1, X0.shape[-1])
    Q_flat = Q.reshape(batch_dim, *Q.shape[-2:])

    # solve the ODEs explicitly
    At = torch.einsum("...i,...jk->...ijk", time, Q_flat)
    Xt = torch.einsum("...ij,...j->...i", 
                      torch.matrix_exp(At), X0_almost_flat)

    # reshape the output with the correct batch dims
    sol_shape = Xt.shape[-2:]
    # reshape back and handle the time dimension correctly
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
        """
        Dynamic model for cell population dynamics.
        Contains a model method and an automatic guide for 
        variational inference.
        
        The model samples initial population sizes from a Dirichlet
        distribution and the off-diagonal elements of the rate matrix Q
        from Exponential distributions. The ODE is solved explicitly
        using the matrix exponential (see dynamic_model function above).

        Parameters
        ----------
            num_clus: int
                Number of clusters (cell populations)
            init_fn: Callable, optional
                Initialization function for the guide (default is init_to_mean)
            init_scale: float, optional
                Scale for the initialization of the guide (default is 0.1)
            use_cuda: bool, optional
                Whether to use CUDA for computations (default is False)

        Returns
        -------
            None
        """
        super().__init__()
        
        self.num_clus = num_clus
        
        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        # we make use of an automatic guide for variational inference
        # that uses a multivariate normal distribution.
        # This accounts for correlations between the parameters
        # in the posterior distribution.

        self.auto_guide = AutoMultivariateNormal(
            self, init_loc_fn=init_fn, init_scale=init_scale
        )
        
        if use_cuda:
            self.cuda()
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Dynamic model for cell population dynamics.
        Samples initial population sizes from a Dirichlet
        distribution and the off-diagonal elements of the rate matrix Q
        from Exponential distributions. The ODE is solved explicitly
        using the matrix exponential (see dynamic_model function above).
        
        Parameters
        ----------
            time: torch.Tensor
                time points at which to evaluate the solution
        Returns
        -------
            Xt: torch.Tensor
                solution of the ODE at the specified time points
        """

        # sample initial population sizes (cf GMM weights in the static model)

        # use a typical symmetric Dirichlet prior with concentration 0.5
        halves = torch.full((self.num_clus,), 0.5, device=self.device)
        X0 = pyro.sample("X0", dist.Dirichlet(halves))

        # and sample the off-diagonal elements of Q
        shp_Q = (*X0.shape[:-1], self.num_clus-1, self.num_clus)
        # WARNING: hard-coded rate parameter for Exponential prior (mean of 0.1)
        rate_Q = 10*torch.ones(shp_Q, device=self.device)
        Qoffdiag = pyro.sample("Qoffdiag", dist.Exponential(rate_Q).to_event(2))
        
        # solve the ODE explicitly using the matrix exponential
        Xt = dynamic_model(time, X0, Qoffdiag)
        return Xt
    
    
    def guide(self, time: torch.Tensor) -> torch.Tensor:       
        """
        Automatic guide for variational inference.
        Uses the AutoMultivariateNormal class from Pyro.

        We compute the log weights from the guide by tracing the guide
        and then replaying the model with the sampled parameters.

        This allows us to obtain the log weights for further
        use in upstream guides. This is a bit of an anti-pattern,
        but it works for our purposes. 
        This could be solved more elegantly with AutoMessenger 
        guides in more complex scenarios.

        Parameters
        ----------
            time: torch.Tensor
                time points at which to evaluate the solution
        Returns
        -------
            logweights: torch.Tensor
                log weights from the guide
        
        """

        guide_trace = poutine.trace(self.auto_guide).get_trace(time)
        logweights = poutine.block(poutine.replay(self, guide_trace))(time)
                
        return logweights