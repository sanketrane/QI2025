import pyro
import torch
import pyro.distributions as dist
from pyro.nn.module import PyroModule
from typing import Tuple, Literal
from pyro.infer import Predictive

from scdynsys.mixture_model import GaussMix
from scdynsys.dynamic_model import DynamicModel
from scdynsys.nets import DecoderX, CondEncoderZ

class VAEgmmdyn(PyroModule):
    def __init__(
        self, 
        data_dim: int, 
        z_dim: int, 
        hidden_dim: int,
        num_clus: int,
        time_scaling: float = 1.0,
        use_cuda: bool = False,
        init_fn = None
    ) -> None:
        """
        A conditional VAE with Gaussian mixture model prior and dynamic model 
        on the mixture weights.

        Parameters
        ----------
        data_dim : int
            The dimensionality of the input data.
        z_dim : int
            The dimensionality of the latent space.
        hidden_dim : int
            The dimensionality of the hidden layers in the encoder and decoder.
        num_clus : int
            The number of clusters in the Gaussian mixture model.
        time_scaling : float, optional
            A scaling factor for the time input to the encoder, by default 1.0
        use_cuda : bool, optional
            Whether to use CUDA for GPU acceleration, by default False.
        init_fn : callable, optional
            An initialization function for the mixture model parameters, by default None.

        Returns
        -------
        None
        """
        
        # initialize the PyroModule (cf torch nn.Module initialization)
        super().__init__()

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        ## dims and settings
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.num_clus = num_clus

        # define the encoder and decoder networks
        self.decoder_x = DecoderX(
            z_dim, 
            hidden_dim, 
            data_dim, 
            use_cuda=use_cuda            
        )
        self.encoder_z = CondEncoderZ(
            z_dim, 
            hidden_dim, 
            data_dim,
            time_scaling=time_scaling, 
            use_cuda=use_cuda            
        )

        # the mixture model
        self.mix = GaussMix(
            z_dim, num_clus, weighted=False, 
            use_cuda=use_cuda, init_fn=init_fn
        )

        # the dynamic model
        self.dynamic_model = DynamicModel(num_clus, use_cuda=use_cuda)
        
        if use_cuda: ## put tensors into GPU RAM
            self.cuda()


    # define the model p(x|z)p(z)
    def model(
        self,
        x: torch.Tensor, ## expression data (mini-batches)
        xtime: torch.Tensor, ## time index for each data point
        utime: torch.Tensor, ## unique time points
        N: torch.Tensor, ## as we're doing mini-batching, we have to re-scale the log prob
    ) -> None:
        """
        The model definition p(x|z)p(z). 
        This also includes the dynamic model and GMM prior.

        Parameters
        ----------
        x : torch.Tensor
            The expression data (possibly mini-batches).
        xtime : torch.Tensor
            The time index for each data point.
        utime : torch.Tensor
            The unique time points.
        N : torch.Tensor
            The total number of samples (for mini-batch scaling).
        Returns
        -------
        None
        """

        ## use GaussMix object to sample parameters for the mixture model
        clus_locs, clus_chol_fact = self.mix()
        weights = self.dynamic_model(utime)[..., xtime, :]
        logweights = torch.log(weights + 1e-10)
                
        with pyro.plate("unobserved", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            # integrate out cluster
            mix = dist.Categorical(logits=logweights)
            comp = dist.MultivariateNormal(clus_locs, scale_tril=clus_chol_fact)
            z = pyro.sample("latent", dist.MixtureSameFamily(mix, comp))

        ## decode the latent code z outsize of the plate (module could be Bayesian)
        x_loc, x_scale = self.decoder_x(z)
                
        with pyro.plate("xdata", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            ## score against actual expression data
            pyro.sample("xobs", dist.Normal(x_loc, x_scale).to_event(1), obs=x)
                            

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(
        self,
        x: torch.Tensor,
        xtime: torch.Tensor,
        utime: torch.Tensor,
        N: torch.Tensor,
    ) -> None:
        """
        The guide (i.e. variational distribution) q(z|x).
        Notice that the dynamic model and mixture model guides 
        are also called here. The arguments to the guide functions
        are the same as those to the model functions, as required by Pyro.

        Parameters
        ----------
        x : torch.Tensor
            The expression data (possibly mini-batches).
        xtime : torch.Tensor
            The time index for each data point.
        utime : torch.Tensor
            The unique time points.
        N : torch.Tensor
            The total number of samples (for mini-batch scaling 
            of the log probability).
        Returns
        -------
        None       
        
        """
        ## use the guide method of GaussMix
        clus_locs, clus_chol_fact = self.mix.guide()
        weights = self.dynamic_model.guide(utime)
            
        # use the encoder to get the parameters used to define q(z|x)
        z_loc, z_scale = self.encoder_z(x, utime[xtime])
        
        # sample latent vectors
        with pyro.plate("unobserved", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


    # define a helper function for reconstructing samples
    @torch.no_grad()
    def reconstruct_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a sample x given time t.
        This is done by encoding x into the latent space, 
        sampling from the latent distribution,
        and then decoding back into the data space.

        Parameters
        ----------
        x : torch.Tensor
            The input data sample.
        t : torch.Tensor
            The time index for the data sample.
        Returns
        -------
        x : torch.Tensor
            The reconstructed data sample.
        """
        # encode sample x
        z_loc, z_scale = self.encoder_z(x, t)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the latent sample
        x_loc, x_scale = self.decoder_x(z)
                
        x = dist.Normal(x_loc, x_scale).sample()
        return x

    # use the VAE as a dimension reduction algo
    @torch.no_grad()
    def dimension_reduction(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform dimension reduction on sample x given time t.
        This is done by encoding x into the latent space and
        returning the mean of the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            The input data sample.
        t : torch.Tensor
            The time index for the data sample.

        Returns
        -------
        z_loc : torch.Tensor
            The mean of the latent distribution 
            (dimension reduced representation).        
        """
        # encode sample x
        z_loc, z_scale = self.encoder_z(x, t)
        return z_loc

    # use the VAE as a classifier
    @torch.no_grad()
    def classifier(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify sample x given time t into one of the mixture components.
        This is done by encoding x into the latent space, computing the
        log-odds of each mixture component, and then sampling from the 
        categorical distribution defined by the log-odds.

        Parameters
        ----------
        x : torch.Tensor
            The input data sample.
        t : torch.Tensor
            The time index for the data sample.

        Returns
        -------
        clus_vec : torch.Tensor
            The assigned cluster indices for each data point.
        
        """

        # encode sample x
        z_loc, z_scale = self.encoder_z(x, t)
        # alternatively, you could sample z from the variational distribution
        # but we use the mean here for simplicity

        # sample from the mixture model to get the parameters
        locs, chols = self.mix.guide()
        # get the dynamic model weights
        weights = self.dynamic_model.guide(t)
        logweights = torch.log(weights + 1e-10)
        
        # compute log-odds for each cluster using the log-probabilities \
        # of the mixture components at the encoded latent locations
        loglikes = torch.t(torch.cat([
            dist.MultivariateNormal(locs[c], scale_tril=chols[c]).log_prob(z_loc).unsqueeze(0)
            for c in range(self.num_clus)
        ]))

        # add prior log-probabilities from the dynamic model
        logodds = loglikes + logweights

        # classify by sampling from the categorical distribution 
        # defined by the log-odds
        return dist.Categorical(logits=logodds).sample()
    
    @torch.no_grad()
    def posterior_sample(
        self,
        t: torch.Tensor, ## time point
        n: int = 1000, ## number of samples
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take a posterior data sample at a given time point.
        This can be used to simulate / interpolate data at that time.

        Parameters
        ----------
        t : torch.Tensor
            The time index for the data sample.
        n : int, optional
            The number of samples to draw, by default 1000.

        Returns
        -------
        zzs : torch.Tensor
            Samples from the latent space.
        xxs : torch.Tensor
            Samples from the observed data space.
        clus_vec : torch.Tensor
            Cluster assignments for each sample.
        """

        zzs, xxs = [], []
                   
        # define wrapped model and guide functions for Predictive
        # to get samples from the GMM and dynamic model
        def gmm_model_wrap():
            locs, chols = self.mix()
            pyro.deterministic("locs", locs)
            pyro.deterministic("chols", chols)

        def dyn_model_wrap():
            weights = self.dynamic_model.guide(t)
            pyro.deterministic("weights", weights)

        pred_gmm = Predictive(
            gmm_model_wrap, guide=self.mix.guide, 
            num_samples=n, parallel=True
        )
        sam_gmm = pred_gmm()

        pred_dyn = Predictive(
            dyn_model_wrap, guide=self.dynamic_model.guide, 
            num_samples=n, parallel=True
        )
        sam_dyn = pred_dyn(t)

        # sample cluster assignments
        logweights = torch.log(sam_dyn["weights"] + 1e-10)
        clus_vec = dist.Categorical(logits=logweights).sample()
        
        locs = sam_gmm["locs"]
        chols = sam_gmm["chols"]
            
        sel_locs = locs.gather(1, clus_vec.view(-1,1,1).expand(-1,1,locs.shape[2])).squeeze(-2)
        sel_chols = chols.gather(1, clus_vec.view(-1,1,1,1).expand(-1,1,*chols.shape[2:])).squeeze(-3)

        # sample z from selected mixture component
        zz = dist.MultivariateNormal(sel_locs, scale_tril=sel_chols).sample()
            
        # decode z into x
        xx_loc, xx_scale = self.decoder_x(zz)
        xx = dist.Normal(xx_loc, xx_scale).to_event(1).sample()
        
        # return latent sample, decoded sample, and cluster assignments
        return zz, xx, clus_vec
    

    @torch.no_grad()
    def sample_trajectories(
        self, 
        ts: torch.Tensor, 
        n: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sample trajectories of the dynamic model with parameters
        sampled from the variational distribution.

        Parameters
        ----------
        ts : torch.Tensor
            The time indices at which to sample the trajectories.
        n : int, optional
            The number of trajectory samples to draw, by default 1000.

        Returns
        -------
        ws : torch.Tensor
            The sampled mixture weights at each time point.
        """
        
        def traj_model(ts):
            weights = self.dynamic_model(ts)
            pyro.deterministic("weights", weights)

        def traj_guide(ts):
            weights = self.dynamic_model.guide(ts)
        
        pred = pyro.infer.Predictive(
            traj_model, guide=traj_guide, 
            num_samples=n, parallel=True
        )
        sams = pred(ts)

        # predictive adds a batch dimension to the samples
        ws = sams["weights"].squeeze(1)

        return ws

