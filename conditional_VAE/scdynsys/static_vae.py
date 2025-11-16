import pyro
import torch
import pyro.distributions as dist
from pyro.nn.module import PyroModule
from typing import Tuple, Literal
from pyro.infer import Predictive
from pyro.distributions.kl import kl_divergence

from scdynsys.mixture_model import GaussMix
from scdynsys.nets import DecoderX, EncoderZ

class VAEgmm(PyroModule):
    def __init__(
        self, 
        data_dim: int, 
        z_dim: int, 
        hidden_dim: int,
        num_clus: int, 
        use_cuda: bool = False,
        init_fn = None
    ) -> None:
        super().__init__()

        self.device = torch.device("cuda") if use_cuda else torch.device("cpu")

        ## dims and settings
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.num_clus = num_clus

        self.decoder_x = DecoderX(
            z_dim, 
            hidden_dim, 
            data_dim, 
            use_cuda=use_cuda            
        )
        self.encoder_z = EncoderZ(
            z_dim, 
            hidden_dim, 
            data_dim, 
            use_cuda=use_cuda            
        )

        # the mixture model
        self.mix = GaussMix(z_dim, num_clus, weighted=True, use_cuda=use_cuda, init_fn=init_fn)
        
        if use_cuda: ## put tensors into GPU RAM
            self.cuda()


    # define the model p(x|z)p(z)
    def model(
        self,
        x: torch.Tensor, ## expression data (mini-batches)
        N: torch.Tensor, ## as we're doing mini-batching, we have to re-scale the log prob
    ) -> None:
        ## use GaussMix object to sample parameters for the mixture model
        clus_locs, clus_chol_fact, weights = self.mix()
        logweights = torch.log(weights + 1e-10)

        ## penalty for overlapping distributions
        comp_dists = [
            dist.MultivariateNormal(
                clus_locs[...,c,:], 
                scale_tril=clus_chol_fact[...,c,:,:]
            ) for c in range(self.num_clus)
        ]
        
        kl_dists = torch.stack([
            kl_divergence(comp_dists[i], comp_dists[j]) 
            for i in range(self.num_clus)
            for j in range(self.num_clus)
            if i != j
        ])

        pyro.factor("penalty", -(1/kl_dists).sum())
                
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
        x: torch.Tensor, ## expression data (mini-batches)
        N: torch.Tensor, ## as we're doing mini-batching, we have to re-scale the log prob
    ) -> None:
        ## use the guide method of GaussMix
        clus_locs, clus_chol_fact, weights = self.mix.guide()
            
        # use the encoder to get the parameters used to define q(z|x)
        z_loc, z_scale = self.encoder_z(x)
        
        # sample latent vectors
        with pyro.plate("unobserved", x.shape[-2]), pyro.poutine.scale(scale=N/x.shape[-2]):
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


    # define a helper function for reconstructing samples
    def reconstruct_sample(self, x: torch.Tensor) -> torch.Tensor:
        # encode sample x
        z_loc, z_scale = self.encoder_z(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the latent sample
        x_loc, x_scale = self.decoder_x(z)
                
        x = dist.Normal(x_loc, x_scale).sample()
        return x

    # use the VAE as a dimension reduction algo
    def dimension_reduction(self, x: torch.Tensor) -> torch.Tensor:
        # encode sample x
        z_loc, z_scale = self.encoder_z(x)
        return z_loc

    # use the VAE as a classifier
    def classifier(
        self,
        x: torch.Tensor,
        method: Literal["map", "sample"]
    ) -> torch.Tensor:
        # encode sample x
        z_loc, z_scale = self.encoder_z(x)
        locs, chols, weights = self.mix.guide()
        logweights = torch.log(weights + 1e-10)
        
        loglikes = torch.t(torch.cat([
            dist.MultivariateNormal(locs[c], scale_tril=chols[c]).log_prob(z_loc).unsqueeze(0)
            for c in range(self.num_clus)
        ]))
        logodds = loglikes + logweights
        match method:
            case "map":
                return torch.argmax(logodds, dim=-1)
            case "sample":
                return dist.Categorical(logits=logodds).sample()
            case _:
                raise Exception("invalid sampling method")

    def posterior_sample(
        self,
        n: int = 1000, ## number of samples
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Take a posterior data sample at a given time point.
        """

        zzs, xxs = [], []
                   
        
        def gmm_model_wrap():
            locs, chols, weights = self.mix()
            pyro.deterministic("locs", locs)
            pyro.deterministic("chols", chols)
            pyro.deterministic("weights", weights)

        pred_gmm = Predictive(gmm_model_wrap, guide=self.mix.guide, num_samples=n, parallel=True)
        sam_gmm = pred_gmm()

        logweights = torch.log(sam_gmm["weights"] + 1e-10)
        clus_vec = dist.Categorical(logits=logweights).sample()
        
        locs = sam_gmm["locs"]
        chols = sam_gmm["chols"]
            
        sel_locs = locs.gather(1, clus_vec.view(-1,1,1).expand(-1,1,locs.shape[2])).squeeze(-2)
        sel_chols = chols.gather(1, clus_vec.view(-1,1,1,1).expand(-1,1,*chols.shape[2:])).squeeze(-3)

        zz = dist.MultivariateNormal(sel_locs, scale_tril=sel_chols).sample()
            
        # decode z into x
        xx_loc, xx_scale = self.decoder_x(zz)
        xx = dist.Normal(xx_loc, xx_scale).to_event(1).sample()
        
        return zz, xx, clus_vec
    

