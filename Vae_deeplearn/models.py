import torch
import torch.nn as nn

class Cell_Wise_VAE(nn.Module):
    """
    A single-modality Variational Autoencoder (VAE) architecture.

    Args:
        input_dim (int): dimensionality of input features
        latent_dim (int): dimensionality of latent representation
        use_mean (bool): if True, skip sampling and use mu directly (deterministic VAE)
    """

    def __init__(self, input_dim=15, latent_dim=3, use_mean=False):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_mean = use_mean

        # Hidden layer sizes scaled with input_dim
        self.h1 = input_dim * 10
        self.h2 = input_dim * 8
        self.h3 = input_dim * 6
        self.h4 = input_dim * 4
        self.h5 = input_dim * 2
        self.h6 = input_dim * 1

        # Encoder body (modality 1)
        self.encoder_body_mod1 = nn.Sequential(
                nn.Linear(self.input_dim, self.h1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.h1, self.h2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.h2, self.h3),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.h3, self.h4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.h4, self.h5),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.h5, self.h6),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Encoder head: outputs 2*latent_dim (mu + log_var)
        self.encoder_head_mod1 = nn.Linear(self.h6, 2 * self.latent_dim)

        # Fusion network (used for the multi-modal extension of the VAE architecture)
        # For one modality, it effectively adds an extra network layer
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        # Decoder body (shared for modality in the mixed VAE variant)
        # Reverses the encoder structure roughly symmetrically
        self.decoder_shared = nn.Sequential(
            nn.Linear(self.latent_dim, self.h6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.h6, self.h5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.h5, self.h4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.h4, self.h3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final reconstruction layer (sigmoid for 0-1 features)
        self.decoder_mod1 = nn.Sequential(
            nn.Linear(self.h3, self.input_dim),
            nn.Sigmoid()
        )

        # Initializing weights from Kaiming uniform distribution
        self._init_weights()

    def _init_weights(self):
        """
        Helper function applying Kaiming initialization to all linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mean, logvar):
        """
        Reparametrization helper: z = mu + std * eps
        Used to allow the gradients to flow through stochastic nodes.
        """
        if self.use_mean:
            return mean
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode_mod(self, body, head, x):
        """
        Helper encoding modality: x -> body -> head -> (mu, log_var)
        """
        h = body(x)
        stats = head(h)
        # First half of stats -> mu, second half -> log_var
        mu, logvar = stats[:, :self.latent_dim], stats[:, self.latent_dim:]
        return mu, logvar

    def decode(self, z_fused):
        """
        Helper decoding the fused latent representation of z_fused:
            z -> shared decoder -> modality-specific final layer
        """
        h = self.decoder_shared(z_fused)
        return {
            "mod1": self.decoder_mod1(h),
        }

    def forward(self, x1=None):
        """
        Forward pass main function:
            1. Encode modality 1
            2. Reparametrize to get the z-representation
            3. Fuse z's (especially for using multiple modalities)
            4. Decode modality 1
            5. Compute the KL divergence
            6. Return reconstructions, KL metric, and encoder statistics.
        """
        mus = {}
        logvars = {}
        zs = []
        kl_sum = 0.0

        def compute_kl(mu, logvar):
            """
            KL divergence between N(mu, sigma) and N(0,1)
            """
            kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return torch.mean(kl_per_sample)

        # Logic for encoding the provided modality
        if x1 is not None:
            mu1, logvar1 = self.encode_mod(self.encoder_body_mod1, self.encoder_head_mod1, x1)
            z1 = self.reparameterize(mu1, logvar1)
            zs.append(z1)
            mus["mod1"], logvars["mod1"] = mu1, logvar1
            kl_sum = kl_sum + compute_kl(mu1, logvar1)

        # Safety check for ensuring minimal requirements
        if len(zs) == 0:
            raise ValueError("No modality has been provided.")

        # Fusion logic: stack z's -> average -> fuse
        z_stack = torch.stack(zs, dim=0)
        z_mean = torch.mean(z_stack, dim=0)
        z_fused = self.fusion_proj(z_mean)

        # Decode shared latent
        recons = self.decode(z_fused)

        total_kl = kl_sum
        return recons, total_kl, mus, logvars

class VAE_Single_Dataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for VAE input.
    """
    def __init__(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx]
        # Ensuring the shape is always (1, input_dim) if input is flattened.
        if sample.dim() == 1:
            sample = sample.unsqueeze(0)
        return sample