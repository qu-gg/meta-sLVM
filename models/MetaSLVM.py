import torch
import functorch
import torch.nn as nn

from torchdiffeq import odeint
from utils.utils import get_act
from utils.layers import Gaussian
from models.CommonVAE import LatentDomainEncoder
from models.CommonDynamics import LatentMetaDynamicsModel
from torch.distributions import Normal, kl_divergence as kl
from hypernn.torch.utils import FunctionalParamVectorWrapper, count_params


class ODE(nn.Module):
    def __init__(self, cfg):
        super(ODE, self).__init__()
        self.cfg = cfg

        """ Main Network """
        dynamics_network = []
        dynamics_network.extend([
            nn.Linear(cfg.layers.latent_dim, cfg.layers.num_hidden),
            get_act(cfg.layers.latent_act)
        ])

        for _ in range(cfg.layers.num_layers - 1):
            dynamics_network.extend([
                nn.Linear(cfg.layers.num_hidden, cfg.layers.num_hidden),
                get_act(cfg.layers.latent_act)
            ])

        dynamics_network.extend([nn.Linear(cfg.layers.num_hidden, cfg.layers.latent_dim), nn.Tanh()])
        dynamics_network = nn.Sequential(*dynamics_network)
        self.dynamics_network = FunctionalParamVectorWrapper(dynamics_network)

        """ Hyper Network """
        # Domain encoder for z_c
        self.domain_encoder = LatentDomainEncoder(cfg, cfg.generation_length)
        self.gaussian = Gaussian(cfg.layers.code_dim, cfg.layers.code_dim)

        # Hypernetwork going from the embeddings to the full main-network weights
        self.hypernet = nn.Linear(cfg.layers.code_dim, count_params(dynamics_network))
        nn.init.normal_(self.hypernet.weight, 0, 0.01)
        nn.init.zeros_(self.hypernet.bias)

    def sample_embeddings(self, D):
        """ Given a batch of data points, embed them into their C representations """
        # Reshape to batch get the domain encodings
        domain_size = D.shape[1]
        D = D.reshape([D.shape[0] * domain_size, -1, self.cfg.dim, self.cfg.dim])

        # Get domain encoder outputs
        embeddings = self.domain_encoder(D)

        # Reshape to batch and take the average C over each sample
        embeddings = embeddings.view([D.shape[0], domain_size, self.cfg.layers.code_dim])
        embeddings = embeddings.mean(dim=[1])

        # From this context set mean, get the distributional parameters
        if self.cfg.stochastic is True:
            _, _, embeddings = self.gaussian(embeddings)
        return embeddings

    def sample_weights(self, x, D):
        """ Given a batch of data points, embed them into their C representations """        
        # Combine the query and context samples for process efficiency before splitting later
        D = torch.concat((x.unsqueeze(1), D), dim=1)
        domain_size = D.shape[1]

        # Reshape to batch get the domain encodings
        D = D.reshape([D.shape[0] * domain_size, -1, self.cfg.dim, self.cfg.dim])

        # Get domain encoder outputs
        self.embeddings = self.domain_encoder(D)

        # Reshape to batch and take the average C over each sample
        self.embeddings = self.embeddings.view([x.shape[0], domain_size, self.cfg.layers.code_dim])

        # Separate into batch usage and kl usage
        self.embeddings, embeddings_kl = self.embeddings[:, 1:], self.embeddings
        self.embeddings = self.embeddings.mean(dim=[1])
        embeddings_kl = embeddings_kl.mean(dim=[1])

        # From this context set mean, get the distributional parameters
        if self.cfg.stochastic is True:
            self.embeddings_mu, self.embeddings_var, self.embeddings = self.gaussian(self.embeddings)
            self.embeddings_kl_mu, self.embeddings_kl_var, _ = self.gaussian(embeddings_kl)

        # Get weight outputs from hypernetwork
        self.params = self.hypernet(self.embeddings)

    def forward(self, t, z):
        """ Wrapper function for the odeint calculation """
        return functorch.vmap(self.dynamics_network)(self.params, z)


class MetaSLVM(LatentMetaDynamicsModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # ODE-Net which holds mixture logic
        self.dynamics_func = ODE(cfg)

    def forward(self, x, D):
        # Sample z_init
        z_init = self.encoder(x)

        # Evaluate model forward over T to get L latent reconstructions
        t = torch.linspace(0, self.cfg.generation_length - 1, self.cfg.generation_length).to(self.device)

        # Draw weights
        self.dynamics_func.sample_weights(x, D[:, :, :self.cfg.generation_length])

        # Evaluate forward over timestep
        zt = odeint(self.dynamics_func, z_init, t, method="rk4", options={"step_size": 0.5})
        zt = zt.reshape([self.cfg.generation_length, x.shape[0], self.cfg.layers.latent_dim])
        zt = zt.permute([1, 0, 2])

        # Stack zt and decode zts
        x_rec = self.decoder(zt)
        return x_rec, zt

    def model_specific_loss(self, x, D, preds):
        """ A standard KL prior is put over the weight codes of the hyper-prior to encourage good latent structure """
        # Skip the loss is the most is deterministic
        if self.cfg.stochastic is False:
            return 0.0

        # Get flattened mus and vars
        embed_mus, embed_vars = self.dynamics_func.embeddings_mu.view([-1]), self.dynamics_func.embeddings_var.view([-1])

        # KL on C with a prior of Normal
        q = Normal(embed_mus, torch.exp(0.5 * embed_vars))
        N = Normal(torch.zeros(len(embed_mus), device=embed_mus.device),
                   torch.ones(len(embed_mus), device=embed_mus.device))

        kl_c_normal = self.cfg.betas.kl * kl(q, N).view([x.shape[0], -1]).sum([1]).mean()
        self.log("kl_c_normal", kl_c_normal, prog_bar=True)

        # KL on C with a prior of the context set with itself in it
        context_mus, context_vars = self.dynamics_func.embeddings_kl_mu.view([-1]), self.dynamics_func.embeddings_kl_var.view([-1])
        q = Normal(embed_mus, torch.exp(0.5 * embed_vars))
        N = Normal(context_mus, torch.exp(0.5 * context_vars))

        kl_c_context = self.cfg.betas.kl * kl(q, N).view([x.shape[0], -1]).sum([1]).mean()
        self.log("kl_c_context", kl_c_context, prog_bar=True)

        # Return them as one loss
        return kl_c_normal + kl_c_context
