import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        #from IPython import embed; embed(); sys.exit()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class CNNSharedNet(nn.Module):
    def __init__(self, observation_space):
        super(CNNSharedNet, self).__init__()
        input_shape = observation_space.shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(observation_space)

    def _get_conv_out(self,observation_space):
        shape = observation_space.shape
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return conv_out

class CNNGaussianActor(Actor):

    def __init__(self, shared_net, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.cnn_mu_net = nn.Sequential(shared_net, self.mu_net)

    def _distribution(self, obs):
        mu = self.cnn_mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class CNNCritic(nn.Module):

    def __init__(self, shared_net, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
        self.cnn_v_net = nn.Sequential(shared_net, self.v_net)

    def forward(self, obs):
        return torch.squeeze(self.cnn_v_net(obs), -1) # Critical to ensure v has right shape.



class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        # shared network
        self.shared = CNNSharedNet(observation_space)
        obs_dim = self.shared._get_conv_out(observation_space)
        ## from IPython import embed; embed(); sys.exit()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(self.shared, obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            return NotImplementedError
            ## self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = CNNCritic(self.shared, obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            ## shared_obs = self.shared(obs.unsqueeze(0)).squeeze()
            ## pi = self.pi._distribution(shared_obs)
            pi = self.pi._distribution(obs.unsqueeze(0))
            a = pi.sample().squeeze()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            ## v = self.v(shared_obs)
            v = self.v(obs.unsqueeze(0)).squeeze()
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]




class MultimodalSharedNet(nn.Module):
    def __init__(self, modal_dims):
        super(MultimodalSharedNet, self).__init__()
        self.modal_dims = modal_dims
        ## self.output_dim = 16
        self.img_width = self.img_height = int(np.sqrt(modal_dims[0]))
        
        ## input_shape = observation_space.shape
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(modal_dims[1], 64 ),
            nn.ReLU()
            )        

    def _get_out_dims(self):
        shape = (1, self.img_width, self.img_height)
        o1 = self.conv(torch.zeros(1, *shape))
        o2 = self.mlp(torch.zeros(1, self.modal_dims[1]))
        return (int(np.prod(o1.size())), int(np.prod(o2.size())))

    def get_nets(self):
        return (self.conv, self.mlp)

    def forward(self, x1, x2):
        return self.conv(x1).view(x1.size()[0], -1), self.mlp(x2).view(x2.size()[0], -1)
        

class n_to_one(nn.Module):
    def __init__(self, obs_dims, act_dim, hidden_sizes, activation):
        super().__init__()
        self.mlp1 = mlp([obs_dims[0]] + [hidden_sizes[0]], activation)
        self.mlp2 = mlp([hidden_sizes[0]+obs_dims[-1]]+[hidden_sizes[-1], act_dim], activation)

    def forward(self, x1, x2):
        x1n = self.mlp1(x1)
        combined = torch.cat((x1n.view(x1n.size(0), -1),
                              x2.view(x2.size(0), -1)), dim=1)
        out = self.mlp2(combined)
        return out

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

class MultimodalGaussianActor(Actor):

    def __init__(self, shared_net, obs_dims, act_dim, hidden_sizes, activation, modal_dims):
        super().__init__()
        obs_dim1, obs_dim2 = obs_dims
        self.modal_dims = modal_dims
        self.img_width = self.img_height = int(np.sqrt(modal_dims[0]))
        
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # n_to_n -> n_to_one
        fc = n_to_one(obs_dims, act_dim, hidden_sizes, activation)
        self.mu_net = mySequential(shared_net, fc)

    def _distribution(self, obs):       
        # split the input
        ## from IPython import embed; embed(); sys.exit()
        x1 = obs[:,:-self.modal_dims[-1]]
        x1 = np.reshape(x1, (-1, 1, self.img_width,self.img_height))
        x2 = obs[:,-self.modal_dims[-1]:]
        
        mu = self.mu_net(x1, x2)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MultimodalCritic(nn.Module):

    def __init__(self, shared_net, obs_dims, hidden_sizes, activation, modal_dims):
        super().__init__()
        obs_dim1, obs_dim2 = obs_dims
        self.modal_dims = modal_dims
        self.img_width = self.img_height = int(np.sqrt(modal_dims[0]))

        # n_to_n -> n_to_one
        fc = n_to_one(obs_dims, 1, hidden_sizes, activation)
        self.v_net = mySequential(shared_net, fc)

    def forward(self, obs):
        x1 = obs[:, :-self.modal_dims[-1]]
        x1 = np.reshape(x1, (-1, 1, self.img_width,self.img_height))
        x2 = obs[:, -self.modal_dims[-1]:]        
        ## from IPython import embed; embed(); sys.exit()
        return torch.squeeze(self.v_net(x1, x2), -1) # Critical to ensure v has right shape.


class MultimodalActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        ## from IPython import embed; embed(); sys.exit()

        shape = observation_space.shape
        modal_dims = [64**2, shape[0]-64**2]
        ## modal_dims = [shape[0]-2, 2]

        # shared network
        self.shared_net = MultimodalSharedNet(modal_dims)
        ## shared_nets = self.shared_net.get_nets()
        obs_dims = self.shared_net._get_out_dims()
        ## from IPython import embed; embed(); sys.exit()

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MultimodalGaussianActor(self.shared_net, obs_dims,
                                              action_space.shape[0],
                                              hidden_sizes, activation,
                                              modal_dims)
        elif isinstance(action_space, Discrete):
            return NotImplementedError

        # build value function
        self.v  = MultimodalCritic(self.shared_net, obs_dims,
                                   hidden_sizes, activation,
                                   modal_dims)

    def step(self, obs):
        with torch.no_grad():
            # split obs into two modality data
            pi = self.pi._distribution(obs.unsqueeze(0))
            a = pi.sample().squeeze()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            ## v = self.v(shared_obs)
            v = self.v(obs.unsqueeze(0)).squeeze()
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
