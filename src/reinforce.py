"""
REINFORCE Algorithm Implementation for Continuous Action Spaces

This module implements the REINFORCE policy gradient algorithm for reinforcement
learning with continuous action spaces. REINFORCE is a Monte Carlo policy gradient
method that directly optimizes the policy parameters by maximizing expected return.

The algorithm:
1. Collects a complete episode of experience
2. Computes discounted returns for each time step
3. Updates policy parameters using policy gradient theorem

Based on the Gymnasium REINFORCE tutorial for Inverted Pendulum:
https://gymnasium.farama.org/v0.27.0/tutorials/reinforce_invpend_gym_v26/

Example:
    >>> import gymnasium as gym
    >>> import numpy as np
    >>>
    >>> # Create environment and agent
    >>> env = gym.make('InvertedPendulum-v4')
    >>> agent = REINFORCE(obs_space_dims=4, action_space_dims=1)
    >>>
    >>> # Training loop
    >>> for episode in range(1000):
    ...     obs, _ = env.reset()
    ...     done = False
    ...
    ...     while not done:
    ...         action = agent.sample_action(obs)
    ...         obs, reward, done, _, _ = env.step(action)
    ...         agent.rewards.append(reward)
    ...
    ...     agent.update()

Author: vnniciusg
"""

import numpy as np
import torch
from torch.distributions.normal import Normal
from torch.optim import AdamW

from .police_network import PolicyNetwork


class REINFORCE:
    """
    REINFORCE Policy Gradient Algorithm for Continuous Action Spaces.

    This class implements the REINFORCE algorithm, which is a Monte Carlo policy
    gradient method. It learns a policy that directly maps states to action
    distributions and updates policy parameters to maximize expected return.

    Attributes:
        learning_rate (float): Learning rate for the optimizer (default: 1e-4)
        gamma (float): Discount factor for future rewards (default: 0.99)
        epsilon (float): Small constant added for numerical stability (default: 1e-6)
        probs (list): Stores log probabilities of actions taken during episode
        rewards (list): Stores rewards received during episode
        policy_net (PolicyNetwork): Neural network that outputs action distribution parameters
        optimizer (torch.optim.AdamW): Optimizer for updating policy network parameters
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int) -> None:
        """
        Initialize the REINFORCE agent.

        Args:
            obs_space_dims (int): Dimensionality of the observation space
            action_space_dims (int): Dimensionality of the action space
        """
        self.learning_rate: float = 1e-4
        self.gamma = 0.99
        self.epsilon = 1e-6

        self.probs, self.rewards = [], []

        self.policy_net: PolicyNetwork = PolicyNetwork(
            obs_space_dims, action_space_dims
        )
        self.optimizer = AdamW(self.policy_net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.array) -> float:
        """
        Sample an action from the policy network given a state.

        This method:
        1. Converts the state to a torch tensor
        2. Gets action distribution parameters from the policy network
        3. Creates a Normal distribution and samples an action
        4. Stores the log probability for later policy updates

        Args:
            state (np.array): Current state/observation from the environment

        Returns:
            float: Sampled action value
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.policy_net(state)

        distribution = Normal(
            action_means[0] + self.epsilon, action_stddevs[0] + self.epsilon
        )
        action = distribution.sample()
        prob = distribution.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """
        Update the policy network parameters using the REINFORCE algorithm.

        This method implements the core REINFORCE update:
        1. Computes discounted returns (G_t) for each time step in the episode
        2. Calculates policy gradient loss using stored log probabilities
        3. Performs backpropagation and parameter update
        4. Clears stored probabilities and rewards for the next episode

        The policy gradient theorem states that:
        ∇J(θ) = E[∇log π(a|s,θ) * G_t]

        Where G_t is the discounted return from time t.
        """
        running_g = 0
        discounted_returns = []

        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            discounted_returns.insert(0, running_g)

        deltas = torch.tensor(discounted_returns)
        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs, self.rewards = [], []
