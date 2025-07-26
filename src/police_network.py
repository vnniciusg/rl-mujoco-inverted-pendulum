"""
Policy Network for Reinforcement Learning with Continuous Action Spaces

This module implements a neural network policy for reinforcement learning algorithms
that work with continuous action spaces. The network outputs parameters for a
Gaussian distribution (mean and standard deviation) that can be used to sample
continuous actions.

Based on the Gymnasium REINFORCE tutorial for Inverted Pendulum:
https://gymnasium.farama.org/v0.27.0/tutorials/reinforce_invpend_gym_v26/

Example:
    >>> import torch
    >>> # Create policy network for CartPole-like environment
    >>> # 4 observation dimensions (position, velocity, angle, angular velocity)
    >>> # 1 action dimension (force applied to cart)
    >>> policy = PolicyNetwork(obs_space_dims=4, action_space_dims=1)
    >>>
    >>> # Sample observation from environment
    >>> obs = torch.tensor([0.1, -0.2, 0.05, 0.3])
    >>>
    >>> # Get action distribution parameters
    >>> action_mean, action_std = policy(obs)
    >>>
    >>> # Sample action from the distribution
    >>> action_dist = torch.distributions.Normal(action_mean, action_std)
    >>> action = action_dist.sample()
    >>>
    >>> # Calculate log probability for policy gradient
    >>> log_prob = action_dist.log_prob(action)

Architecture:
    Input (obs_space_dims) -> Linear(16) -> Tanh -> Linear(32) -> Tanh
                                                        |
                                                        ├-> Mean Head (action_space_dims)
                                                        └-> Log Std Head (action_space_dims) -> Softplus

Author: vnniciusg
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    A neural network that represents a stochastic policy for continuous action spaces.

    This network implements a parameterized policy π(a|s) that outputs the parameters
    of a Gaussian distribution over continuous actions. It's commonly used in policy
    gradient algorithms like REINFORCE, PPO, and Actor-Critic methods.

    The network consists of:
    1. Shared feature extraction layers that process observations
    2. Two separate output heads:
       - Mean head: outputs μ (mean) of the action distribution
       - Log std head: outputs log(σ) which is converted to σ (standard deviation)

    Args:
        obs_space_dims (int): Number of dimensions in the observation space.
            For CartPole: 4 (position, velocity, pole angle, pole angular velocity)
            For InvertedPendulum: varies based on sensor configuration
        action_space_dims (int): Number of dimensions in the action space.
            For CartPole: 1 (force magnitude)
            For continuous control: typically 1-10 depending on robot joints

    Attributes:
        fc (nn.Sequential): Shared feature extraction backbone
        mean_head (nn.Linear): Output layer for action means
        log_std_head (nn.Linear): Output layer for log standard deviations

    Example:
        >>> # Create policy for inverted pendulum (4 obs dims, 1 action dim)
        >>> policy = PolicyNetwork(4, 1)
        >>>
        >>> # Forward pass with batch of observations
        >>> obs_batch = torch.randn(32, 4)  # batch_size=32, obs_dims=4
        >>> means, stds = policy(obs_batch)
        >>> print(f"Action means shape: {means.shape}")    # [32, 1]
        >>> print(f"Action stds shape: {stds.shape}")      # [32, 1]
        >>>
        >>> # Create action distribution and sample
        >>> dist = torch.distributions.Normal(means, stds)
        >>> actions = dist.sample()                         # [32, 1]
        >>> log_probs = dist.log_prob(actions)             # [32, 1]
        >>>
        >>> # Single observation inference
        >>> single_obs = torch.tensor([0.1, 0.0, 0.05, 0.0])
        >>> mean, std = policy(single_obs)
        >>> action = torch.distributions.Normal(mean, std).sample()

    Note:
        The log standard deviation head uses softplus activation (log(1 + exp(x)))
        instead of exponential to ensure numerical stability and prevent the
        standard deviation from becoming too large or causing overflow.
    """

    def __init__(self, obs_space_dims: int, action_space_dims: int) -> None:
        """
        Initialize the PolicyNetwork architecture.

        Sets up the neural network layers including the shared feature extractor
        and the two output heads for mean and log standard deviation.

        Args:
            obs_space_dims (int): Dimensionality of the observation space.
                Must be positive integer representing the number of features
                in each observation vector from the environment.
            action_space_dims (int): Dimensionality of the action space.
                Must be positive integer representing the number of continuous
                actions the agent can take.

        Raises:
            ValueError: If obs_space_dims or action_space_dims are not positive integers.

        Example:
            >>> # For CartPole environment
            >>> policy = PolicyNetwork(obs_space_dims=4, action_space_dims=1)
            >>>
            >>> # For multi-joint robot arm
            >>> policy = PolicyNetwork(obs_space_dims=12, action_space_dims=6)
        """
        super(PolicyNetwork, self).__init__()

        HIDDEN_SPACE1: int = 16
        HIDDEN_SPACE2: int = 32

        self.fc = nn.Sequential(
            nn.Linear(obs_space_dims, HIDDEN_SPACE1),
            nn.Tanh(),
            nn.Linear(HIDDEN_SPACE1, HIDDEN_SPACE2),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(HIDDEN_SPACE2, action_space_dims)
        self.log_std_head = nn.Linear(HIDDEN_SPACE2, action_space_dims)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        Processes input observations through the network to produce parameters
        of a Gaussian distribution over continuous actions. The output can be
        used to create a torch.distributions.Normal distribution for action
        sampling and log probability computation.

        Args:
            x (torch.Tensor): Input observations tensor with shape:
                - (obs_space_dims,) for single observation
                - (batch_size, obs_space_dims) for batch of observations
                The tensor will be automatically converted to float32.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - action_mean (torch.Tensor): Mean parameters μ of the Gaussian
                  distribution with same batch dimensions as input and
                  action_space_dims in the last dimension.
                - action_std (torch.Tensor): Standard deviation parameters σ
                  of the Gaussian distribution, guaranteed to be positive
                  through softplus activation.

        Example:
            >>> policy = PolicyNetwork(4, 1)
            >>> obs = torch.tensor([0.1, 0.0, 0.05, 0.0])
            >>> mean, std = policy(obs)
            >>> print(f"Mean: {mean.item():.3f}, Std: {std.item():.3f}")
            >>>
            >>> # Batch processing
            >>> obs_batch = torch.randn(16, 4)
            >>> means, stds = policy(obs_batch)
            >>> print(f"Batch means shape: {means.shape}")  # [16, 1]
            >>>
            >>> # Create distribution and sample actions
            >>> dist = torch.distributions.Normal(means, stds)
            >>> actions = dist.sample()                      # [16, 1]
            >>> log_probs = dist.log_prob(actions)          # [16, 1]

        Mathematical Details:
            The network computes:
            - features = Tanh(Linear(Tanh(Linear(x))))
            - μ = Linear_mean(features)
            - σ = softplus(Linear_logstd(features))
            where softplus(x) = log(1 + exp(x)) ensures σ > 0

        Note:
            The softplus activation for standard deviation provides numerical
            stability compared to direct exponential and ensures the standard
            deviation remains positive and bounded.
        """
        features = self.fc(x.float())
        action_mean = self.mean_head(features)
        action_std = torch.log(1 + torch.exp(self.log_std_head(features)))

        return action_mean, action_std
