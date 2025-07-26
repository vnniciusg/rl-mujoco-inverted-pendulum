"""
Environment setup utilities for reinforcement learning on InvertedPendulum.

This module provides functionality to create and configure InvertedPendulum environments
with appropriate wrappers for training and evaluation of reinforcement learning agents.
The environments are configured with video recording and episode statistics tracking
capabilities for the MuJoCo InvertedPendulum control task.

Functions:
    setup_environment: Creates a configured InvertedPendulum environment with wrappers.

Dependencies:
    - gymnasium: For the base InvertedPendulum-v4 environment
    - gymnasium.wrappers: For recording and statistics wrappers

Example:
    >>> from src.environment import setup_environment
    >>> env = setup_environment()
    >>> observation, info = env.reset()
    >>> action = env.action_space.sample()
    >>> observation, reward, terminated, truncated, info = env.step(action)

Author: vnniciusg
"""

from typing import Literal

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


def setup_environment(
    env_name: str = "InvertedPendulum-v5",
    num_eval_episodes: int = 5000,
    render_mode: Literal["rgb_array", "human", "ansi"] = "rgb_array",
    training_period: int = 500,
    video_folder: str = "inverted-pendulum",
    name_prefix: str = "eval",
):
    """
    Create and configure an InvertedPendulum environment with recording capabilities.

    This function sets up an InvertedPendulum-v4 environment with the following features:
    - MuJoCo-based continuous control simulation
    - Video recording of episodes at specified intervals
    - Episode statistics tracking for performance monitoring
    - Configurable rendering modes for different use cases

    The InvertedPendulum task involves balancing a pendulum on a cart by applying
    horizontal forces. The observation space includes cart position, velocity,
    pendulum angle, and angular velocity. The action space is continuous, representing
    the force applied to the cart.

    Args:
        env_name (str, optional): Name of the Gymnasium environment to create.
            Defaults to "InvertedPendulum-v4".
        num_eval_episodes (int, optional): Buffer length for episode statistics.
            Defaults to 5000.
        render_mode (Literal["rgb_array", "human", "ansi"], optional):
            Rendering mode for the environment:
            - "rgb_array": Returns RGB array for video recording
            - "human": Renders to screen for human observation
            - "ansi": Text-based rendering for terminal output
            Defaults to "rgb_array".
        training_period (int, optional): Interval between recorded episodes.
            Videos are recorded every `training_period` episodes. Defaults to 1000.
        video_folder (str, optional): Directory path where recorded videos
            will be saved. Defaults to "inverted-pendulum".
        name_prefix (str, optional): Prefix for video filenames. The recorded videos
            will be named as "{name_prefix}-episode-{episode_number}.mp4".
            Defaults to "eval".

    Returns:
        gymnasium.Env: A wrapped InvertedPendulum environment with the following wrappers:
            - RecordVideo: Records videos at specified intervals
            - RecordEpisodeStatistics: Tracks episode rewards, lengths, and times

    Note:
        The InvertedPendulum-v4 environment uses MuJoCo physics simulation and requires
        a valid MuJoCo installation. The environment has a continuous action space
        with values typically in the range [-3.0, 3.0] representing forces applied
        to the cart.

    Example:
        >>> env = setup_environment(
        ...     render_mode="human",
        ...     training_period=100,
        ...     video_folder="my_videos"
        ... )
        >>> obs, info = env.reset()
        >>> # Environment is ready for training/evaluation with REINFORCE
    """

    _env = gym.make(env_name, render_mode=render_mode)

    _env = RecordVideo(
        _env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: x % training_period == 0,
    )

    return RecordEpisodeStatistics(env=_env, buffer_length=num_eval_episodes)
