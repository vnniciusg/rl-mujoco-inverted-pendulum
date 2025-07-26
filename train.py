"""
Training Script for REINFORCE Algorithm on InvertedPendulum-v4

This module implements the training pipeline for the REINFORCE policy gradient algorithm
on the InvertedPendulum-v4 environment. The script trains multiple agents with different
random seeds and generates learning curves to visualize performance.

The training process:
1. Creates multiple REINFORCE agents with different random seeds
2. Trains each agent for a specified number of episodes
3. Collects episode rewards and tracks learning progress
4. Generates and saves learning curve plots

Based on the Gymnasium REINFORCE tutorial for Inverted Pendulum:
https://gymnasium.farama.org/v0.27.0/tutorials/reinforce_invpend_gym_v26/

Functions:
    plot_learning_curve: Creates and saves learning curve visualization
    train: Main training loop for REINFORCE agents

Author: vnniciusg
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from tqdm import tqdm

from src.environment import setup_environment
from src.reinforce import REINFORCE

warnings.filterwarnings("ignore")

NUM_EPISODES: int = 5000


def plot_learning_curve(
    rewards_over_seeds: list, filename: str = "reinforce_learning_curve.png"
) -> None:
    """
    Create and save a learning curve plot showing REINFORCE training progress.

    This function takes reward data from multiple training runs (different seeds)
    and creates a line plot showing the learning progress over episodes. The plot
    is saved as a PNG file instead of being displayed.

    Args:
        rewards_over_seeds (list): List of reward lists, where each inner list
            contains episode rewards for one training run (seed). Each reward
            should be a tuple where the first element is the actual reward value.
        filename (str, optional): Name of the file to save the plot.
            Defaults to "reinforce_learning_curve.png".

    Returns:
        None: The function saves the plot to disk and doesn't return anything.

    Note:
        The plot uses seaborn styling with a dark grid theme and rainbow palette
        for better visualization of multiple training runs.
    """
    rewards_to_plot = [
        [reward[0] for reward in rewards] for rewards in rewards_over_seeds
    ]
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title="REINFORCE for InvertedPendulum-v4"
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    logger.success(f"Learning curve saved as {filename}")


def train() -> None:
    """
    Main training function for REINFORCE algorithm on InvertedPendulum-v4.

    This function implements the complete training pipeline:
    1. Sets up the InvertedPendulum environment with recording capabilities
    2. Trains multiple REINFORCE agents with different random seeds for robustness
    3. Collects episode rewards and performance statistics
    4. Generates learning curves showing training progress

    The training uses multiple seeds to ensure statistical significance and
    account for the stochastic nature of policy gradient methods. Each agent
    is trained for NUM_EPISODES episodes, with progress logged every 1000 episodes.

    Training Process:
    - For each seed, creates a new REINFORCE agent
    - Runs episodes until termination or truncation
    - Collects rewards and updates policy after each episode
    - Tracks average returns using environment's return queue
    - Saves learning curve plot after training completion

    Returns:
        None: The function trains agents and saves results to disk.

    Note:
        The environment is reset with seed=42 for consistency, while agent
        initialization uses different seeds for diversity in training runs.
    """
    rewards_over_seeds = []

    for seed in [1, 2, 3, 5, 8]:
        logger.info(f"Starting training with seed {seed}")

        env = setup_environment(
            num_eval_episodes=NUM_EPISODES,
            video_folder=f"inverted-pendulum-seed-{seed}",
            name_prefix=f"seed-{seed}-eval",
        )

        agent = REINFORCE(env.observation_space.shape[0], env.action_space.shape[0])
        rewards_over_eps = []

        for episode in tqdm(range(NUM_EPISODES), desc=f"Seed {seed}"):
            obs, _ = env.reset(seed=42)
            done: bool = False

            while not done:
                action = agent.sample_action(obs)

                obs, reward, terminated, truncated, _ = env.step(action=action)
                agent.rewards.append(reward)

                done = terminated or truncated

            rewards_over_eps.append(env.return_queue[-1])
            agent.update()

            if episode % 1000 == 0:
                logger.info(
                    f"EPISODE: {episode}, AVG_REWARD: {int(np.mean(env.return_queue))}"
                )

        rewards_over_seeds.append(rewards_over_eps)

    logger.success("Training finished succesfully")

    plot_learning_curve(rewards_over_seeds=rewards_over_seeds)


if __name__ == "__main__":
    train()
