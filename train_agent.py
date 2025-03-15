import gymnasium as gym
import numpy as np
import argparse
import torch
import os
from stable_baselines3 import PPO

POLICY_PATH = "best_policy.npy"

def train_ppo(env_id="LunarLander-v3", total_timesteps=100000, learning_rate=2.5e-4, gamma=0.999, ent_coef=0.02, clip_range=0.1, batch_size=64, n_steps=2048):
    """Train or resume training PPO model and save policy as a flat NumPy array."""
    env = gym.make(env_id)
    device = "cpu"

    if os.path.exists(POLICY_PATH):
        print("Resuming training from existing policy...")
        # Load existing flattened policy
        flattened_params = np.load(POLICY_PATH)
        # Reconstruct state_dict
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            batch_size=batch_size,
            n_steps=n_steps,
        )
        state_dict = model.policy.state_dict()
        param_shapes = [param.shape for param in state_dict.values()]
        param_sizes = [param.numel() for param in state_dict.values()]
        param_slices = np.cumsum([0] + param_sizes)

        reconstructed_state_dict = {}
        for i, (name, param) in enumerate(state_dict.items()):
            start, end = param_slices[i], param_slices[i + 1]
            reconstructed_param = flattened_params[start:end].reshape(param_shapes[i])
            reconstructed_state_dict[name] = torch.tensor(reconstructed_param)

        model.policy.load_state_dict(reconstructed_state_dict)
    else:
        print("Training PPO from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            clip_range=clip_range,
        )

    model.learn(total_timesteps=total_timesteps, progress_bar=True, reset_num_timesteps = False)

    # Flatten the state_dict into a single NumPy array
    state_dict = model.policy.state_dict()
    flattened_params = np.concatenate([param.cpu().numpy().flatten() for param in state_dict.values()])
    np.save(POLICY_PATH, flattened_params)
    print(f"Flattened policy saved at {POLICY_PATH}")

def play_ppo(env_id="LunarLander-v3", episodes=5):
    """Load trained PPO policy and play."""
    if not os.path.exists(POLICY_PATH):
        print("No trained policy found! Train it first with --train.")
        return

    env = gym.make(env_id, render_mode="human")

    # Load flattened policy
    flattened_params = np.load(POLICY_PATH)

    # Reconstruct state_dict
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    state_dict = model.policy.state_dict()
    param_shapes = [param.shape for param in state_dict.values()]
    param_sizes = [param.numel() for param in state_dict.values()]
    param_slices = np.cumsum([0] + param_sizes)

    reconstructed_state_dict = {}
    for i, (name, param) in enumerate(state_dict.items()):
        start, end = param_slices[i], param_slices[i + 1]
        reconstructed_param = flattened_params[start:end].reshape(param_shapes[i])
        reconstructed_state_dict[name] = torch.tensor(reconstructed_param)

    model.policy.load_state_dict(reconstructed_state_dict)

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play PPO agent for Lunar Lander.")
    parser.add_argument("--train", action="store_true", help="Train PPO and save the model.")
    parser.add_argument("--play", action="store_true", help="Load and play the trained PPO model.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.999, help="Gamma value.")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="Entropy coefficient.")
    parser.add_argument("--clip_range", type=float, default=0.1, help="Clip range.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps.")

    args = parser.parse_args()

    if args.train:
        train_ppo(
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
        )
    elif args.play:
        play_ppo(episodes=args.episodes)
    else:
        print("Specify --train to train or --play to test a trained model.")