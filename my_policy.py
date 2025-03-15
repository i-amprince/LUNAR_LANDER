import numpy as np
import torch
from stable_baselines3 import PPO
import gymnasium as gym

POLICY_PATH = "best_policy.npy"

# Load flattened policy
flattened_params = np.load(POLICY_PATH)

# Reconstruct state_dict
env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
state_dict = model.policy.state_dict()
param_shapes = [param.shape for param in state_dict.values()]
param_sizes = [param.numel() for param in state_dict.values()]  # Corrected line
param_slices = np.cumsum([0] + param_sizes)

reconstructed_state_dict = {}
for i, (name, param) in enumerate(state_dict.items()):
    start, end = param_slices[i], param_slices[i + 1]
    reconstructed_param = flattened_params[start:end].reshape(param_shapes[i])
    reconstructed_state_dict[name] = torch.tensor(reconstructed_param)

model.policy.load_state_dict(reconstructed_state_dict)

def policy_action(policy, observation):
    """Use the loaded PPO model to select an action."""
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(model.device)
    action, _ = model.predict(obs_tensor, deterministic=True)
    return int(action)