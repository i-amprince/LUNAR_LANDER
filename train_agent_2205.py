import gymnasium as gym
import numpy as np
import argparse
import os
import torch
from stable_baselines3 import PPO

# Optimized PSO parameters
NUM_PARTICLES = 100
NUM_GENERATIONS = 20 
INERTIA_WEIGHT_START = 0.9
INERTIA_WEIGHT_END = 0.2  
COGNITIVE_WEIGHT = 1.6  
SOCIAL_WEIGHT = 2.0  
VELOCITY_CLAMP = 0.2  
LOWER_BOUND, UPPER_BOUND = -5, 5
MUTATION_RATE = 0.4  
BEST_POLICY_PATH = "best_policy_2205.npy"
NEW_POLICY_PATH = "best_policy_2205_2.npy"

# Initialize PPO model
env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
state_dict = model.policy.state_dict()
param_shapes = [param.shape for param in state_dict.values()]
EXPECTED_POLICY_SIZE = sum(p.numel() for p in state_dict.values())

def flatten_params(state_dict):
    return np.concatenate([p.cpu().numpy().flatten() for p in state_dict.values()])

def unflatten_params(flat_params):
    reconstructed_state_dict = {}
    param_slices = np.cumsum([0] + [p.numel() for p in state_dict.values()])
    
    for i, (name, param) in enumerate(state_dict.items()):
        start, end = param_slices[i], param_slices[i + 1]
        reconstructed_param = flat_params[start:end].reshape(param_shapes[i])
        reconstructed_state_dict[name] = torch.tensor(reconstructed_param)
    
    return reconstructed_state_dict

def policy_action(params, observation):
    reconstructed_state_dict = unflatten_params(params)
    model.policy.load_state_dict(reconstructed_state_dict)
    
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(model.device)
    action, _ = model.predict(obs_tensor, deterministic=True)

    return int(action.item())  # Extract the scalar value properly

def evaluate_policy(params, episodes=10):
    total_reward = 0.0
    for _ in range(episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action = policy_action(params, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
    return total_reward / episodes

def particle_swarm_optimization():
    dim = EXPECTED_POLICY_SIZE

    if os.path.exists(BEST_POLICY_PATH):
        print("Loading previous best policy...")
        global_best_position = np.load(BEST_POLICY_PATH)
    else:
        print("No previous best policy found. Training from scratch...")
        global_best_position = flatten_params(state_dict)

    global_best_score = evaluate_policy(global_best_position)
    particles = np.random.uniform(LOWER_BOUND, UPPER_BOUND, (NUM_PARTICLES, dim))
    velocities = np.random.uniform(-VELOCITY_CLAMP, VELOCITY_CLAMP, (NUM_PARTICLES, dim))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([evaluate_policy(p) for p in particles])

    for i in range(NUM_PARTICLES):
        if personal_best_scores[i] > global_best_score:
            global_best_score = personal_best_scores[i]
            global_best_position = personal_best_positions[i].copy()

    for generation in range(NUM_GENERATIONS):
        inertia_weight = INERTIA_WEIGHT_START - (INERTIA_WEIGHT_START - INERTIA_WEIGHT_END) * (generation / NUM_GENERATIONS)
        
        for i in range(NUM_PARTICLES):
            velocities[i] = (
                inertia_weight * velocities[i]
                + COGNITIVE_WEIGHT * np.random.rand() * (personal_best_positions[i] - particles[i])
                + SOCIAL_WEIGHT * np.random.rand() * (global_best_position - particles[i])
            )
            velocities[i] = np.clip(velocities[i], -VELOCITY_CLAMP, VELOCITY_CLAMP)
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], LOWER_BOUND, UPPER_BOUND)
            
            if np.random.rand() < MUTATION_RATE:
                particles[i] += np.random.uniform(-0.5, 0.5, size=dim)
                particles[i] = np.clip(particles[i], LOWER_BOUND, UPPER_BOUND)
            
            score = evaluate_policy(particles[i])
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i].copy()
                
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()

        print(f"Generation {generation+1}: Best Reward = {global_best_score:.2f}")
    
    return global_best_position

def train_and_save():
    best_params = particle_swarm_optimization()
    np.save(NEW_POLICY_PATH, best_params)
    print(f"Best policy saved to {NEW_POLICY_PATH}")

def load_policy(filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    best_params = np.load(filename)
    if best_params.shape[0] != EXPECTED_POLICY_SIZE:
        print("Error: Loaded policy has incorrect shape. Cannot use.")
        return None
    print(f"Loaded best policy from {filename}")
    return best_params

def play_policy(best_params, episodes=5):
    avg_reward = evaluate_policy(best_params, episodes=episodes)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train policy using PSO")
    parser.add_argument("--play", action="store_true", help="Load and play with best policy")
    args = parser.parse_args()
    
    if args.train:
        train_and_save()
    elif args.play:
        best_params = load_policy(NEW_POLICY_PATH)
        if best_params is not None:
            play_policy(best_params, episodes=5)
    else:
        print("Specify --train to train or --play to test a saved policy.")
