import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Network Definitions (Duplicated for self-contained script) ---
class ScoreNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_timesteps, time_embedding_dim, feature_dim):
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.time_embedding_dim = time_embedding_dim
        self.feature_dim = feature_dim

        self.time_embedding = nn.Embedding(max_timesteps + 1, time_embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim + time_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score_output_layer = nn.Linear(hidden_dim, self.input_dim)
        self.feature_extractor_head = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        h = self.mlp(torch.cat([x, t_emb], dim=-1))
        score_output = self.score_output_layer(h)
        spectral_feature = self.feature_extractor_head(h)
        return score_output, spectral_feature

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2) # For mean and log_std
        )

    def forward(self, s):
        mu_log_std = self.net(s)
        mu, log_std = mu_log_std.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist

# --- Helper Functions (Duplicated for self-contained script) ---

def get_spectral_feature(s, a, score_net_fixed, t_idx):
    sa_combined = torch.cat([s, a], dim=-1).float()
    t = torch.full((sa_combined.shape[0],), t_idx, dtype=torch.long, device=sa_combined.device)
    with torch.no_grad():
        _, spectral_feature = score_net_fixed(sa_combined, t)
    return spectral_feature

def get_ebm_energy(s, a, score_net_fixed, t_idx):
    sa_combined = torch.cat([s, a], dim=-1).float()
    t = torch.full((sa_combined.shape[0],), t_idx, dtype=torch.long, device=sa_combined.device)
    with torch.no_grad():
        score_output, _ = score_net_fixed(sa_combined, t)
    return torch.sum(score_output**2, dim=-1)


# --- Evaluation Function ---
def evaluate_policy(policy_net, env_name, score_net_fixed, config, device):
    eval_episodes = config['evaluation']['eval_episodes']
    state_dim = config['experiment']['state_dim']
    action_dim = config['experiment']['action_dim']
    
    print(f"\n--- Evaluating Policy on {env_name} for {eval_episodes} episodes ---")
    
    episode_returns = []
    ood_energy_per_episode = []
    
    # In a real D4RL setup, you would use gymnasium and d4rl:
    # import gymnasium as gym
    # import d4rl
    # env = gym.make(env_name)
    # obs = env.reset()
    # for i in range(eval_episodes):
    #     current_return = 0
    #     done = False
    #     while not done:
    #         with torch.no_grad():
    #             s_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    #             action_dist = policy_net(s_tensor)
    #             action = action_dist.sample().cpu().numpy().squeeze(0)
    #             action = np.clip(action, -1.0, 1.0) # Ensure actions are within valid range

    #             # Calculate OOD energy for the action taken by policy
    #             a_tensor = torch.from_numpy(action).float().unsqueeze(0).to(device)
    #             ebm_energy = get_ebm_energy(s_tensor, a_tensor, score_net_fixed, t_idx=1).item()
    #             ood_energy_per_episode.append(ebm_energy)
                
    #         next_obs, reward, terminated, truncated, _ = env.step(action)
    #         done = terminated or truncated
    #         current_return += reward
    #         obs = next_obs
    #     episode_returns.append(current_return)
    #     obs = env.reset() # Reset for next episode

    # For this conceptual example, we will return dummy normalized scores.
    for i in range(eval_episodes):
        current_return = np.random.uniform(50, 200) # Example raw return
        current_ood_energy = np.random.uniform(0.1, 5.0) # Example OOD energy
        episode_returns.append(current_return)
        ood_energy_per_episode.append(current_ood_energy)

    avg_raw_return = np.mean(episode_returns)
    avg_ood_energy = np.mean(ood_energy_per_episode)
    
    # Dummy normalized score calculation (replace with actual D4RL normalization)
    # For testing, just map avg_raw_return to a [0, 100] scale, simulating normalization
    normalized_score = (avg_raw_return - 50) / 150 * 100 # Maps 50-200 to 0-100
    normalized_score = np.clip(normalized_score, 0, 100)

    print(f"Average Raw Return: {avg_raw_return:.2f}")
    print(f"Average Policy OOD Energy (during evaluation): {avg_ood_energy:.4f}")
    print(f"Normalized Score (D4RL Standard - Dummy): {normalized_score:.2f}")
    print("Evaluation complete.")
    return normalized_score, avg_raw_return, avg_ood_energy
