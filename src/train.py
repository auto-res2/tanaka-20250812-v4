import torch
import torch.nn as nn
import torch.optim as optim
import itertools

# --- Network Definitions ---
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

class QFunction(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, phi_sa):
        return self.linear(phi_sa)

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)
        )

    def forward(self, s):
        mu_log_std = self.net(s)
        mu, log_std = mu_log_std.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist

# --- Helper Functions (for EBM/Diffusion, used in training) ---

def sample_xt(x0, t, alpha_bars_tensor):
    sqrt_alpha_bar = torch.sqrt(alpha_bars_tensor[t-1]).reshape(-1, 1).to(x0.device)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bars_tensor[t-1]).reshape(-1, 1).to(x0.device)
    epsilon = torch.randn_like(x0)
    xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
    return xt, epsilon

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

# --- Training Phases ---

def train_ebm_phase(score_net, dataloader_D_sa, config, device, alpha_bars):
    num_epochs = config['training']['ebm_epochs']
    T = config['ebm']['T']
    lr_ebm = config['training']['lr_ebm']
    
    print(f"\n--- Phase 1: EBM/Score Network Pre-training (Epochs: {num_epochs}) ---")
    optimizer_ebm = optim.Adam(score_net.parameters(), lr=lr_ebm)
    mse_loss = nn.MSELoss()

    ebm_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (s_batch, a_batch) in enumerate(dataloader_D_sa):
            s_batch, a_batch = s_batch.to(device), a_batch.to(device)
            x0_batch = torch.cat([s_batch, a_batch], dim=-1)
            
            t_batch = torch.randint(1, T + 1, (x0_batch.shape[0],), device=device)

            xt_batch, epsilon_batch = sample_xt(x0_batch, t_batch, alpha_bars)
            predicted_epsilon, _ = score_net(xt_batch, t_batch)

            loss_ebm = mse_loss(predicted_epsilon, epsilon_batch)

            optimizer_ebm.zero_grad()
            loss_ebm.backward()
            optimizer_ebm.step()
            epoch_loss += loss_ebm.item()

        avg_epoch_loss = epoch_loss / len(dataloader_D_sa)
        ebm_losses.append(avg_epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"EBM Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")

    print("EBM/Score network pre-training complete.")
    for param in score_net.parameters():
        param.requires_grad = False
    print("ScoreNetwork parameters frozen for RL phase.")
    return ebm_losses

def train_rl_phase(score_net_fixed, state_dim, action_dim, dataloader_offline, config, device):
    feature_dim = config['model']['feature_dim']
    hidden_dim = config['model']['hidden_dim']
    num_rl_steps = config['training']['rl_gradient_steps']
    lr_q = config['training']['lr_q']
    lr_policy = config['training']['lr_policy']
    gamma = config['training']['gamma']
    tau = config['training']['tau']
    lambda_ebm = config['training']['lambda_ebm']

    print(f"\n--- Phase 2: Offline RL Training (Steps: {num_rl_steps}) ---")

    Q1 = QFunction(feature_dim).to(device)
    Q2 = QFunction(feature_dim).to(device)
    Q1_target = QFunction(feature_dim).to(device)
    Q2_target = QFunction(feature_dim).to(device)
    Q1_target.load_state_dict(Q1.state_dict())
    Q2_target.load_state_dict(Q2.state_dict())

    policy_net = Policy(state_dim, action_dim, hidden_dim).to(device)
    policy_target_net = Policy(state_dim, action_dim, hidden_dim).to(device)
    policy_target_net.load_state_dict(policy_net.state_dict())

    optimizer_q = optim.Adam(list(Q1.parameters()) + list(Q2.parameters()), lr=lr_q)
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr_policy)

    mse_loss = nn.MSELoss()

    q_losses = []
    policy_losses = []
    ood_energy_log = []

    data_iterator = itertools.cycle(dataloader_offline)

    for step in range(1, num_rl_steps + 1):
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = next(data_iterator)
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
            s_batch.to(device), a_batch.to(device), r_batch.to(device), \
            s_prime_batch.to(device), done_batch.to(device)

        # --- Update Q-functions ---
        with torch.no_grad():
            policy_dist_s_prime = policy_target_net(s_prime_batch)
            a_prime = policy_dist_s_prime.sample() 
            a_prime = a_prime.clamp(-1.0, 1.0)
            
            # Using t=1 for low noise features, as suggested in original code comment
            phi_s_prime_a_prime = get_spectral_feature(s_prime_batch, a_prime, score_net_fixed, t_idx=1)
            q1_target = Q1_target(phi_s_prime_a_prime)
            q2_target = Q2_target(phi_s_prime_a_prime)
            q_target = torch.min(q1_target, q2_target)
            y_batch = r_batch + (1 - done_batch) * gamma * q_target
        
        # Using t=1 for low noise features for current state-action pair
        phi_s_a = get_spectral_feature(s_batch, a_batch, score_net_fixed, t_idx=1)
        q1_pred = Q1(phi_s_a)
        q2_pred = Q2(phi_s_a)
        
        # EBM regularization for Q-loss
        ebm_energy_s_a = get_ebm_energy(s_batch, a_batch, score_net_fixed, t_idx=1)
        q_loss = mse_loss(q1_pred, y_batch) + mse_loss(q2_pred, y_batch) + lambda_ebm * ebm_energy_s_a.mean()

        optimizer_q.zero_grad()
        q_loss.backward()
        optimizer_q.step()
        q_losses.append(q_loss.item())

        # --- Update Policy ---
        if step % 2 == 0:
            policy_dist_s = policy_net(s_batch)
            a_from_policy = policy_dist_s.sample()
            a_from_policy = a_from_policy.clamp(-1.0, 1.0)
            
            # Using t=1 for low noise features for policy actions
            phi_s_a_policy = get_spectral_feature(s_batch, a_from_policy, score_net_fixed, t_idx=1)
            q_val_policy = Q1(phi_s_a_policy)

            # EBM regularization for policy loss
            ebm_energy_s_a_policy = get_ebm_energy(s_batch, a_from_policy, score_net_fixed, t_idx=1)
            policy_loss = (-q_val_policy + lambda_ebm * ebm_energy_s_a_policy).mean()

            optimizer_policy.zero_grad()
            policy_loss.backward()
            optimizer_policy.step()
            policy_losses.append(policy_loss.item())
            ood_energy_log.append(ebm_energy_s_a_policy.mean().item())

            # --- Update target networks (Polyak averaging) ---
            for param, target_param in zip(Q1.parameters(), Q1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(Q2.parameters(), Q2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(policy_net.parameters(), policy_target_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        if step % (num_rl_steps // 10) == 0 or step == 1:
            print(f"RL Step {step}/{num_rl_steps}: Q Loss = {q_losses[-1]:.4f}, Policy Loss = {policy_losses[-1] if policy_losses else 'N/A':.4f}, Policy OOD Energy = {ood_energy_log[-1] if ood_energy_log else 'N/A':.4f}")
    
    print("Offline RL training complete.")
    return q_losses, policy_losses, ood_energy_log, policy_net
