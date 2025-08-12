import torch
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(state_dim, action_dim, num_samples, batch_size, device):
    """
    Prepares dummy D4RL-like datasets for the experiment.
    In a real D4RL setup, you'd load data using d4rl.qlearning_dataset(env).
    """
    print("--- Data Preprocessing: Generating Dummy Dataset ---")
    
    # Generate dummy data matching D4RL format
    dummy_states = torch.randn(num_samples, state_dim)
    dummy_actions = torch.randn(num_samples, action_dim)
    dummy_rewards = torch.randn(num_samples, 1)
    dummy_next_states = torch.randn(num_samples, state_dim)
    dummy_terminals = torch.randint(0, 2, (num_samples, 1)).float()

    # Dataloader for EBM training (s,a pairs)
    dataset_D_sa = TensorDataset(dummy_states, dummy_actions)
    dataloader_D_sa = DataLoader(dataset_D_sa, batch_size=batch_size, shuffle=True)

    # Dataloader for RL training (s,a,r,s',done tuples)
    dataset_offline = TensorDataset(dummy_states, dummy_actions, dummy_rewards, dummy_next_states, dummy_terminals)
    dataloader_offline = DataLoader(dataset_offline, batch_size=batch_size, shuffle=True)

    print(f"Dummy dataset prepared: {num_samples} samples. State Dim: {state_dim}, Action Dim: {action_dim}")
    return dataloader_D_sa, dataloader_offline
