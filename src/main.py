import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# Import components from other scripts
from src.preprocess import prepare_data
from src.train import ScoreNetwork, QFunction, Policy, train_ebm_phase, train_rl_phase
from src.evaluate import evaluate_policy

# --- Global Configuration Loader ---
def load_config_from_string(config_string):
    """Loads YAML configuration from a string."""
    return yaml.safe_load(config_string)

# --- Plotting Function ---
def plot_results(ebm_losses, q_losses, policy_losses, ood_energy_log, eval_scores, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"\n--- Generating Plots in {plot_dir} ---")

    # Plot 1: EBM Training Loss
    if ebm_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(ebm_losses) + 1), ebm_losses, label='EBM Loss')
        plt.xlabel('EBM Epochs')
        plt.ylabel('Loss')
        plt.title('EBM/Score Network Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "training_loss_ebm.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved training_loss_ebm.pdf")

    # Plot 2: Q-function Training Loss
    if q_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(q_losses) + 1), q_losses, label='Q-function Loss', color='orange')
        plt.xlabel('RL Gradient Steps')
        plt.ylabel('Loss')
        plt.title('Q-function Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "training_loss_q.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved training_loss_q.pdf")

    # Plot 3: Policy Training Loss
    if policy_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(policy_losses) + 1), policy_losses, label='Policy Loss', color='green')
        plt.xlabel('RL Gradient Steps (Policy Updates)')
        plt.ylabel('Loss')
        plt.title('Policy Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "training_loss_policy.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved training_loss_policy.pdf")

    # Plot 4: OOD Energy of Policy Actions
    if ood_energy_log:
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(ood_energy_log) + 1), ood_energy_log, label='Policy OOD Energy', color='red')
        plt.xlabel('RL Gradient Steps (Policy Updates)')
        plt.ylabel('EBM Energy Proxy')
        plt.title('Average EBM Energy of Policy Actions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "policy_ood_energy.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved policy_ood_energy.pdf")

    # Plot 5: Evaluation Scores
    if eval_scores:
        labels = [score[0] for score in eval_scores]
        scores = [score[1] for score in eval_scores]
        # raw_returns = [score[2] for score in eval_scores] # Not directly plotted
        eval_ood_energies = [score[3] for score in eval_scores]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.barplot(x=labels, y=scores, ax=axes[0], palette='viridis')
        axes[0].set_ylabel('Normalized Score (D4RL Standard)')
        axes[0].set_title('Normalized Policy Performance')
        axes[0].tick_params(axis='x', rotation=15)
        for i, v in enumerate(scores):
            axes[0].text(i, v + 1, f"{v:.1f}", color='black', ha='center')
        axes[0].set_ylim(0, 100)

        sns.barplot(x=labels, y=eval_ood_energies, ax=axes[1], palette='magma')
        axes[1].set_ylabel('Average EBM Energy')
        axes[1].set_title('Policy OOD Energy During Evaluation')
        axes[1].tick_params(axis='x', rotation=15)
        for i, v in enumerate(eval_ood_energies):
            axes[1].text(i, v + 0.1, f"{v:.2f}", color='black', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "evaluation_metrics.pdf"), bbox_inches="tight")
        plt.close()
        print("Saved evaluation_metrics.pdf")

    print("All plots generated.")

# --- Main Experiment Orchestration Function ---
def run_experiment(config):
    env_name = config['experiment']['env_name']
    dataset_config_name = config['experiment']['dataset_config']
    run_id = config['experiment']['run_id']
    state_dim = config['experiment']['state_dim']
    action_dim = config['experiment']['action_dim']
    num_samples = config['experiment']['num_samples']
    batch_size = config['training']['batch_size']
    plot_dir_base = config['paths']['plot_dir']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize diffusion parameters
    T = config['ebm']['T']
    beta_start = config['ebm']['beta_start']
    beta_end = config['ebm']['beta_end']
    betas = torch.linspace(beta_start, beta_end, T).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0).to(device)

    print(f"\n{'='*50}\nStarting Experiment: {env_name} ({dataset_config_name}) - Run ID: {run_id}\n{'='*50}")

    # --- 1. Prepare Data ---
    dataloader_D_sa, dataloader_offline = prepare_data(state_dim, action_dim, num_samples, batch_size, device)

    # --- 2. Initialize Models ---
    score_net = ScoreNetwork(
        state_dim, action_dim, 
        config['model']['hidden_dim'], 
        config['ebm']['T'], 
        config['model']['time_embedding_dim'], 
        config['model']['feature_dim']
    ).to(device)
    print("Models initialized.")

    # --- 3. Phase 1: EBM/Representation Learning ---
    ebm_losses = train_ebm_phase(score_net, dataloader_D_sa, config, device, alpha_bars)

    # --- 4. Phase 2: Offline RL (Q-function and Policy Learning) ---
    q_losses, policy_losses, ood_energy_log, policy_net = train_rl_phase(
        score_net, state_dim, action_dim, dataloader_offline, config, device
    )

    # --- 5. Evaluation ---
    normalized_score, raw_return, avg_ood_energy = evaluate_policy(
        policy_net, f"{env_name}-{dataset_config_name}", score_net, config, device
    )
    eval_scores = [(f"{env_name}-{dataset_config_name}", normalized_score, raw_return, avg_ood_energy)]

    # --- 6. Plot Results ---
    plot_dir_run = os.path.join(plot_dir_base, f"{env_name}_{dataset_config_name}_{run_id}")
    plot_results(ebm_losses, q_losses, policy_losses, ood_energy_log, eval_scores, plot_dir_run)

    print(f"Experiment {env_name} ({dataset_config_name}) - Run ID: {run_id} finished.")
    return {
        "experiment_code": f"{env_name}-{dataset_config_name}",
        "normalized_score": normalized_score,
        "avg_ood_energy": avg_ood_energy
    }


# --- Test Function (modified for direct config string) ---
def test_experiment_code():
    print("\n========================================")
    print("Running quick test of experiment code...")
    print("========================================")
    
    # Hardcode a test config string
    test_config_string = """
experiment:
  run_id: "test_run"
  env_name: "halfcheetah"
  dataset_config: "medium-v2"
  state_dim: 17
  action_dim: 6
  num_samples: 1000
  eval_episodes: 2
model:
  hidden_dim: 32 # Reduced for speed
  time_embedding_dim: 16 # Reduced for speed
  feature_dim: 32 # Reduced for speed
ebm:
  T: 100 # Reduced for speed
  beta_start: 0.0001
  beta_end: 0.02
training:
  batch_size: 64 # Reduced for speed
  ebm_epochs: 2 # Very few epochs for test
  rl_gradient_steps: 200 # Very few steps for test
  lr_ebm: 0.001 # Adjusted for smaller scale
  lr_q: 0.001
  lr_policy: 0.001
  gamma: 0.99
  tau: 0.005
  lambda_ebm: 0.1 # Adjusted for smaller scale
paths:
  plot_dir: "./.research/iteration1/images"
"""
    test_config = load_config_from_string(test_config_string)
    
    try:
        run_experiment(test_config)
        print("\nTest successful: Code ran without critical errors and produced plots.")
    except Exception as e:
        print(f"\nTest failed with an error: {e}")
        import traceback
        traceback.print_exc()
    
    print("========================================")
    print("Test complete.")
    print("========================================\n")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure plot directory exists for the test run
    os.makedirs("./.research/iteration1/images", exist_ok=True)
    test_experiment_code()
