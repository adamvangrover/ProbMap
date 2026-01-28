import sys
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation.oswm import OneShotWorldModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_regime_data(num_samples=100, seq_len=20, input_dim=5, regime_type='normal'):
    """
    Generates synthetic sequences representing different credit regimes.
    """
    data = []
    targets = []

    for _ in range(num_samples):
        # Base trend
        t = np.linspace(0, 10, seq_len + 1)

        if regime_type == 'normal':
            # Slow, steady improvement or stability
            noise = np.random.normal(0, 0.1, (seq_len + 1, input_dim))
            trend = np.repeat(np.sin(t)[:, np.newaxis] * 0.5, input_dim, axis=1)
        elif regime_type == 'crisis':
            # Volatile, downward trends
            noise = np.random.normal(0, 0.5, (seq_len + 1, input_dim))
            trend = np.repeat(-np.exp(t * 0.2)[:, np.newaxis], input_dim, axis=1)
        else:
            # Random walk
            noise = np.random.normal(0, 0.2, (seq_len + 1, input_dim))
            trend = np.cumsum(np.random.normal(0, 0.1, (seq_len + 1, input_dim)), axis=0)

        seq = trend + noise

        # Normalize roughly to [-1, 1] or similar scale
        seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-6)

        data.append(seq[:-1]) # Input: t=0 to T-1
        targets.append(seq[1:]) # Target: t=1 to T

    return torch.FloatTensor(np.array(data)), torch.FloatTensor(np.array(targets))

def main():
    logger.info("--- Starting OSWM Demo ---")

    input_dim = 5
    seq_len = 20

    # Initialize Model
    oswm = OneShotWorldModel(input_dim=input_dim, hidden_dim=32, num_layers=2, nhead=2)
    logger.info("One-Shot World Model initialized.")

    # 1. Meta-Training Phase
    logger.info("Generating synthetic training data (Normal & Crisis regimes)...")
    train_data_normal, train_targets_normal = generate_synthetic_regime_data(num_samples=50, seq_len=seq_len, input_dim=input_dim, regime_type='normal')
    train_data_crisis, train_targets_crisis = generate_synthetic_regime_data(num_samples=50, seq_len=seq_len, input_dim=input_dim, regime_type='crisis')

    # Combine datasets
    train_data = torch.cat([train_data_normal, train_data_crisis], dim=0)
    train_targets = torch.cat([train_targets_normal, train_targets_crisis], dim=0)

    dataset = TensorDataset(train_data, train_targets)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    logger.info("Starting Meta-Training...")
    oswm.meta_train(dataloader, epochs=5)
    logger.info("Meta-Training complete.")

    # 2. One-Shot Adaptation Phase
    logger.info("\n--- Testing One-Shot Adaptation ---")

    # Create a NEW, unseen regime (e.g., specific recovery pattern)
    logger.info("Simulating a new, unseen 'Recovery' regime context...")
    # Generate a single sequence manually
    t = np.linspace(0, 10, seq_len + 1)
    # Recovery: sharp dip then rapid rise
    recovery_trend = np.repeat((-(t - 5)**2 * 0.1)[:, np.newaxis], input_dim, axis=1)
    noise = np.random.normal(0, 0.05, (seq_len + 1, input_dim))
    new_regime_seq = recovery_trend + noise
    new_regime_seq = (new_regime_seq - np.mean(new_regime_seq)) / (np.std(new_regime_seq) + 1e-6)

    # Feed the first T steps as context
    context_steps = 15
    context_data = new_regime_seq[:context_steps]
    future_ground_truth = new_regime_seq[context_steps:]

    logger.info(f"Provided Context: {context_steps} steps of the new regime.")

    # Model predicts next steps based on this context
    # It hasn't seen this specific function before, but should adapt based on the context pattern
    predictions = oswm.adapt_and_predict(context_data, prediction_steps=len(future_ground_truth))

    logger.info("Model Predictions generated.")

    # Compare simple metrics
    mse = np.mean((predictions - future_ground_truth) ** 2)
    logger.info(f"Prediction MSE on unseen regime: {mse:.4f}")

    logger.info("Sample Prediction (Dim 0):")
    logger.info(f"Ground Truth: {future_ground_truth[:, 0]}")
    logger.info(f"Predicted:    {predictions.squeeze()[:, 0]}")

    logger.info("\n--- OSWM Demo Complete ---")

if __name__ == "__main__":
    main()
