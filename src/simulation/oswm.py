import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SyntheticCreditPrior:
    """
    Generates synthetic episodes of credit dynamics.
    Each episode is governed by a distinct, randomly initialized 'true' model (prior).
    This simulates the challenge of adapting to a new environment (e.g., a new company or market regime)
    using only in-context data.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def sample_batch(self, batch_size: int, sequence_length: int, device: str = 'cpu'):
        """
        Generates a batch of episodes.

        Args:
            batch_size (int): Number of independent episodes (worlds).
            sequence_length (int): Length of each episode.
            device (str): Device to put tensors on.

        Returns:
            states (torch.Tensor): Shape (batch_size, sequence_length, state_dim)
            targets (torch.Tensor): Shape (batch_size, sequence_length, state_dim).
                                    targets[:, t] is the state at t+1.
        """
        # We need to generate T+1 steps to have T input-target pairs.
        total_steps = sequence_length + 1

        batch_states = []

        # For each episode in the batch, sample a random "true" dynamics model
        for _ in range(batch_size):
            # Random MLP weights for this episode's dynamics
            # s_{t+1} = activation(s_t @ W1 + b1) @ W2 + b2 + s_t (residual)
            w1 = torch.randn(self.state_dim, self.hidden_dim, device=device) / np.sqrt(self.state_dim)
            b1 = torch.zeros(self.hidden_dim, device=device)
            w2 = torch.randn(self.hidden_dim, self.state_dim, device=device) / np.sqrt(self.hidden_dim)
            b2 = torch.zeros(self.state_dim, device=device)

            # Initial state
            current_state = torch.randn(self.state_dim, device=device)
            episode_states = [current_state]

            for _ in range(total_steps - 1):
                # Apply dynamics using tanh for stability
                h = torch.tanh(current_state @ w1 + b1)
                delta = h @ w2 + b2
                # Damping factor to prevent explosion
                next_state = 0.95 * current_state + 0.05 * delta

                episode_states.append(next_state)
                current_state = next_state

            batch_states.append(torch.stack(episode_states))

        batch_tensor = torch.stack(batch_states) # (B, T+1, D)

        inputs = batch_tensor[:, :-1, :] # (B, T, D)
        targets = batch_tensor[:, 1:, :] # (B, T, D)

        return inputs, targets



class OneShotWorldModel(nn.Module):
    """
    A Transformer-based World Model that learns to predict the next state
    given a history of states. It is trained on the SyntheticCreditPrior
    to be able to adapt to new dynamics via in-context learning.
    """
    def __init__(self, state_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model

        # Input embedding: Project state to d_model
        self.input_proj = nn.Linear(state_dim, d_model)

        # Positional Encoding
        # Simple learnable positional encoding
        self.max_len = 1000
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_len, d_model))

        # Transformer Decoder (Causal)
        # We use TransformerEncoder but with a causal mask, effectively acting as a Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model*4, dropout=dropout,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: Project d_model back to state_dim
        self.output_proj = nn.Linear(d_model, state_dim)

    def forward(self, src):
        """
        Args:
            src: (batch_size, seq_len, state_dim)

        Returns:
            output: (batch_size, seq_len, state_dim)
        """
        batch_size, seq_len, _ = src.shape

        # Embed inputs
        x = self.input_proj(src) # (B, T, d_model)

        # Add positional embedding
        if seq_len > self.max_len:
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            x = x + self.pos_embedding[:, :seq_len, :]

        # Create causal mask
        # Mask is (T, T). value is -inf for future positions, 0 for past/present.
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=src.device)

        # Pass through transformer
        x = self.transformer(x, mask=mask, is_causal=True)

        # Project to output
        output = self.output_proj(x)

        return output

    def adapt_and_predict(self, context_sequence: np.ndarray, prediction_steps: int = 1) -> np.ndarray:
        """
        One-Shot Adaptation (Inference Utility):
        Given a context sequence (history of the current specific regime),
        the model runs forward. The Transformer's attention mechanism implicitly "adapts"
        to the context provided in the sequence to predict the next steps.

        Args:
            context_sequence: (seq_len, state_dim) numpy array
            prediction_steps: Number of future steps to predict autoregressively
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            curr_seq_tensor = torch.FloatTensor(context_sequence).unsqueeze(0).to(device) # (1, seq_len, dim)

            predictions = []

            for _ in range(prediction_steps):
                # Forward pass
                output = self.forward(curr_seq_tensor)

                # Get last predicted step
                next_step_pred = output[:, -1, :].unsqueeze(1) # (1, 1, dim)

                predictions.append(next_step_pred.squeeze().cpu().numpy())

                # Append prediction to sequence for autoregressive generation
                curr_seq_tensor = torch.cat([curr_seq_tensor, next_step_pred], dim=1)

            return np.array(predictions)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        logger.info(f"OSWM model saved to {path}")

    def load_model(self, path: str):
        try:
            self.load_state_dict(torch.load(path))
            self.eval()
            logger.info(f"OSWM model loaded from {path}")
            return True
        except FileNotFoundError:
            logger.error(f"OSWM model file not found at {path}")
            return False

class OSWMTrainer:
    def __init__(self, model: OneShotWorldModel, prior: SyntheticCreditPrior, lr: float = 1e-3):
        self.model = model
        self.prior = prior
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, batch_size: int = 32, sequence_length: int = 50, device: str = 'cpu'):
        self.model.train()
        self.optimizer.zero_grad()

        inputs, targets = self.prior.sample_batch(batch_size, sequence_length, device)

        predictions = self.model(inputs)

        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate_adaptation(self, sequence_length: int = 100, device: str = 'cpu'):
        """
        Evaluates how well the model adapts to a single new environment over time.
        Returns the sequence of squared errors.
        """
        self.model.eval()
        with torch.no_grad():
            inputs, targets = self.prior.sample_batch(1, sequence_length, device)
            predictions = self.model(inputs)

            # Calculate error at each step
            # error shape: (1, T, D) -> (T,) mean over D
            errors = (predictions - targets).pow(2).mean(dim=2).squeeze(0)

        return errors.cpu().numpy()