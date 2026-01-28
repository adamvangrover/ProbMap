import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class OneShotWorldModel(nn.Module):
    """
    A Transformer-based World Model designed for One-Shot (or Few-Shot) adaptation
    to new credit risk regimes.

    It treats the evolution of credit risk factors as a sequence and uses self-attention
    to model temporal dependencies and adapt to the specific "context" of the input sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, nhead: int = 4, dropout: float = 0.1):
        super(OneShotWorldModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding (Simple learned)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, hidden_dim)) # Max seq length 1000

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output head (predicts next state)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        self.criterion = nn.MSELoss()
        self.optimizer = None # Initialized in train_mode

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (batch_size, seq_len, input_dim)
            mask: Optional mask for the sequence
        Returns:
            Predicted sequence of shape (batch_size, seq_len, input_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Embed
        x = self.embedding(x)

        # Add Positional Encoding
        # Slice pos_encoder to match seq_len
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer Pass
        # We use a causal mask so position t can only attend to 0...t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        output = self.transformer_encoder(x, mask=causal_mask, is_causal=True)

        # Decode
        prediction = self.decoder(output)
        return prediction

    def meta_train(self, dataloader, epochs: int = 10, learning_rate: float = 0.001):
        """
        Meta-training phase: Learns the general dynamics across various synthetic credit regimes.
        """
        self.train()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (sequences, targets) in enumerate(dataloader):
                # sequences: (batch, seq_len, dim)
                # targets: (batch, seq_len, dim) - typically sequences shifted by 1

                self.optimizer.zero_grad()
                output = self(sequences)

                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info(f"OSWM Meta-Train Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    def adapt_and_predict(self, context_sequence: np.ndarray, prediction_steps: int = 1) -> np.ndarray:
        """
        One-Shot Adaptation:
        Given a context sequence (history of the current specific regime),
        the model runs forward. The Transformer's attention mechanism implicitly "adapts"
        to the context provided in the sequence to predict the next steps.

        This is "in-context learning" - no gradient updates are strictly needed
        for the model to adjust its predictions based on the immediate history,
        provided it was meta-trained on diverse regimes.
        """
        self.eval()
        with torch.no_grad():
            curr_seq_tensor = torch.FloatTensor(context_sequence).unsqueeze(0) # (1, seq_len, dim)

            predictions = []

            for _ in range(prediction_steps):
                # Forward pass
                output = self(curr_seq_tensor)

                # Get last predicted step
                next_step_pred = output[:, -1, :].unsqueeze(1) # (1, 1, dim)

                predictions.append(next_step_pred.squeeze().numpy())

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
