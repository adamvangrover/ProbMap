import pytest
import torch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from simulation.oswm import SyntheticCreditPrior, OneShotWorldModel

def test_synthetic_credit_prior_shape():
    batch_size = 4
    seq_len = 10
    state_dim = 5
    prior = SyntheticCreditPrior(state_dim=state_dim, hidden_dim=10)

    inputs, targets = prior.sample_batch(batch_size, seq_len)

    assert inputs.shape == (batch_size, seq_len, state_dim)
    assert targets.shape == (batch_size, seq_len, state_dim)
    assert torch.is_tensor(inputs)
    assert torch.is_tensor(targets)

def test_one_shot_world_model_forward():
    batch_size = 2
    seq_len = 10
    state_dim = 5
    d_model = 16

    model = OneShotWorldModel(state_dim=state_dim, d_model=d_model, nhead=2, num_layers=1)

    # Create dummy input
    src = torch.randn(batch_size, seq_len, state_dim)

    output = model(src)

    assert output.shape == (batch_size, seq_len, state_dim)

def test_overfitting_single_batch():
    """
    Test that the model can reduce loss on a single fixed batch (sanity check for gradients).
    """
    state_dim = 4
    seq_len = 5
    model = OneShotWorldModel(state_dim=state_dim, d_model=16, nhead=2, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    # Fixed data
    inputs = torch.randn(1, seq_len, state_dim)
    targets = torch.randn(1, seq_len, state_dim)

    initial_loss = criterion(model(inputs), targets).item()

    # Train for a few steps
    for _ in range(50):
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(inputs), targets).item()

    assert final_loss < initial_loss
