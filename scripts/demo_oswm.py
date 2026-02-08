import sys
import os
import torch
import numpy as np
import logging

# Add src to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from simulation.oswm import OneShotWorldModel, SyntheticCreditPrior, OSWMTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing One-Shot World Model Demo...")

    # Hyperparameters
    STATE_DIM = 10
    D_MODEL = 32
    NHEAD = 2
    NUM_LAYERS = 2
    BATCH_SIZE = 16
    SEQ_LEN = 50
    TRAIN_STEPS = 500

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize components
    prior = SyntheticCreditPrior(state_dim=STATE_DIM, hidden_dim=32)
    model = OneShotWorldModel(state_dim=STATE_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS).to(device)
    trainer = OSWMTrainer(model, prior, lr=1e-3)

    logger.info("Starting training on synthetic prior...")

    # Training Loop
    for step in range(TRAIN_STEPS):
        loss = trainer.train_step(batch_size=BATCH_SIZE, sequence_length=SEQ_LEN, device=device)

        if (step + 1) % 50 == 0:
            logger.info(f"Step {step+1}/{TRAIN_STEPS} | Loss: {loss:.4f}")

    logger.info("Training complete.")

    # Evaluation: In-Context Learning
    logger.info("Evaluating in-context adaptation on a new synthetic environment...")

    # We generate a longer sequence to see adaptation
    eval_len = 100
    errors = trainer.evaluate_adaptation(sequence_length=eval_len, device=device)

    # We expect error to decrease as the model sees more context of the current environment
    # Let's average errors over windows to see the trend clearer
    window = 10
    smoothed_errors = [np.mean(errors[i:i+window]) for i in range(0, len(errors), window)]

    logger.info("Mean Squared Error over time (binned by 10 steps):")
    for i, err in enumerate(smoothed_errors):
        logger.info(f"Steps {i*window}-{(i+1)*window}: MSE = {err:.4f}")

    # Check if adaptation occurred (simple check: early error > late error)
    early_error = np.mean(errors[:10])
    late_error = np.mean(errors[-10:])

    logger.info(f"Early Error (first 10 steps): {early_error:.4f}")
    logger.info(f"Late Error (last 10 steps): {late_error:.4f}")

    if late_error < early_error:
        logger.info("SUCCESS: Model demonstrates in-context learning (error decreased).")
    else:
        logger.info("NOTE: Adaptation might not be visible with this short training or model size.")

if __name__ == "__main__":
    main()
