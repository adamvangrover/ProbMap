import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class BayesianCreditModel:
    """
    A simplified discrete Bayesian model for updating credit risk beliefs (PD)
    based on incoming evidence.

    It models the "True Risk State" of an obligor as a discrete variable
    (e.g., Ratings: AAA, AA, ..., Default).
    """

    # Standard S&P-like one-year default rates (Illustrative priors)
    DEFAULT_RATES = {
        'AAA': 0.0001,
        'AA': 0.0002,
        'A': 0.0006,
        'BBB': 0.0024,
        'BB': 0.0100,
        'B': 0.0400,
        'CCC': 0.2500,
        'D': 1.0000
    }

    STATES = list(DEFAULT_RATES.keys())

    def __init__(self, prior_probs: Optional[Dict[str, float]] = None):
        """
        Args:
            prior_probs: Dictionary mapping rating states to probabilities.
                         Must sum to 1.0. If None, assumes uniform (uninformative)
                         or a standard base distribution.
        """
        if prior_probs:
            self.probs = pd.Series(prior_probs)
        else:
            # Default to a generic investment grade skew
            self.probs = pd.Series({
                'AAA': 0.05, 'AA': 0.10, 'A': 0.20, 'BBB': 0.30,
                'BB': 0.20, 'B': 0.10, 'CCC': 0.04, 'D': 0.01
            })

        self._normalize()
        logger.info("BayesianCreditModel initialized.")

    def _normalize(self):
        total = self.probs.sum()
        if total == 0:
            logger.warning("Probabilities sum to 0. Resetting to uniform.")
            self.probs[:] = 1.0 / len(self.STATES)
        else:
            self.probs = self.probs / total

    def get_expected_pd(self) -> float:
        """
        Calculates the expected PD based on the current belief distribution over states.
        E[PD] = Sum(P(State) * PD(State))
        """
        expected_pd = 0.0
        for state in self.STATES:
            expected_pd += self.probs[state] * self.DEFAULT_RATES[state]
        return expected_pd

    def update(self, evidence_type: str, strength: str = 'medium'):
        """
        Updates the belief distribution based on evidence.

        Args:
            evidence_type: Description of evidence (e.g., 'high_revenue_growth', 'lawsuit').
            strength: 'low', 'medium', 'high' indicating the likelihood ratio impact.
        """
        # Define likelihood ratios (Bayes Factors) for evidence
        # P(Evidence | State) / P(Evidence | Not State) roughly
        # Here we simplify: we multiply the prob of "Good" states by X and "Bad" states by Y

        # Mapping evidence to whether it supports Good or Bad credit
        # This is a rule-based likelihood function for the PoC

        positive_evidence = ['high_revenue_growth', 'positive_earnings', 'new_contract', 'upgrade']
        negative_evidence = ['lawsuit', 'missed_payment', 'downgrade', 'covenant_breach', 'management_churn']

        likelihoods = pd.Series(1.0, index=self.STATES)

        # Impact factors
        impact = 1.5 if strength == 'medium' else (2.0 if strength == 'high' else 1.2)

        if evidence_type in positive_evidence:
            # Boost Good states (AAA to BBB), Penalize Bad states (BB to D)
            likelihoods[['AAA', 'AA', 'A', 'BBB']] *= impact
            likelihoods[['BB', 'B', 'CCC', 'D']] *= (1.0 / impact)
            logger.info(f"Applying Positive Evidence: {evidence_type} (Strength: {strength})")

        elif evidence_type in negative_evidence:
            # Penalize Good states, Boost Bad states
            likelihoods[['AAA', 'AA', 'A', 'BBB']] *= (1.0 / impact)
            likelihoods[['BB', 'B', 'CCC', 'D']] *= impact
            logger.info(f"Applying Negative Evidence: {evidence_type} (Strength: {strength})")

        else:
            logger.warning(f"Unknown evidence type '{evidence_type}'. No update applied.")
            return

        # Bayes Update: Posterior propto Prior * Likelihood
        self.probs = self.probs * likelihoods
        self._normalize()

    def get_most_likely_rating(self) -> str:
        return self.probs.idxmax()

    def get_belief_distribution(self) -> Dict[str, float]:
        return self.probs.to_dict()
