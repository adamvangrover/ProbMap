import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from datetime import datetime
import io

logger = logging.getLogger(__name__)

class RiskVisualizationService:
    """
    Service for generating visualizations of risk trajectories.
    """
    def __init__(self):
        pass

    def generate_risk_trajectory_plot(self, risk_history: List[Dict[str, Any]], output_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generates a line plot of Probability of Default (PD) over time, grouped by Industry Sector.

        Args:
            risk_history: List of dictionaries with keys: 'timestamp', 'company_id', 'industry_sector', 'pd'.
            output_path: Optional path to save the plot.

        Returns:
            Bytes of the PNG image if output_path is None, else None.
        """
        if not risk_history:
            logger.warning("No risk history provided for visualization.")
            return None

        df = pd.DataFrame(risk_history)

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Pivot or melt? Seaborn handles long format well.

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="darkgrid")

        # Plot PD over time, hue by Industry Sector
        # If we have many companies, maybe plot average by sector?
        # Let's plot average PD by sector for clarity

        if 'industry_sector' not in df.columns:
            # If industry sector is missing, try to plot by company_id directly if few, or average all
            logger.info("Industry sector not found in history, plotting by company_id")
            sns.lineplot(data=df, x="timestamp", y="pd", hue="company_id", marker="o")
        else:
            # Aggregate by sector and timestamp
            sns.lineplot(data=df, x="timestamp", y="pd", hue="industry_sector", style="company_id", markers=True, dashes=False)

        plt.title("Risk Trajectory: Probability of Default (PD) over Time")
        plt.xlabel("Time")
        plt.ylabel("Probability of Default (PD)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Risk trajectory plot saved to {output_path}")
            plt.close()
            return None
        else:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return buf.getvalue()

    def generate_heatmap(self, risk_snapshot: List[Dict[str, Any]], output_path: Optional[str] = None):
        """
        Generates a heatmap of current risk (PD) across sectors and countries (or other dimensions).
        """
        pass
