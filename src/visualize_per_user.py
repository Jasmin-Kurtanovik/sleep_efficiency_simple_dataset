"""
Visualization generator for per-user model evaluation.
Creates individual plots for each user and combined subplot visualization.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class PerUserVisualizer:
    """Generate visualizations for per-user evaluation results."""
    
    def __init__(self, output_dir: str = 'evaluation_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / 'plots_per_user'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def create_individual_user_plots(
        self,
        user_predictions: Dict[int, List[float]],
        user_actuals: Dict[int, List[float]],
        per_user_metrics: Dict[int, Dict],
        model_name: str = "LSTM Model"
    ):
        """
        Create individual plots for each user.
        
        Args:
            user_predictions: {user_id: [predictions]}
            user_actuals: {user_id: [actuals]}
            per_user_metrics: {user_id: {metrics_dict}}
        """
        if not user_predictions:
            logger.warning('No user predictions to visualize')
            return
        for user_id in sorted(user_predictions.keys()):
            predictions = user_predictions[user_id]
            actuals = user_actuals[user_id]
            metrics = per_user_metrics[user_id]
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'User {user_id} - {model_name} Performance', fontsize=16, fontweight='bold')
            
            # Plot 1: Time series prediction vs actual
            ax = axes[0, 0]
            x_range = range(len(actuals))
            ax.plot(x_range, actuals, 'o-', label='Actual', linewidth=2, markersize=6, color='#1f77b4')
            ax.plot(x_range, predictions, 's--', label='Predicted', linewidth=2, markersize=6, color='#ff7f0e')
            ax.set_xlabel('Sequence Index')
            ax.set_ylabel('Sleep Efficiency')
            ax.set_title('Predictions vs Actual Sleep Efficiency Over Time')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Plot 2: Residuals
            ax = axes[0, 1]
            residuals = np.array(actuals) - np.array(predictions)
            colors = ['red' if r < 0 else 'green' for r in residuals]
            ax.bar(x_range, residuals, color=colors, alpha=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Sequence Index')
            ax.set_ylabel('Residual (Actual - Predicted)')
            ax.set_title('Prediction Errors')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Scatter plot
            ax = axes[1, 0]
            ax.scatter(actuals, predictions, alpha=0.6, s=100, color='#2ca02c')
            min_val, max_val = 0, 1
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Sleep Efficiency')
            ax.set_ylabel('Predicted Sleep Efficiency')
            ax.set_title(f'R² Score: {metrics["r2"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([min_val, max_val])
            ax.set_ylim([min_val, max_val])
            
            # Plot 4: Metrics summary
            ax = axes[1, 1]
            ax.axis('off')
            metrics_text = f"""
            Performance Metrics:
            
            RMSE: {metrics['rmse']:.4f}
            MAE: {metrics['mae']:.4f}
            R² Score: {metrics['r2']:.4f}
            
            Data Summary:
            
            Sequences: {metrics['n_sequences']}
            Avg Efficiency: {metrics['avg_efficiency']:.3f} ± {metrics['std_efficiency']:.3f}
            Min Efficiency: {metrics['min_efficiency']:.3f}
            Max Efficiency: {metrics['max_efficiency']:.3f}
            
            Model Output:
            
            Avg Prediction: {metrics['avg_prediction']:.3f}
            Std Prediction: {metrics['std_prediction']:.3f}
            """
            ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            output_path = self.plots_dir / f'user_{user_id}_performance.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f'Saved plot: {output_path}')
            plt.close()
    
    def create_combined_subplot(
        self,
        user_predictions: Dict[int, List[float]],
        user_actuals: Dict[int, List[float]],
        per_user_metrics: Dict[int, Dict]
    ):
        """
        Create combined subplots for all users (time series only).
        """
        if not user_predictions:
            logger.warning('No user predictions for combined subplot')
            return
        user_ids = sorted(user_predictions.keys())
        n_users = len(user_ids)
        
        # Create grid layout - roughly square
        n_cols = int(np.ceil(np.sqrt(n_users)))
        n_rows = int(np.ceil(n_users / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.suptitle('All Test Users - Predictions vs Actual Sleep Efficiency', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten() if n_users > 1 else [axes]
        
        for idx, user_id in enumerate(user_ids):
            ax = axes_flat[idx]
            predictions = user_predictions[user_id]
            actuals = user_actuals[user_id]
            metrics = per_user_metrics[user_id]
            
            x_range = range(len(actuals))
            ax.plot(x_range, actuals, 'o-', label='Actual', linewidth=2, markersize=4, color='#1f77b4')
            ax.plot(x_range, predictions, 's--', label='Predicted', linewidth=2, markersize=4, color='#ff7f0e')
            
            ax.set_title(f'User {user_id} (R²={metrics["r2"]:.3f}, RMSE={metrics["rmse"]:.3f})', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Sequence', fontsize=9)
            ax.set_ylabel('Sleep Efficiency', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for idx in range(n_users, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        output_path = self.output_dir / 'combined_all_users.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f'Saved combined plot: {output_path}')
        plt.close()
    
    def create_metrics_comparison_plot(self, per_user_metrics: Dict[int, Dict]):
        """Create comparison plots for all users' metrics."""
        if not per_user_metrics:
            logger.warning('No metrics to compare')
            return
        user_ids = sorted(per_user_metrics.keys())
        rmse_scores = [per_user_metrics[uid]['rmse'] for uid in user_ids]
        r2_scores = [per_user_metrics[uid]['r2'] for uid in user_ids]
        mae_scores = [per_user_metrics[uid]['mae'] for uid in user_ids]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Performance Metrics Across All Test Users', fontsize=14, fontweight='bold')
        
        # RMSE comparison
        ax = axes[0]
        colors_rmse = ['green' if r < 0.1 else 'orange' if r < 0.15 else 'red' for r in rmse_scores]
        ax.bar(range(len(user_ids)), rmse_scores, color=colors_rmse, alpha=0.7)
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Squared Error')
        ax.set_xticks(range(len(user_ids)))
        ax.set_xticklabels([f'U{uid}' for uid in user_ids], fontsize=9)
        ax.axhline(y=np.mean(rmse_scores), color='black', linestyle='--', label=f'Mean: {np.mean(rmse_scores):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # R² comparison
        ax = axes[1]
        colors_r2 = ['green' if r > 0.8 else 'orange' if r > 0.7 else 'red' for r in r2_scores]
        ax.bar(range(len(user_ids)), r2_scores, color=colors_r2, alpha=0.7)
        ax.set_ylabel('R² Score')
        ax.set_title('R² Score (Higher is Better)')
        ax.set_xticks(range(len(user_ids)))
        ax.set_xticklabels([f'U{uid}' for uid in user_ids], fontsize=9)
        ax.axhline(y=np.mean(r2_scores), color='black', linestyle='--', label=f'Mean: {np.mean(r2_scores):.4f}')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # MAE comparison
        ax = axes[2]
        colors_mae = ['green' if m < 0.08 else 'orange' if m < 0.12 else 'red' for m in mae_scores]
        ax.bar(range(len(user_ids)), mae_scores, color=colors_mae, alpha=0.7)
        ax.set_ylabel('MAE')
        ax.set_title('Mean Absolute Error')
        ax.set_xticks(range(len(user_ids)))
        ax.set_xticklabels([f'U{uid}' for uid in user_ids], fontsize=9)
        ax.axhline(y=np.mean(mae_scores), color='black', linestyle='--', label=f'Mean: {np.mean(mae_scores):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f'Saved metrics comparison: {output_path}')
        plt.close()


def create_model_comparison_plot(
    results_att_path: str = 'evaluation_results/evaluation_results.json',
    results_no_att_path: str = 'evaluation_results/evaluation_results_without_att.json',
    output_dir: str = 'evaluation_results'
):
    """
    Create comparison plots between models with and without attention.
    
    Args:
        results_att_path: Path to attention model results
        results_no_att_path: Path to no-attention model results
        output_dir: Directory to save comparison plots
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    try:
        with open(results_att_path, 'r') as f:
            att_results = json.load(f)
        with open(results_no_att_path, 'r') as f:
            no_att_results = json.load(f)
    except FileNotFoundError as e:
        logger.warning(f'Could not load results for comparison: {e}')
        return
    
    att_agg = att_results.get('aggregated_metrics') or att_results.get('aggregated', {})
    no_att_agg = no_att_results.get('aggregated_metrics') or no_att_results.get('aggregated', {})
    
    # Extract metrics
    models = ['With Attention', 'Without Attention']
    rmse_vals = [att_agg.get('rmse', 0), no_att_agg.get('rmse', 0)]
    mae_vals = [att_agg.get('mae', 0), no_att_agg.get('mae', 0)]
    r2_vals = [att_agg.get('r2', 0), no_att_agg.get('r2', 0)]
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Comparison: LSTM with vs without Attention', fontsize=16, fontweight='bold')
    
    # RMSE comparison
    ax = axes[0]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax.bar(models, rmse_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Root Mean Squared Error (Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, rmse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # MAE comparison
    ax = axes[1]
    bars = ax.bar(models, mae_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Mean Absolute Error (Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, mae_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # R² comparison
    ax = axes[2]
    bars = ax.bar(models, r2_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_ylim([-5, 1])
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, val in zip(bars, r2_vals):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        y_pos = height + 0.1 if height >= 0 else height - 0.1
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.4f}', ha='center', va=va, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir_path / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f'Saved model comparison plot: {output_path}')
    plt.close()


def load_evaluation_results(results_path: str = 'evaluation_results/evaluation_results.json') -> Tuple[Dict, Dict]:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results['per_user'], results['aggregated']


def main():
    """Generate all visualizations from evaluation results."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    results_path = 'evaluation_results/evaluation_results.json'
    
    if not Path(results_path).exists():
        print(f"Results not found at {results_path}")
        print("Run evaluate_per_user.py first to generate results.")
        return
    
    # Load results
    per_user_metrics, agg_metrics = load_evaluation_results(results_path)
    
    print(f"\nLoaded results for {len(per_user_metrics)} users")
    print(f"Aggregated RMSE: {agg_metrics['rmse']:.4f}")
    
    # Note: This is a template. The actual user_predictions and user_actuals
    # need to be passed from evaluate_per_user.py or saved to disk.
    # For now, this shows the visualization structure.


if __name__ == '__main__':
    main()
