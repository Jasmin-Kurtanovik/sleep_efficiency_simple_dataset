"""
Unified runner for training, validating, testing, and evaluating both LSTM models.
Trains and evaluates both the attention-based and attention-free variants in sequence.
"""
import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import training and testing modules
from src import train_validate
from src import train_validate_without_att
from src import test_model
from src import test_model_without_att
from src.visualize_per_user import create_model_comparison_plot


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def run_all():
    """Run both models: train, validate, test, and evaluate."""
    
    print_header("UNIFIED MODEL RUNNER - LSTM SLEEP EFFICIENCY PREDICTION")
    print("This script will train and evaluate both models:")
    print("  1. LSTM with Attention")
    print("  2. LSTM without Attention")
    print()
    
    results = {}
    
    # ========================================================================
    # MODEL 1: LSTM with Attention
    # ========================================================================
    print_header("MODEL 1: LSTM WITH ATTENTION")
    
    print("\n[1/2] Training model with attention...")
    start_time = time.time()
    try:
        train_validate.main()
        results['attention'] = {'training': 'success'}
        print(f"[OK] Training completed in {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"[FAILED] Training failed: {e}")
        results['attention'] = {'training': 'failed', 'error': str(e)}
        return results
    
    print("\n[2/2] Evaluating model with attention...")
    start_time = time.time()
    try:
        test_model.main()
        print(f"[OK] Evaluation completed in {time.time() - start_time:.1f}s")
        results['attention']['evaluation'] = 'success'
    except Exception as e:
        print(f"[FAILED] Evaluation failed: {e}")
        results['attention']['evaluation'] = 'failed'
        results['attention']['error'] = str(e)
        return results
    
    # Load attention model results
    try:
        with open("evaluation_results/evaluation_results.json", 'r') as f:
            att_results = json.load(f)
            # Handle both old and new JSON formats
            if 'aggregated' in att_results:
                results['attention']['metrics'] = att_results['aggregated']
            elif 'aggregated_metrics' in att_results:
                results['attention']['metrics'] = att_results['aggregated_metrics']
    except Exception as e:
        print(f"Warning: Could not load attention model results: {e}")
    
    # ========================================================================
    # MODEL 2: LSTM without Attention
    # ========================================================================
    print_header("MODEL 2: LSTM WITHOUT ATTENTION")
    
    print("\n[1/2] Training model without attention...")
    start_time = time.time()
    try:
        train_validate_without_att.main()
        results['no_attention'] = {'training': 'success'}
        print(f"[OK] Training completed in {time.time() - start_time:.1f}s")
    except Exception as e:
        print(f"[FAILED] Training failed: {e}")
        results['no_attention'] = {'training': 'failed', 'error': str(e)}
        return results
    
    print("\n[2/2] Evaluating model without attention...")
    start_time = time.time()
    try:
        test_model_without_att.main()
        print(f"[OK] Evaluation completed in {time.time() - start_time:.1f}s")
        results['no_attention']['evaluation'] = 'success'
    except Exception as e:
        print(f"[FAILED] Evaluation failed: {e}")
        results['no_attention']['evaluation'] = 'failed'
        results['no_attention']['error'] = str(e)
        return results
    
    # Load no-attention model results
    try:
        with open("evaluation_results/evaluation_results_without_att.json", 'r') as f:
            no_att_results = json.load(f)
            # Handle both old and new JSON formats
            if 'aggregated' in no_att_results:
                results['no_attention']['metrics'] = no_att_results['aggregated']
            elif 'aggregated_metrics' in no_att_results:
                results['no_attention']['metrics'] = no_att_results['aggregated_metrics']
    except Exception as e:
        print(f"Warning: Could not load no-attention model results: {e}")
    
    # ========================================================================
    # COMPARISON & SUMMARY
    # ========================================================================
    print_header("MODEL COMPARISON SUMMARY")
    
    print("\n[METRICS] AGGREGATED METRICS COMPARISON:\n")
    print(f"{'Metric':<15} {'With Attention':<20} {'Without Attention':<20}")
    print("-" * 55)
    
    if 'metrics' in results.get('attention', {}) and 'metrics' in results.get('no_attention', {}):
        att_metrics = results['attention']['metrics']
        no_att_metrics = results['no_attention']['metrics']
        
        metrics_to_compare = ['rmse', 'mae', 'r2']
        for metric in metrics_to_compare:
            att_val = att_metrics.get(metric, 'N/A')
            no_att_val = no_att_metrics.get(metric, 'N/A')
            
            if isinstance(att_val, float) and isinstance(no_att_val, float):
                print(f"{metric.upper():<15} {att_val:<20.6f} {no_att_val:<20.6f}")
            else:
                print(f"{metric.upper():<15} {str(att_val):<20} {str(no_att_val):<20}")
    else:
        print("Could not load metrics for comparison")
    
    # Generate comparison plot
    print("\n[PLOTS] Generating model comparison plot...")
    try:
        create_model_comparison_plot(
            results_att_path="evaluation_results/evaluation_results.json",
            results_no_att_path="evaluation_results/evaluation_results_without_att.json",
            output_dir="evaluation_results"
        )
        print("[OK] Model comparison plot saved to evaluation_results/model_comparison.png")
    except Exception as e:
        print(f"[WARNING] Could not generate comparison plot: {e}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("  RUN COMPLETE")
    print("="*70)
    print("\n[OK] Both models trained and evaluated successfully!")
    print("\n[FILES] Output files:")
    print("  - Models: models/best_model.pt, models/best_model_without_att.pt")
    print("  - Results: evaluation_results/evaluation_results.json")
    print("            evaluation_results/evaluation_results_without_att.json")
    print("  - Plots: evaluation_results/plots_per_user/")
    print("           evaluation_results_without_att/plots_per_user/")
    print()
    
    return results


if __name__ == "__main__":
    try:
        results = run_all()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
