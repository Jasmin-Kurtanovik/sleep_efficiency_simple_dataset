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
from src import train_validate_transformer
from src import test_model
from src import test_model_without_att
from src import test_model_transformer
from src.visualize_per_user import create_model_comparison_plot


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def load_metrics(results_path):
    """Load aggregated metrics from a saved evaluation JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
        if 'aggregated' in results:
            return results['aggregated']
        if 'aggregated_metrics' in results:
            return results['aggregated_metrics']
    return None


def run_all():
    """Run all three models: train, validate, test, and evaluate."""

    def run_pipeline(title, train_fn, test_fn, results_path, result_key):
        print_header(title)

        print("\n[1/2] Training model...")
        start_time = time.time()
        try:
            train_fn.main()
            print(f"[OK] Training completed in {time.time() - start_time:.1f}s")
        except Exception as e:
            print(f"[FAILED] Training failed: {e}")
            return {"training": "failed", "error": str(e)}

        print("\n[2/2] Evaluating model...")
        start_time = time.time()
        try:
            test_fn.main()
            print(f"[OK] Evaluation completed in {time.time() - start_time:.1f}s")
        except Exception as e:
            print(f"[FAILED] Evaluation failed: {e}")
            return {"training": "success", "evaluation": "failed", "error": str(e)}

        metrics = None
        try:
            metrics = load_metrics(results_path)
        except Exception as e:
            print(f"Warning: Could not load {result_key} model results: {e}")

        payload = {"training": "success", "evaluation": "success"}
        if metrics is not None:
            payload["metrics"] = metrics
        return payload

    print_header("UNIFIED MODEL RUNNER - LSTM SLEEP EFFICIENCY PREDICTION")
    print("This script will train and evaluate three models:")
    print("  1. LSTM with Attention")
    print("  2. LSTM without Attention")
    print("  3. Transformer")
    print()

    results = {}

    results['attention'] = run_pipeline(
        "MODEL 1: LSTM WITH ATTENTION",
        train_validate,
        test_model,
        "evaluation_results/evaluation_results.json",
        "attention",
    )
    if results['attention'].get('training') != 'success' or results['attention'].get('evaluation') != 'success':
        return results

    results['no_attention'] = run_pipeline(
        "MODEL 2: LSTM WITHOUT ATTENTION",
        train_validate_without_att,
        test_model_without_att,
        "evaluation_results/evaluation_results_without_att.json",
        "no-attention",
    )
    if results['no_attention'].get('training') != 'success' or results['no_attention'].get('evaluation') != 'success':
        return results

    results['transformer'] = run_pipeline(
        "MODEL 3: TRANSFORMER",
        train_validate_transformer,
        test_model_transformer,
        "evaluation_results_transformer/evaluation_results_transformer.json",
        "transformer",
    )
    if results['transformer'].get('training') != 'success' or results['transformer'].get('evaluation') != 'success':
        return results

    print_header("MODEL COMPARISON SUMMARY")

    print("\n[METRICS] AGGREGATED METRICS COMPARISON:\n")
    print(f"{'Metric':<15} {'With Attention':<20} {'Without Attention':<20} {'Transformer':<20}")
    print("-" * 76)

    if all('metrics' in results.get(key, {}) for key in ('attention', 'no_attention', 'transformer')):
        model_metrics = {
            'attention': results['attention']['metrics'],
            'no_attention': results['no_attention']['metrics'],
            'transformer': results['transformer']['metrics'],
        }

        for metric in ['rmse', 'mae', 'r2']:
            values = [model_metrics[key].get(metric, 'N/A') for key in ('attention', 'no_attention', 'transformer')]
            if all(isinstance(value, float) for value in values):
                print(f"{metric.upper():<15} {values[0]:<20.6f} {values[1]:<20.6f} {values[2]:<20.6f}")
            else:
                print(f"{metric.upper():<15} {str(values[0]):<20} {str(values[1]):<20} {str(values[2]):<20}")
    else:
        print("Could not load metrics for comparison")

    print("\n[PLOTS] Generating model comparison plot...")
    try:
        create_model_comparison_plot(
            results_att_path="evaluation_results/evaluation_results.json",
            results_no_att_path="evaluation_results/evaluation_results_without_att.json",
            results_transformer_path="evaluation_results_transformer/evaluation_results_transformer.json",
            output_dir="evaluation_results",
        )
        print("[OK] Model comparison plot saved to evaluation_results/model_comparison.png")
    except Exception as e:
        print(f"[WARNING] Could not generate comparison plot: {e}")

    print("\n" + "="*70)
    print("  RUN COMPLETE")
    print("="*70)
    print("\n[OK] All three models trained and evaluated successfully!")
    print("\n[FILES] Output files:")
    print("  - Models: models/best_model.pt, models/best_model_without_att.pt, models/best_model_transformer.pt")
    print("  - Results: evaluation_results/evaluation_results.json")
    print("            evaluation_results/evaluation_results_without_att.json")
    print("            evaluation_results_transformer/evaluation_results_transformer.json")
    print("  - Plots: evaluation_results/plots_per_user/")
    print("           evaluation_results_without_att/plots_per_user/")
    print("           evaluation_results_transformer/plots_per_user/")
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
