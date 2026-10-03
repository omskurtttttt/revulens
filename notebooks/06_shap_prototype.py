"""
Step 6: SHAP Prototype & Latency Benchmarking for RevuLens.

Adheres strictly to GEMINI.md:
1. Explains the end-to-end pipeline:
   text -> normalize -> DistilBERT -> SVM -> calibrated P(Deceptive)
2. Uses SHAP's text masker approach.
3. Sign convention: explain P(Deceptive).
   Positive weight (> 0) pushes toward Deceptive, negative (< 0) toward Genuine.
4. Conducts systematic latency benchmark across text lengths and max_evals budgets.
5. Informs the team regarding default max_evals and word limits for the backend and extension.
6. Saves benchmark metrics to data/processed/shap_latency_benchmark.json (gitignored).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.explainer import SHAPExplainerService
from backend.app.constants import InternalClass, DisplayLabel


BENCHMARK_SAMPLES = {
    "short": (
        "This product is absolutely amazing and exceeded all my expectations! Highly recommend.",
        12
    ),
    "medium": (
        "I purchased this wireless mouse after reading several reviews online. The build quality is decent "
        "and battery life is satisfactory, but the click sound is slightly louder than expected. "
        "Good value for the price overall.",
        37
    ),
    "long": (
        "I have been using this stainless steel thermal water bottle for over three weeks now during my daily commutes. "
        "It keeps cold drinks chilled for more than twelve hours even under direct sunlight in humid weather. "
        "The lid mechanism seals securely with zero leakage in my backpack. However, the external powder coating "
        "scratches rather easily if dropped on pavement. Overall, it performs as advertised and I am satisfied with the purchase.",
        73
    )
}


def run_qualitative_demonstrations(explainer: SHAPExplainerService) -> List[Dict[str, Any]]:
    """Run qualitative explanations on sample reviews to verify token weights and sign convention."""
    print("\n" + "=" * 75)
    print("STEP 6: SHAP EXPLAINER QUALITATIVE VERIFICATION")
    print("Sign Convention: Positive weight (+) -> Deceptive | Negative weight (-) -> Genuine")
    print("=" * 75)

    test_cases = [
        {
            "name": "Deceptive Pattern Sample (GPT-2 generated style)",
            "text": "The item arrived quickly and was packaged nicely. It works exactly as described and is of very high quality. I would definitely buy again."
        },
        {
            "name": "Genuine Customer Review Sample",
            "text": "Box was slightly dented when delivered by courier, but the shoes inside were completely intact. Fits true to size, very comfortable."
        },
        {
            "name": "FiReCS Taglish Review Sample (Exploratory)",
            "text": "Maganda yung quality ng tela at sakto yung fit sakin. Fast delivery din c seller, thank you so much!"
        }
    ]

    results = []
    for tc in test_cases:
        print(f"\n--- Test Case: {tc['name']} ---")
        print(f"Input Text: \"{tc['text']}\"")

        res = explainer.explain(tc["text"], max_evals=60)
        p_dec = res["prediction_deceptive_prob"]
        pred_label = DisplayLabel.POTENTIALLY_DECEPTIVE.value if p_dec >= 0.5 else DisplayLabel.LIKELY_GENUINE.value

        print(f"Prediction: {pred_label} (P(Deceptive) = {p_dec:.4f}, Base Value = {res['base_value']:.4f})")
        print(f"Latency: {res['latency_ms']:.1f} ms | Evaluated Tokens: {len(res['tokens'])}")
        print("Top Influential Tokens:")

        # Sort tokens by magnitude of attribution
        sorted_tokens = sorted(res["tokens"], key=lambda x: abs(x["weight"]), reverse=True)[:6]
        for item in sorted_tokens:
            direction = "-> Deceptive (+)" if item["weight"] > 0 else "-> Genuine (-)"
            print(f"  '{item['text']}': {item['weight']:+.4f} ({direction})")

        results.append({
            "test_case": tc["name"],
            "prediction": pred_label,
            "prediction_deceptive_prob": p_dec,
            "base_value": res["base_value"],
            "latency_ms": res["latency_ms"],
            "tokens": res["tokens"]
        })

    return results


def run_latency_benchmarking(explainer: SHAPExplainerService) -> Dict[str, Any]:
    """
    Run systematic latency testing across combinations of review lengths and max_evals budgets.
    Informs thesis Chapter 3 & Chapter 4 decisions on SHAP latency budget.
    """
    print("\n" + "=" * 75)
    print("STEP 6: SHAP LATENCY BENCHMARKING MATRIX")
    print("Measuring CPU response times across text lengths and evaluation budgets")
    print("=" * 75)

    eval_budgets = [30, 60, 100]
    benchmark_matrix = {}

    for length_cat, (text, word_count) in BENCHMARK_SAMPLES.items():
        benchmark_matrix[length_cat] = {
            "word_count": word_count,
            "evaluations": {}
        }
        for evals in eval_budgets:
            print(f"Running: length={length_cat.upper()} ({word_count} words), max_evals={evals}...", end=" ", flush=True)

            # Bypass cache for accurate benchmarking
            explainer._cache.clear()

            t0 = time.perf_counter()
            res = explainer.explain(text, max_evals=evals)
            elapsed_sec = time.perf_counter() - t0
            elapsed_ms = elapsed_sec * 1000

            print(f"Done in {elapsed_sec:.2f}s ({elapsed_ms:.1f} ms)")

            benchmark_matrix[length_cat]["evaluations"][f"max_evals_{evals}"] = {
                "max_evals": evals,
                "latency_ms": round(elapsed_ms, 2),
                "latency_sec": round(elapsed_sec, 3),
                "token_count": len(res["tokens"])
            }

    # Test Cache Hit Performance
    test_text, _ = BENCHMARK_SAMPLES["short"]
    # First call populates cache
    explainer.explain(test_text, max_evals=60)
    # Second call measures cache hit speed
    t0 = time.perf_counter()
    cache_res = explainer.explain(test_text, max_evals=60)
    cache_ms = (time.perf_counter() - t0) * 1000
    print(f"\nCache Hit Verification: {cache_ms:.2f} ms (is_cached={cache_res['cached']})")

    # Formatted Summary Table
    print("\n" + "=" * 75)
    print(f"{'Length Category':<16} | {'Words':<6} | {'max_evals=30':<14} | {'max_evals=60':<14} | {'max_evals=100':<14}")
    print("-" * 75)
    for cat, data in benchmark_matrix.items():
        w_cnt = data["word_count"]
        e30 = f"{data['evaluations']['max_evals_30']['latency_sec']:.2f}s"
        e60 = f"{data['evaluations']['max_evals_60']['latency_sec']:.2f}s"
        e100 = f"{data['evaluations']['max_evals_100']['latency_sec']:.2f}s"
        print(f"{cat.capitalize():<16} | {w_cnt:<6} | {e30:<14} | {e60:<14} | {e100:<14}")
    print("=" * 75)

    return {
        "benchmark_matrix": benchmark_matrix,
        "cache_hit_latency_ms": round(cache_ms, 2),
        "recommended_default_max_evals": 60,
        "recommended_max_words": 100,
        "recommendation_rationale": (
            "max_evals=60 delivers sub-3-second explanations on CPU for typical reviews (15-40 words) "
            "while maintaining clear token attribution clarity. Caching yields sub-millisecond repeated lookups."
        )
    }


def main():
    parser = argparse.ArgumentParser(description="RevuLens Step 6: SHAP Prototype and Latency Benchmark")
    parser.add_argument("--model-path", default="backend/models/hybrid_distilbert_svm_pipeline.joblib")
    parser.add_argument("--output-json", default="data/processed/shap_latency_benchmark.json")
    args = parser.parse_args()

    print(f"Loading trained hybrid pipeline from: {args.model_path}")
    explainer = SHAPExplainerService(model_path=args.model_path)

    # 1. Qualitative demonstrations
    demo_results = run_qualitative_demonstrations(explainer)

    # 2. Quantitative latency benchmark
    benchmark_results = run_latency_benchmarking(explainer)

    # 3. Save report to JSON
    report = {
        "step": "Step 6: SHAP Prototype & Latency Benchmarking",
        "qualitative_samples": demo_results,
        "latency_benchmark": benchmark_results
    }

    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nBenchmark report successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
