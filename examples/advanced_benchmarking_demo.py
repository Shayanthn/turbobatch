#!/usr/bin/env python3
"""
🚀 Advanced Dynamic Batching Demo
This example demonstrates advanced features of DynamicBatcher including:
- Real-time performance monitoring
- Memory usage optimization
- Batch size adaptation
- Multiple model comparison
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import DynamicBatcher
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from DynamicBatcher import DynamicBatcher
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_models():
    """Setup multiple models for comparison"""
    models = {
        "distilbert": {
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer": None,
            "model": None,
            "batcher": None,
        },
        "roberta": {
            "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "tokenizer": None,
            "model": None,
            "batcher": None,
        },
    }

    for model_key, model_info in models.items():
        logger.info(f"Loading {model_key} model...")
        model_info["tokenizer"] = AutoTokenizer.from_pretrained(
            model_info["model_name"]
        )
        model_info["model"] = AutoModelForSequenceClassification.from_pretrained(
            model_info["model_name"]
        )

        # Create DynamicBatcher for each model
        model_info["batcher"] = DynamicBatcher(
            model=model_info["model"],
            tokenizer=model_info["tokenizer"],
            max_batch_size=32,
            timeout_ms=100,
            adaptive_batching=True,
            performance_monitoring=True,
        )
        logger.info(f"✅ {model_key} model loaded successfully!")

    return models


def generate_test_data(num_samples: int = 1000) -> List[str]:
    """Generate diverse test data for sentiment analysis"""
    test_texts = [
        "I absolutely love this product! It's amazing and exceeded all my expectations.",
        "This is the worst experience I've ever had. Completely disappointed.",
        "The weather is nice today, perfect for a walk in the park.",
        "I'm feeling quite neutral about this whole situation.",
        "Fantastic work! This is exactly what I was looking for.",
        "Terrible quality, would not recommend to anyone.",
        "It's okay, nothing special but does the job.",
        "Outstanding performance and great value for money!",
        "Meh, could be better but it's not the worst thing ever.",
        "Incredible! This has changed my life completely!",
        "Not bad, but there's definitely room for improvement.",
        "Perfect! Everything works exactly as described.",
        "Disappointing results, expected much more from this.",
        "Surprisingly good quality considering the price point.",
        "Average product, meets basic requirements but nothing more.",
    ]

    # Generate random combinations and variations
    generated_texts = []
    for i in range(num_samples):
        base_text = test_texts[i % len(test_texts)]
        # Add some variation
        if i % 3 == 0:
            base_text = f"In my opinion, {base_text.lower()}"
        elif i % 3 == 1:
            base_text = f"{base_text} What do you think?"

        generated_texts.append(base_text)

    return generated_texts


def benchmark_models(
    models: Dict, test_data: List[str], batch_sizes: List[int] = [1, 4, 8, 16, 32]
):
    """Comprehensive benchmarking of different models and batch sizes"""
    results = {}

    for model_name, model_info in models.items():
        logger.info(f"\n🔄 Benchmarking {model_name} model...")
        results[model_name] = {}

        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")

            # Update batch size
            model_info["batcher"].max_batch_size = batch_size

            # Warm up
            _ = model_info["batcher"].predict(test_data[:10])

            # Benchmark
            start_time = time.time()
            predictions = model_info["batcher"].predict(test_data)
            end_time = time.time()

            # Calculate metrics
            total_time = end_time - start_time
            throughput = len(test_data) / total_time

            results[model_name][batch_size] = {
                "total_time": total_time,
                "throughput": throughput,
                "predictions": len(predictions),
            }

            logger.info(f"    ⏱️  Time: {total_time:.2f}s")
            logger.info(f"    🚀 Throughput: {throughput:.2f} samples/sec")

    return results


def plot_results(results: Dict):
    """Create beautiful visualizations of benchmark results"""
    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "🚀 Dynamic Batching Performance Analysis", fontsize=16, fontweight="bold"
    )

    # Extract data for plotting
    models = list(results.keys())
    batch_sizes = list(results[models[0]].keys())

    # Plot 1: Throughput comparison
    for model in models:
        throughputs = [results[model][bs]["throughput"] for bs in batch_sizes]
        ax1.plot(
            batch_sizes, throughputs, marker="o", linewidth=2, label=model.capitalize()
        )

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (samples/sec)")
    ax1.set_title("📈 Throughput vs Batch Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total time comparison
    for model in models:
        times = [results[model][bs]["total_time"] for bs in batch_sizes]
        ax2.plot(batch_sizes, times, marker="s", linewidth=2, label=model.capitalize())

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Total Time (seconds)")
    ax2.set_title("⏱️ Processing Time vs Batch Size")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Efficiency heatmap
    efficiency_data = []
    for model in models:
        model_efficiencies = []
        for bs in batch_sizes:
            efficiency = (
                results[model][bs]["throughput"] / bs
            )  # Samples per second per batch unit
            model_efficiencies.append(efficiency)
        efficiency_data.append(model_efficiencies)

    sns.heatmap(
        efficiency_data,
        xticklabels=batch_sizes,
        yticklabels=[m.capitalize() for m in models],
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax3,
    )
    ax3.set_title("🔥 Efficiency Heatmap (Throughput/Batch Size)")
    ax3.set_xlabel("Batch Size")
    ax3.set_ylabel("Model")

    # Plot 4: Speedup comparison (relative to batch size 1)
    for model in models:
        baseline_time = results[model][1]["total_time"]
        speedups = [
            baseline_time / results[model][bs]["total_time"] for bs in batch_sizes
        ]
        ax4.bar(
            [f"{model.capitalize()}\n{bs}" for bs in batch_sizes],
            speedups,
            alpha=0.7,
            label=model.capitalize(),
        )

    ax4.set_ylabel("Speedup Factor")
    ax4.set_title("⚡ Speedup Factor (vs Batch Size 1)")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def demonstrate_adaptive_batching():
    """Demonstrate adaptive batching capabilities"""
    logger.info("\n🤖 Demonstrating Adaptive Batching...")

    # Setup a model with adaptive batching
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    batcher = DynamicBatcher(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=32,
        timeout_ms=50,
        adaptive_batching=True,
        performance_monitoring=True,
    )

    # Simulate varying load
    logger.info("Simulating varying input load...")

    loads = [
        ("Low load", ["I love this!", "Great product"]),
        ("Medium load", generate_test_data(10)),
        ("High load", generate_test_data(50)),
        ("Burst load", generate_test_data(100)),
    ]

    for load_name, data in loads:
        logger.info(f"\n📊 Testing {load_name} ({len(data)} samples)...")

        start_time = time.time()
        predictions = batcher.predict(data)
        end_time = time.time()

        # Get performance stats
        stats = batcher.get_performance_stats()

        logger.info(f"  ✅ Processed {len(predictions)} predictions")
        logger.info(f"  ⏱️  Time: {end_time - start_time:.2f}s")
        logger.info(f"  📈 Avg batch size: {stats.get('avg_batch_size', 'N/A')}")
        logger.info(f"  🎯 Cache hit rate: {stats.get('cache_hit_rate', 'N/A')}")


def main():
    """Main execution function"""
    logger.info("🚀 Starting Advanced Dynamic Batching Demo...")

    try:
        # Generate test data
        logger.info("📝 Generating test data...")
        test_data = generate_test_data(100)  # Smaller set for demo
        logger.info(f"✅ Generated {len(test_data)} test samples")

        # Setup models
        models = setup_models()

        # Run benchmarks
        logger.info("\n🏁 Starting comprehensive benchmarks...")
        batch_sizes = [1, 4, 8, 16]  # Smaller range for demo
        results = benchmark_models(models, test_data, batch_sizes)

        # Display results
        logger.info("\n📊 Benchmark Results Summary:")
        for model_name, model_results in results.items():
            logger.info(f"\n{model_name.upper()} Model:")
            for batch_size, metrics in model_results.items():
                logger.info(
                    f"  Batch Size {batch_size:2d}: {metrics['throughput']:6.2f} samples/sec"
                )

        # Create visualizations
        logger.info("\n📈 Creating performance visualizations...")
        plot_results(results)

        # Demonstrate adaptive batching
        demonstrate_adaptive_batching()

        logger.info("\n🎉 Demo completed successfully!")
        logger.info("📸 Check 'benchmark_results.png' for detailed performance charts")

    except Exception as e:
        logger.error(f"❌ Error during demo: {str(e)}")
        raise


if __name__ == "__main__":
    main()
