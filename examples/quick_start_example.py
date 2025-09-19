#!/usr/bin/env python3
"""
🚀 DynamicBatcher: Quick Start Example

This example demonstrates basic usage of DynamicBatcher with mock data
to avoid downloading large models. Perfect for testing and understanding
the library's functionality.

Run this example:
    python examples/quick_start_example.py
"""

import os
import sys
import time

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from turbobatch import DynamicBatcher


class SimpleMockModel(nn.Module):
    """Simple mock model for demonstration purposes."""

    def __init__(self, vocab_size=1000, hidden_size=128, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embeddings = self.embedding(input_ids)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = embeddings.mean(dim=1)

        logits = self.classifier(pooled)

        # Return object with logits attribute (like HuggingFace models)
        class ModelOutput:
            def __init__(self, logits):
                self.logits = logits

        return ModelOutput(logits)


class SimpleMockTokenizer:
    """Simple mock tokenizer for demonstration purposes."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        **kwargs,
    ):
        # Convert text to simple token IDs based on word hash
        words = text.lower().split()
        input_ids = [hash(word) % (self.vocab_size - 1) + 1 for word in words]

        if add_special_tokens:
            input_ids = [101] + input_ids + [102]  # CLS and SEP tokens

        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        result = {"input_ids": input_ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)

        return result

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=True,
        truncation=True,
        max_length=512,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        all_input_ids = []
        all_attention_masks = []

        for text in texts:
            encoded = self.encode_plus(
                text, max_length=max_length, truncation=truncation
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])

        # Pad to same length
        if padding:
            max_len = max(len(ids) for ids in all_input_ids)
            for i in range(len(all_input_ids)):
                pad_length = max_len - len(all_input_ids[i])
                all_input_ids[i].extend([self.pad_token_id] * pad_length)
                all_attention_masks[i].extend([0] * pad_length)

        result = {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(all_attention_masks),
        }

        return result


def main():
    print("🚀 DynamicBatcher Quick Start Example")
    print("=" * 50)

    # Create mock model and tokenizer
    print("📦 Setting up mock model and tokenizer...")
    model = SimpleMockModel()
    tokenizer = SimpleMockTokenizer()

    # Create DynamicBatcher
    print("⚡ Creating DynamicBatcher...")
    batcher = DynamicBatcher(
        model=model,
        tokenizer=tokenizer,
        max_batch_size=8,
        max_sequence_length=128,
        adaptive_batching=True,
        performance_monitoring=True,
    )

    # Sample texts for sentiment analysis
    sample_texts = [
        "I absolutely love this amazing product!",
        "This is the worst experience I've ever had.",
        "It's okay, nothing particularly special about it.",
        "Outstanding quality and excellent customer service!",
        "Poor design and terrible build quality.",
        "Great value for money, highly recommended!",
        "Disappointing performance, would not buy again.",
        "Perfect! Exactly what I was looking for.",
        "Could be better, but it serves its purpose.",
        "Excellent product with fast shipping and great packaging!",
    ]

    print(f"📝 Processing {len(sample_texts)} sample texts...")

    # Single prediction example
    print("\n🔍 Single Prediction Example:")
    single_text = "This product is absolutely fantastic!"
    start_time = time.time()
    single_result = batcher.predict(single_text)
    end_time = time.time()

    sentiment = "POSITIVE" if single_result.label == 1 else "NEGATIVE"
    print(f'Text: "{single_text}"')
    print(f"Prediction: {sentiment} (confidence: {single_result.score:.3f})")
    print(f"Processing time: {(end_time - start_time)*1000:.2f}ms")

    # Batch prediction example
    print("\n📊 Batch Prediction Example:")
    start_time = time.time()
    batch_results = batcher.predict(sample_texts)
    end_time = time.time()

    print(
        f"Processed {len(batch_results)} texts in {(end_time - start_time)*1000:.2f}ms"
    )
    print(
        f"Average time per text: {((end_time - start_time) / len(sample_texts))*1000:.2f}ms"
    )

    print("\nResults:")
    for i, (text, result) in enumerate(zip(sample_texts, batch_results)):
        sentiment = "POSITIVE" if result.label == 1 else "NEGATIVE"
        print(f"{i+1:2d}. {sentiment:8s} ({result.score:.3f}) | {text[:40]}...")

    # Performance statistics
    print("\n📈 Performance Statistics:")
    stats = batcher.get_performance_stats()
    for key, value in stats.items():
        if key != "device":
            print(f"  {key}: {value}")

    # Test caching
    print("\n💾 Testing Cache Performance:")
    # Run same predictions again
    start_time = time.time()
    cached_results = batcher.predict(sample_texts[:5])  # First 5 texts
    end_time = time.time()

    print(f"Cached prediction time: {(end_time - start_time)*1000:.2f}ms")
    print("Cache statistics:")
    updated_stats = batcher.get_performance_stats()
    print(f"  Cache hit rate: {updated_stats['cache_hit_rate']:.1f}%")
    print(f"  Cache hits: {updated_stats['cache_hits']}")
    print(f"  Cache misses: {updated_stats['cache_misses']}")

    # Test adaptive batching
    print("\n🧠 Testing Adaptive Batching:")
    print("Running multiple batches to see adaptation...")

    for i in range(3):
        start_time = time.time()
        batcher.predict(sample_texts)
        end_time = time.time()
        print(f"  Batch {i+1}: {(end_time - start_time)*1000:.2f}ms")

    final_stats = batcher.get_performance_stats()
    print(f"\nFinal Performance Summary:")
    print(f"  Total batches processed: {final_stats['total_batches']}")
    print(f"  Total samples processed: {final_stats['total_samples']}")
    print(f"  Average batch size: {final_stats['avg_batch_size']:.2f}")
    print(f"  Overall throughput: {final_stats['throughput']:.2f} samples/sec")

    print("\n✅ Quick start example completed successfully!")
    print("\n💡 Next Steps:")
    print("  1. Try with your own texts and models")
    print("  2. Experiment with different batch sizes")
    print("  3. Monitor performance in your use case")
    print("  4. Check out other examples in the examples/ directory")


if __name__ == "__main__":
    main()
