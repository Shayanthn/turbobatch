#!/usr/bin/env python3
"""
🧪 Unit Tests for DynamicBatcher

Comprehensive test suite for the DynamicBatcher library ensuring reliability,
performance, and correctness across different scenarios.

Author: Shayan Taherkhani
"""

import os
import sys
import tempfile
import unittest
from typing import List
from unittest.mock import Mock, patch

import torch

# Add parent directory to path to import TurboBatch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from turbobatch import BatchStats, DynamicBatcher, PredictionResult


class TestDynamicBatcher(unittest.TestCase):
    """Test suite for DynamicBatcher functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        # Use a small model for faster testing
        cls.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForSequenceClassification.from_pretrained(cls.model_name)

        cls.sample_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Amazing quality and fast delivery!",
            "Poor customer service experience.",
        ]

    def setUp(self):
        """Set up test fixtures before each test."""
        self.batcher = DynamicBatcher(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=4,
            max_sequence_length=128,
            timeout_ms=50,
            adaptive_batching=True,
            performance_monitoring=True,
        )

    def test_initialization(self):
        """Test DynamicBatcher initialization with various parameters."""
        # Test default initialization
        batcher = DynamicBatcher(self.model, self.tokenizer)
        self.assertEqual(batcher.max_batch_size, 32)
        self.assertEqual(batcher.max_sequence_length, 512)
        self.assertTrue(batcher.adaptive_batching)
        self.assertTrue(batcher.performance_monitoring)

        # Test custom parameters
        custom_batcher = DynamicBatcher(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=16,
            max_sequence_length=256,
            adaptive_batching=False,
            performance_monitoring=False,
        )
        self.assertEqual(custom_batcher.max_batch_size, 16)
        self.assertEqual(custom_batcher.max_sequence_length, 256)
        self.assertFalse(custom_batcher.adaptive_batching)
        self.assertFalse(custom_batcher.performance_monitoring)

    def test_initialization_errors(self):
        """Test initialization with invalid parameters."""
        # Invalid model type
        with self.assertRaises(TypeError):
            DynamicBatcher("invalid_model", self.tokenizer)

        # Invalid tokenizer type
        with self.assertRaises(TypeError):
            DynamicBatcher(self.model, "invalid_tokenizer")

        # Invalid batch size
        with self.assertRaises(ValueError):
            DynamicBatcher(self.model, self.tokenizer, max_batch_size=0)

        # Invalid sequence length
        with self.assertRaises(ValueError):
            DynamicBatcher(self.model, self.tokenizer, max_sequence_length=-1)

    def test_single_prediction(self):
        """Test single text prediction."""
        text = "I love this amazing product!"
        result = self.batcher.predict(text)

        self.assertIsInstance(result, PredictionResult)
        self.assertIn(result.label, [0, 1])  # Binary classification
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)

    def test_batch_prediction(self):
        """Test batch prediction with multiple texts."""
        results = self.batcher.predict(self.sample_texts)

        self.assertEqual(len(results), len(self.sample_texts))
        for result in results:
            self.assertIsInstance(result, PredictionResult)
            self.assertIn(result.label, [0, 1])
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_empty_input(self):
        """Test handling of empty input."""
        result = self.batcher.predict([])
        self.assertEqual(result, [])

    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        with self.assertRaises(TypeError):
            self.batcher.create_batches("not_a_list")

        with self.assertRaises(TypeError):
            self.batcher.create_batches([123, 456])  # Not strings

    def test_create_batches(self):
        """Test batch creation functionality."""
        batches = self.batcher.create_batches(self.sample_texts, batch_size=2)

        # Should create 3 batches (2, 2, 1 texts)
        self.assertEqual(len(batches), 3)

        # Check batch structure
        for batch_encoded, original_indices in batches:
            self.assertIn("input_ids", batch_encoded)
            self.assertIn("attention_mask", batch_encoded)
            self.assertIsInstance(original_indices, list)
            self.assertLessEqual(len(original_indices), 2)

    def test_sort_and_index_texts(self):
        """Test text sorting and indexing functionality."""
        texts = [
            "Short",
            "This is a much longer text with more words",
            "Medium length text",
        ]
        sorted_texts = self.batcher._sort_and_index_texts(texts)

        # Should be sorted by token length (ascending)
        lengths = [item[0] for item in sorted_texts]
        self.assertEqual(lengths, sorted(lengths))

        # Check original indices are preserved
        for length, text, original_idx in sorted_texts:
            self.assertEqual(text, texts[original_idx])

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        # Run some predictions
        self.batcher.predict(self.sample_texts)

        stats = self.batcher.get_performance_stats()

        self.assertIn("total_batches", stats)
        self.assertIn("total_samples", stats)
        self.assertIn("avg_batch_size", stats)
        self.assertIn("throughput", stats)
        self.assertIn("cache_hit_rate", stats)

        self.assertGreater(stats["total_batches"], 0)
        self.assertGreater(stats["total_samples"], 0)

    def test_caching(self):
        """Test caching functionality."""
        # First prediction
        text = "I love this product!"
        result1 = self.batcher.predict(text)

        # Second prediction (should use cache)
        result2 = self.batcher.predict(text)

        # Results should be identical
        self.assertEqual(result1.label, result2.label)
        self.assertEqual(result1.score, result2.score)

        # Check cache statistics
        stats = self.batcher.get_performance_stats()
        self.assertGreater(stats["cache_hits"], 0)

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Make a prediction to populate cache
        self.batcher.predict("Test text")

        # Clear cache
        self.batcher.clear_cache()

        # Cache should be empty (this is internal, so we test indirectly)
        # by checking that subsequent predictions don't have cache hits
        self.batcher.reset_stats()
        self.batcher.predict("Test text")
        stats = self.batcher.get_performance_stats()
        self.assertEqual(stats["cache_hits"], 0)

    def test_adaptive_batching(self):
        """Test adaptive batching functionality."""
        adaptive_batcher = DynamicBatcher(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=8,
            adaptive_batching=True,
        )

        # Run multiple predictions to trigger adaptation
        for _ in range(3):
            adaptive_batcher.predict(self.sample_texts)

        # Check that adaptation metrics are being tracked
        self.assertTrue(hasattr(adaptive_batcher, "_recent_batch_sizes"))
        self.assertTrue(hasattr(adaptive_batcher, "_recent_processing_times"))

    def test_device_handling(self):
        """Test device handling for CPU and GPU."""
        # Test explicit CPU device
        cpu_batcher = DynamicBatcher(
            model=self.model, tokenizer=self.tokenizer, device="cpu"
        )
        self.assertEqual(cpu_batcher.device, torch.device("cpu"))

        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_batcher = DynamicBatcher(
                model=self.model, tokenizer=self.tokenizer, device="cuda"
            )
            self.assertTrue(cuda_batcher.device.type == "cuda")

    def test_return_logits(self):
        """Test returning raw logits in predictions."""
        result = self.batcher.predict("I love this!", return_logits=True)

        self.assertIsNotNone(result.logits)
        self.assertIsInstance(result.logits, torch.Tensor)

    def test_performance_stats_reset(self):
        """Test resetting performance statistics."""
        # Generate some statistics
        self.batcher.predict(self.sample_texts)

        # Reset stats
        self.batcher.reset_stats()

        stats = self.batcher.get_performance_stats()
        self.assertEqual(stats["total_batches"], 0)
        self.assertEqual(stats["total_samples"], 0)

    def test_string_representation(self):
        """Test string representation of DynamicBatcher."""
        repr_str = repr(self.batcher)
        self.assertIn("DynamicBatcher", repr_str)
        self.assertIn("max_batch_size", repr_str)
        self.assertIn("device", repr_str)

    def test_large_batch_processing(self):
        """Test processing large batches."""
        # Create a larger dataset
        large_texts = self.sample_texts * 20  # 100 texts

        results = self.batcher.predict(large_texts)

        self.assertEqual(len(results), len(large_texts))
        for result in results:
            self.assertIsInstance(result, PredictionResult)

    def test_variable_length_texts(self):
        """Test processing texts of varying lengths."""
        variable_texts = [
            "Short",
            "This is a medium length text with several words",
            "This is a very long text that contains many words and should test the batching functionality properly by ensuring that sequences of different lengths are handled correctly",
            "Another short one",
            "Medium length text here with some words",
        ]

        results = self.batcher.predict(variable_texts)

        self.assertEqual(len(results), len(variable_texts))
        for result in results:
            self.assertIsInstance(result, PredictionResult)

    def test_batch_stats_dataclass(self):
        """Test BatchStats dataclass functionality."""
        stats = BatchStats()

        # Test initial values
        self.assertEqual(stats.total_batches, 0)
        self.assertEqual(stats.cache_hit_rate, 0.0)
        self.assertEqual(stats.throughput, 0.0)

        # Test with some data
        stats.cache_hits = 10
        stats.cache_misses = 5
        stats.total_samples = 100
        stats.total_processing_time = 10.0

        self.assertAlmostEqual(stats.cache_hit_rate, 66.67, places=1)
        self.assertEqual(stats.throughput, 10.0)

    def test_prediction_result_dataclass(self):
        """Test PredictionResult dataclass functionality."""
        result = PredictionResult(label=1, score=0.95)

        self.assertEqual(result.label, 1)
        self.assertEqual(result.score, 0.95)
        self.assertIsNone(result.logits)
        self.assertIsNone(result.processing_time)


class TestBenchmarking(unittest.TestCase):
    """Test benchmarking functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = AutoModelForSequenceClassification.from_pretrained(cls.model_name)

        cls.test_texts = [
            "Great product!",
            "Terrible experience.",
            "It's okay.",
            "Amazing quality!",
            "Poor service.",
        ]

    def test_benchmark_batcher(self):
        """Test the benchmark_batcher function."""
        from DynamicBatcher import benchmark_batcher

        results = benchmark_batcher(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=self.test_texts,
            batch_sizes=[1, 2],
            num_runs=1,
        )

        self.assertIn("batch_sizes", results)
        self.assertIn("dynamic_batcher_times", results)
        self.assertIn("throughputs", results)
        self.assertIn("optimal_batch_size", results)

        self.assertEqual(len(results["batch_sizes"]), 2)
        self.assertEqual(len(results["dynamic_batcher_times"]), 2)
        self.assertEqual(len(results["throughputs"]), 2)


if __name__ == "__main__":
    # Set up test environment
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    print("🧪 Running DynamicBatcher Unit Tests")
    print("=" * 50)

    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"💥 {len(result.errors)} error(s) occurred")

    print(f"🧪 Total tests run: {result.testsRun}")
    print("=" * 50)
