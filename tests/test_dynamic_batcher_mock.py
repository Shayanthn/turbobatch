#!/usr/bin/env python3
"""
🧪 Unit Tests for DynamicBatcher - Lightweight Version

Comprehensive test suite for the DynamicBatcher library ensuring reliability,
performance, and correctness across different scenarios using mock models.

Author: Shayan Taherkhani
"""

import os
import sys
import unittest
from typing import List
from unittest.mock import MagicMock, Mock

import torch
import torch.nn as nn

# Add parent directory to path to import TurboBatch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turbobatch import BatchStats, DynamicBatcher, PredictionResult


class MockModel(nn.Module):
    """Mock transformer model for testing."""

    def __init__(self, vocab_size=30522, hidden_size=768, num_labels=2):
        super().__init__()
        self.config = Mock()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.num_labels = num_labels
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Simple forward pass for testing
        embeddings = self.embedding(input_ids)
        # Mean pooling
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)
            pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = embeddings.mean(dim=1)

        logits = self.classifier(pooled)

        # Return object with logits attribute
        output = Mock()
        output.logits = logits
        return output


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=30522):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102

    def encode_plus(
        self,
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        **kwargs
    ):
        # Simple tokenization - just convert to numbers based on text length
        base_length = min(len(text.split()) + 2, max_length)  # +2 for CLS and SEP

        if add_special_tokens:
            input_ids = (
                [self.cls_token_id]
                + list(range(1, base_length - 1))
                + [self.sep_token_id]
            )
        else:
            input_ids = list(range(1, base_length))

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
        **kwargs
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


class TestDynamicBatcher(unittest.TestCase):
    """Test suite for DynamicBatcher functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        cls.model = MockModel()
        cls.tokenizer = MockTokenizer()

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

        # Cache should be empty (test indirectly)
        stats = self.batcher.get_performance_stats()
        # Reset stats to test fresh state
        self.batcher.reset_stats()

        # New prediction after clearing cache
        self.batcher.predict("Test text")
        new_stats = self.batcher.get_performance_stats()
        self.assertEqual(new_stats["cache_hits"], 0)

    def test_adaptive_batching(self):
        """Test adaptive batching functionality."""
        adaptive_batcher = DynamicBatcher(
            model=self.model,
            tokenizer=self.tokenizer,
            max_batch_size=8,
            adaptive_batching=True,
        )

        # Run multiple predictions to trigger adaptation
        for _ in range(5):
            adaptive_batcher.predict(self.sample_texts)

        # Check that it's working (no errors)
        self.assertTrue(adaptive_batcher.adaptive_batching)

    def test_device_handling(self):
        """Test device handling."""
        # Test CPU device
        cpu_batcher = DynamicBatcher(
            model=self.model, tokenizer=self.tokenizer, device="cpu"
        )
        self.assertEqual(str(cpu_batcher.device), "cpu")

    def test_stats_reset(self):
        """Test statistics reset functionality."""
        # Run some predictions
        self.batcher.predict(self.sample_texts)

        # Get initial stats
        stats_before = self.batcher.get_performance_stats()
        self.assertGreater(stats_before["total_batches"], 0)

        # Reset stats
        self.batcher.reset_stats()

        # Check stats are reset
        stats_after = self.batcher.get_performance_stats()
        self.assertEqual(stats_after["total_batches"], 0)
        self.assertEqual(stats_after["total_samples"], 0)


class TestBatchStats(unittest.TestCase):
    """Test suite for BatchStats functionality."""

    def test_batch_stats_initialization(self):
        """Test BatchStats initialization."""
        stats = BatchStats()
        self.assertEqual(stats.total_batches, 0)
        self.assertEqual(stats.total_samples, 0)
        self.assertEqual(stats.total_processing_time, 0.0)
        self.assertEqual(stats.cache_hits, 0)
        self.assertEqual(stats.cache_misses, 0)

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = BatchStats()
        stats.cache_hits = 7
        stats.cache_misses = 3
        self.assertEqual(stats.cache_hit_rate, 70.0)

        # Test zero case
        empty_stats = BatchStats()
        self.assertEqual(empty_stats.cache_hit_rate, 0.0)

    def test_throughput(self):
        """Test throughput calculation."""
        stats = BatchStats()
        stats.total_samples = 100
        stats.total_processing_time = 10.0
        self.assertEqual(stats.throughput, 10.0)

        # Test zero case
        empty_stats = BatchStats()
        self.assertEqual(empty_stats.throughput, 0.0)


class TestPredictionResult(unittest.TestCase):
    """Test suite for PredictionResult functionality."""

    def test_prediction_result_creation(self):
        """Test PredictionResult creation."""
        result = PredictionResult(label=1, score=0.95)
        self.assertEqual(result.label, 1)
        self.assertEqual(result.score, 0.95)
        self.assertIsNone(result.logits)
        self.assertIsNone(result.processing_time)

        # Test with optional parameters
        logits = torch.tensor([0.1, 0.9])
        result_with_extras = PredictionResult(
            label=1, score=0.95, logits=logits, processing_time=0.001
        )
        self.assertTrue(torch.equal(result_with_extras.logits, logits))
        self.assertEqual(result_with_extras.processing_time, 0.001)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
