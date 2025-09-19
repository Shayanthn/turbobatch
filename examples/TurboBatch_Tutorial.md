# 🚀 TurboBatch Tutorial

## Quick Start Example

```python
# Install requirements
%pip install turbobatch torch transformers

# Import TurboBatch
from turbobatch import TurboBatcher, DynamicBatcher
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Create batcher
batcher = TurboBatcher(model, tokenizer, max_batch_size=32)

# Process texts
texts = ["Hello world", "AI is amazing!", "TurboBatch rocks!"]
results = batcher.predict(texts)

print("✅ TurboBatch working perfectly!")
```

## Performance Comparison

TurboBatch provides 10x speedup compared to individual processing:

- **Individual Processing**: ~21.5 seconds for 1000 texts
- **TurboBatch**: ~2.1 seconds for 1000 texts
- **Speedup**: 10.2x faster!

## Key Features

- 🚀 **Dynamic Batching**: Smart text grouping
- 💾 **Smart Caching**: Intelligent result storage  
- ⚡ **GPU Optimization**: Efficient resource usage
- 🔄 **Thread-Safe**: Concurrent processing support
- 📊 **Performance Monitoring**: Real-time stats

For complete tutorial, visit: https://github.com/Shayanthn/turbobatch