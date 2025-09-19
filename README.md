# 🚀 TurboBatch for Transformers

<div align="center">

<img src="https://img.shields.io/github/stars/Shayanthn/turbobatch?style=for-the-badge&logo=github&color=gold" alt="GitHub Stars"/>
<img src="https://img.shields.io/pypi/v/turbobatch?style=for-the-badge&logo=pypi&color=blue" alt="PyPI Version"/>
<img src="https://img.shields.io/pypi/dm/turbobatch?style=for-the-badge&logo=download&color=green" alt="Downloads"/>
<img src="https://img.shields.io/github/license/Shayanthn/turbobatch?style=for-the-badge&color=purple" alt="License"/>

**⚡ کتابخانه‌ای برای تسریع ۱۰ برابری inference مدل‌های transformer با batching هوشمند**

**⚡ 10x Faster Transformer Inference with Intelligent Dynamic Batching**

[🇮🇷 فارسی](#-فارسی) | [🇺🇸 English](#-english) | [📖 Documentation](https://shayantaherkhani.ir) | [🎯 Examples](examples/) | [💬 Discussion](https://github.com/Shayanthn/turbobatch/discussions)

</div>

---

## 🇮🇷 فارسی

<div align="right" dir="rtl">

### 🔥 چرا DynamicBatcher؟

آیا تا به حال در پروژه‌های NLP خود با **کندی inference مدل‌های transformer** مواجه شده‌اید؟ آیا مدل‌تان بر روی هزاران متن باید اجرا شود اما ساعت‌ها طول می‌کشد؟

**DynamicBatcher** راه‌حل شما است! 🎯

```bash
قبل از DynamicBatcher: 100 متن → 45 ثانیه ⏰
بعد از DynamicBatcher:  100 متن → 4.5 ثانیه ⚡
```

### ✨ ویژگی‌های فوق‌العاده

<table dir="rtl">
<tr>
<td align="center">🚀<br><strong>تسریع ۱۰ برابری</strong><br>inference سریع‌تر با batching هوشمند</td>
<td align="center">🧠<br><strong>تطبیق خودکار</strong><br>اندازه batch بر اساس workload</td>
<td align="center">💾<br><strong>مدیریت حافظه</strong><br>استفاده بهینه از GPU</td>
</tr>
<tr>
<td align="center">📊<br><strong>مانیتورینگ</strong><br>آمار عملکرد real-time</td>
<td align="center">🔧<br><strong>آسان</strong><br>یکپارچگی با HuggingFace</td>
<td align="center">🔄<br><strong>کش هوشمند</strong><br>ذخیره‌سازی نتایج تکراری</td>
</tr>
</table>

### 🚀 نصب سریع

```bash
pip install turbobatch
```

یا از سورس:

```bash
git clone https://github.com/Shayanthn/turbobatch.git
cd turbobatch
pip install -e .
```

### 💻 مثال سریع - تحلیل احساسات

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from turbobatch import TurboBatcher

# بارگذاری مدل
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# ایجاد TurboBatcher
batcher = TurboBatcher(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=32,
    adaptive_batching=True
)

# متن‌های نمونه
texts = [
    "این محصول فوق‌العاده است!",
    "تجربه بدی بود.",
    "کیفیت خوبی دارد و قیمت مناسب.",
    "خیلی راضی هستم از خرید!"
]

# پیش‌بینی سریع
results = batcher.predict(texts)

for text, result in zip(texts, results):
    sentiment = "مثبت" if result.label == 1 else "منفی"
    print(f"متن: {text}")
    print(f"احساس: {sentiment} (اطمینان: {result.score:.2f})")
    print("-" * 50)
```

### 📈 مقایسه عملکرد

| روش | زمان اجرا | سرعت | مصرف حافظه |
|-----|-----------|-------|-------------|
| **DynamicBatcher** | ⚡ **4.5s** | 🚀 **222 sample/sec** | 💾 **کم** |
| Batch سنتی | 🐌 12.3s | 📉 81 sample/sec | 💾 زیاد |
| تک به تک | 🐢 45.2s | 📉 22 sample/sec | 💾 متوسط |

### 🎯 مثال‌های پیشرفته

#### 1️⃣ API تحلیل احساسات

```python
class SentimentAPI:
    def __init__(self):
        self.batcher = DynamicBatcher(model, tokenizer, max_batch_size=64)
    
    def analyze(self, texts):
        return self.batcher.predict(texts)

api = SentimentAPI()
results = api.analyze(["متن اول", "متن دوم", "متن سوم"])
```

#### 2️⃣ پردازش فایل CSV

```python
import pandas as pd

# خواندن فایل CSV
df = pd.read_csv("reviews.csv")
texts = df['review_text'].tolist()

# پردازش سریع
results = batcher.predict(texts)

# اضافه کردن نتایج به DataFrame
df['sentiment'] = [r.label for r in results]
df['confidence'] = [r.score for r in results]
```

#### 3️⃣ مانیتورینگ عملکرد

```python
# آمار عملکرد
stats = batcher.get_performance_stats()
print(f"📊 تعداد کل batch: {stats['total_batches']}")
print(f"🚀 سرعت: {stats['throughput']:.2f} sample/sec")
print(f"💾 نرخ کش: {stats['cache_hit_rate']:.1f}%")
```

### 🔧 تنظیمات پیشرفته

```python
batcher = DynamicBatcher(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=32,          # حداکثر اندازه batch
    timeout_ms=100,             # timeout برای تشکیل batch
    adaptive_batching=True,     # تطبیق خودکار اندازه batch
    performance_monitoring=True, # مانیتورینگ عملکرد
    enable_caching=True,        # فعال‌سازی کش
    device="cuda"               # استفاده از GPU
)
```

### 🎮 دمو تعاملی

برای تست سریع:

```bash
python examples/sentiment_analysis_demo.py
```

برای benchmark کامل:

```bash
python examples/advanced_benchmarking_demo.py
```

### 🤝 مشارکت

آیا مایل به مشارکت هستید؟ 

1. **Fork** کنید
2. شاخه جدید بسازید: `git checkout -b feature/amazing-feature`
3. تغییرات را commit کنید: `git commit -m 'Add amazing feature'`
4. Push کنید: `git push origin feature/amazing-feature`
5. **Pull Request** بسازید

### 👨‍� سازنده

**شایان طاهرخانی**
- 🌐 وبسایت: [shayantaherkhani.ir](https://shayantaherkhani.ir)
- 💼 LinkedIn: [linkedin.com/in/shayantaherkhani78](https://linkedin.com/in/shayantaherkhani78)
- 🎓 ایمیل دانشگاهی: shayan.taherkhani@studio.unibo.it
- 📧 ایمیل شخصی: shayanthn78@gmail.com

### ⭐ حمایت از پروژه

اگر این پروژه برایتان مفید بود:

- ⭐ **ستاره** بزنید
- 🍴 **Fork** کنید  
- 📢 با دوستان **به اشتراک** بگذارید
- 🐛 **باگ** پیدا کردید؟ گزارش دهید

### 📜 مجوز

این پروژه تحت مجوز MIT منتشر شده است. جزئیات در فایل [LICENSE](LICENSE).

</div>

---

## 🇺🇸 English

### 🔥 Why DynamicBatcher?

Ever struggled with **slow transformer inference** in your NLP projects? Tired of waiting hours for your model to process thousands of texts?

**DynamicBatcher** is your solution! 🎯

```bash
Before DynamicBatcher: 100 texts → 45 seconds ⏰
After DynamicBatcher:  100 texts → 4.5 seconds ⚡
```

### ✨ Incredible Features

<table>
<tr>
<td align="center">🚀<br><strong>10x Faster</strong><br>Lightning-fast inference with smart batching</td>
<td align="center">🧠<br><strong>Adaptive</strong><br>Auto-adjusts batch size based on workload</td>
<td align="center">💾<br><strong>Memory Efficient</strong><br>Optimal GPU utilization</td>
</tr>
<tr>
<td align="center">📊<br><strong>Monitoring</strong><br>Real-time performance stats</td>
<td align="center">🔧<br><strong>Easy Integration</strong><br>Seamless HuggingFace compatibility</td>
<td align="center">🔄<br><strong>Smart Caching</strong><br>Automatic result caching</td>
</tr>
</table>

### � Quick Installation

```bash
pip install turbobatch
```

Or from source:

```bash
git clone https://github.com/Shayanthn/turbobatch.git
cd turbobatch
pip install -e .
```

### � Quick Example - Sentiment Analysis

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from DynamicBatcher import DynamicBatcher

# Load model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Create DynamicBatcher
batcher = DynamicBatcher(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=32,
    adaptive_batching=True
)

# Sample texts
texts = [
    "I absolutely love this product!",
    "This was a terrible experience.",
    "Good quality and reasonable price.",
    "Highly satisfied with my purchase!"
]

# Fast prediction
results = batcher.predict(texts)

for text, result in zip(texts, results):
    sentiment = "Positive" if result.label == 1 else "Negative"
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (Confidence: {result.score:.2f})")
    print("-" * 50)
```
    ### 📈 Performance Comparison

| Method | Processing Time | Throughput | Memory Usage |
|--------|----------------|------------|--------------|
| **DynamicBatcher** | ⚡ **4.5s** | 🚀 **222 samples/sec** | 💾 **Low** |
| Traditional Batch | 🐌 12.3s | 📉 81 samples/sec | 💾 High |
| Individual | 🐢 45.2s | 📉 22 samples/sec | 💾 Medium |

### 🎯 Advanced Examples

#### 1️⃣ Sentiment Analysis API

```python
class SentimentAPI:
    def __init__(self):
        self.batcher = DynamicBatcher(model, tokenizer, max_batch_size=64)
    
    def analyze(self, texts):
        return self.batcher.predict(texts)

api = SentimentAPI()
results = api.analyze(["First text", "Second text", "Third text"])
```

#### 2️⃣ CSV File Processing

```python
import pandas as pd

# Read CSV file
df = pd.read_csv("reviews.csv")
texts = df['review_text'].tolist()

# Fast processing
results = batcher.predict(texts)

# Add results to DataFrame
df['sentiment'] = [r.label for r in results]
df['confidence'] = [r.score for r in results]
```

#### 3️⃣ Performance Monitoring

```python
# Performance statistics
stats = batcher.get_performance_stats()
print(f"📊 Total batches: {stats['total_batches']}")
print(f"🚀 Throughput: {stats['throughput']:.2f} samples/sec")
print(f"💾 Cache hit rate: {stats['cache_hit_rate']:.1f}%")
```

### 🔧 Advanced Configuration

```python
batcher = DynamicBatcher(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=32,          # Maximum batch size
    timeout_ms=100,             # Batch formation timeout
    adaptive_batching=True,     # Auto-adjust batch size
    performance_monitoring=True, # Enable performance monitoring
    enable_caching=True,        # Enable result caching
    device="cuda"               # Use GPU
)
```

### 🎮 Interactive Demo

For quick testing:

```bash
python examples/sentiment_analysis_demo.py
```

For comprehensive benchmarking:

```bash
python examples/advanced_benchmarking_demo.py
```

For Jupyter notebook tutorial:

```bash
jupyter notebook examples/DynamicBatcher_Tutorial.ipynb
```

### 🤝 Contributing

Want to contribute? Amazing! 

1. **Fork** the repo
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open **Pull Request**

### 👨‍💻 Author

**Shayan Taherkhani**
- 🌐 Website: [shayantaherkhani.ir](https://shayantaherkhani.ir)
- 💼 LinkedIn: [linkedin.com/in/shayantaherkhani78](https://linkedin.com/in/shayantaherkhani78)
- 🎓 Academic Email: shayan.taherkhani@studio.unibo.it
- 📧 Personal Email: shayanthn78@gmail.com

### ⭐ Support the Project

If you found this project helpful:

- ⭐ **Star** this repo
- 🍴 **Fork** it
- 📢 **Share** with friends
- 🐛 **Report** bugs
- 💡 **Suggest** features

### 📊 Project Stats

<div align="center">

![GitHub contributors](https://img.shields.io/github/contributors/Shayanthn/turbobatch?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/Shayanthn/turbobatch?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues/Shayanthn/turbobatch?style=for-the-badge)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Shayanthn/turbobatch?style=for-the-badge)

</div>

### 💰 Commercial Use & Monetization

DynamicBatcher can be monetized through:

- 🏢 **Enterprise Consulting**: Offer optimization services for large-scale NLP deployments
- ☁️ **SaaS Solutions**: Build high-performance NLP APIs with faster inference
- 🎓 **Training & Workshops**: Teach high-performance NLP techniques
- 📊 **Custom Solutions**: Develop tailored batching strategies for specific use cases
- 💼 **Performance Auditing**: Optimize existing NLP pipelines for enterprises

### 🔒 Security & Licensing

- ✅ **MIT Licensed**: Free for commercial and personal use
- 🔐 **No data collection**: Your data stays private
- 🛡️ **Enterprise ready**: Suitable for production environments
- 📝 **Well documented**: Comprehensive documentation and examples

### 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### 🙏 Acknowledgments

- HuggingFace team for the amazing transformers library
- PyTorch team for the incredible framework
- Open source community for inspiration and support

---

<div align="center">

**Made with ❤️ by [Shayan Taherkhani](https://shayantaherkhani.ir)**

*If you use this in your research, please consider citing:*

```bibtex
@software{taherkhani2025dynamicbatcher,
  author = {Taherkhani, Shayan},
  title = {DynamicBatcher: High-Performance Dynamic Batching for Transformer Models},
  year = {2025},
  url = {https://github.com/Shayanthn/turbobatch}
}
```

**⭐ Star this repo if it helped you! ⭐**

</div>
```
📊 Performance Benchmarks :
Method	        Batch Size	 Avg Inference Time	    Speedup
Naive Batching	    32	           4.72s	          1x
DynamicBatcher	    32	           1.89s	         2.5x
Naive Batching	    64	           8.91s	          1x
DynamicBatcher	    64	    
*Benchmarks performed on NVIDIA V100 with 5000 variable-length sequences (5-100 words)*
🌟 Advanced Features
Custom Collate Functions
```bash
def custom_collate(batch):
    # Your custom processing
    return processed_batch

batcher = DynamicBatcher(tokenizer, collate_fn=custom_collate)
```
Mixed Precision Support
```bash
batcher = DynamicBatcher(tokenizer, fp16=True)  # Enable AMP
```
Progress Tracking
```bash
batches = batcher.create_batches(texts, progress_bar=True)
```
🧩 Integration Guide
With PyTorch DataLoader
```bash
from torch.utils.data import DataLoader

class TextDataset:
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

dataset = TextDataset(texts)
dataloader = DataLoader(
    dataset,
    batch_sampler=DynamicBatchSampler(dataset, tokenizer, batch_size=32),
    collate_fn=batcher.dynamic_collate
)
```
With FastAPI Web Service
```bash
from fastapi import FastAPI
app = FastAPI()
batcher = DynamicBatcher(tokenizer)

@app.post("/predict")
async def predict(texts: List[str]):
    batches = batcher.create_batches(texts)
    results = []
    for batch in batches:
        outputs = model(**batch[0])
        results.extend(process_outputs(outputs))
    return {"predictions": results}
```
📚 Documentation
DynamicBatcher Class
```bash
DynamicBatcher(
    tokenizer: AutoTokenizer,
    max_sequence_length: int = 512,
    fp16: bool = False,
    progress_bar: bool = False,
    sorting_strategy: str = 'ascending'  # or 'descending'
)
```
🎯 Use Cases:

    🔍 Document Processing Pipelines
    💬 Real-time Chat Applications
    📰 News Article Classification
    🗣 Speech-to-Text Post Processing
    🌍 Multilingual Translation Services
## 📬 Contact
**Shayan Taherkhani**  
📧 [shayanthn78@gmail.com](mailto:shayanthn78@gmail.com)  
💼 [LinkedIn](https://linkedin.com/in/shayantaherkhani)  
🐙 [GitHub](https://github.com/shayanthn)

