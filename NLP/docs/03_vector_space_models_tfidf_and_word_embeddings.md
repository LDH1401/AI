# Chương 3: Mô hình Không gian Vector, TF-IDF & Word Embeddings (Vector Space Models)

---

Để máy tính có thể thực hiện toán học trên văn bản, toàn bộ chuỗi ký tự/từ ngữ bắt buộc phải được chuyển đổi thành các **Vector số thực (Numerical Vectors)** trong một không gian vector đa chiều.

---

## 1. Biểu diễn Văn bản Truyền thống (Traditional Text Representations)

### 1.1. One-Hot Encoding
Biểu diễn mỗi từ trong từ điển $V$ bằng một vector thưa (sparse vector) kích thước $|V|$, trong đó vị trí của từ đó bằng 1 và tất cả vị trí khác bằng 0.

- **Nhược điểm nghiêm trọng:**
  - Kích thước vector bùng nổ theo chiều dài từ điển ($|V| \ge 100,000$).
  - Khoảng cách giữa 2 vector bất kỳ luôn vuông góc nhau:
    $$\text{Cosine Similarity}(v_{\text{king}}, v_{\text{queen}}) = 0$$
    Máy tính không nhận biết được sự tương đồng về ngữ nghĩa giữa các từ!

### 1.2. Bag-of-Words (BoW - Túi các từ)
Đếm tần số xuất hiện của từng từ trong câu mà bỏ qua hoàn toàn thứ tự ngữ pháp.

---

## 2. Mô hình TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF đánh giá mức độ quan trọng của một từ $t$ trong một văn bản $d$ nằm trong tập hợp gồm $D$ văn bản (Corpus).

```
   TF-IDF = (Độ phổ biến trong văn bản hiện tại) x (Độ hiếm trên toàn bộ tập dữ liệu)
```

### 2.1. Công thức Toán học:

1. **Term Frequency (TF):** Tần suất xuất hiện của từ $t$ trong văn bản $d$:
   $$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

2. **Inverse Document Frequency (IDF):** Đo độ "hiếm" của từ $t$ trên toàn bộ tập $D$:
   $$\text{IDF}(t, D) = \log \left( \frac{|D|}{|\{d \in D : t \in d\}| + 1} \right)$$

3. **Chỉ số TF-IDF:**
   $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

---

## 3. Vector Từ Tĩnh (Static Word Embeddings)

Word Embeddings biểu diễn mỗi từ dưới dạng một **Vector dày (Dense Vector)** dung lượng nhỏ ($d \in [100, 300]$ dimensions), trong đó các từ có ngữ nghĩa tương đồng sẽ nằm gần nhau trong không gian hình học.

```
       King - Man + Woman ≈ Queen   (Hình học Không gian Vector Embeddings)
```

---

### 3.1. Mô hình Word2Vec (Mikolov et al., Google 2013)

Word2Vec dựa trên giả thuyết phân phối (*"A word is characterized by the company it keeps"*): Các từ xuất hiện trong cùng bối cảnh sẽ có ngữ nghĩa tương tự nhau.

#### 1. CBOW (Continuous Bag-of-Words):
Nhận đầu vào là $2k$ từ ngữ cảnh xung quanh $(w_{t-k}, \dots, w_{t+k})$ để dự đoán **từ trung tâm $w_t$**.

#### 2. Skip-gram:
Nhận đầu vào là **từ trung tâm $w_t$** để dự đoán các **từ ngữ cảnh xung quanh**.

#### 3. Negative Sampling Loss:
Thay vì tính Softmax trên toàn bộ từ điển $V$ (cực kỳ đắt đỏ), Negative Sampling biến bài toán thành phân loại nhị phân: Phân biệt từ ngữ cảnh thật $w_O$ với $K$ từ ngẫu nhiên lấy từ phân phối nhiễu (Negative samples):

$$\mathcal{L}_{\text{NegativeSampling}} = -\log \sigma(v_{w_O}'^T v_{w_I}) - \sum_{i=1}^{K} \mathbb{E}_{w_i \sim P_n(w)} \left[ \log \sigma(-v_{w_i}'^T v_{w_I}) \right]$$

---

### 3.2. Thực thi Code Python cho Word2Vec (PyTorch & Gensim)

#### 1. Mô hình PyTorch CBOW & Skip-Gram (`src/models/word2vec_cbow_skipgram.py`):

```python
import torch
from src.models.word2vec_cbow_skipgram import CBOWModelPyTorch, SkipGramNegativeSamplingPyTorch

vocab_size = 10000
embed_dim = 100

# Khởi tạo mô hình CBOW PyTorch
cbow_model = CBOWModelPyTorch(vocab_size=vocab_size, embedding_dim=embed_dim)

# Khởi tạo mô hình Skip-Gram Negative Sampling PyTorch
sgns_model = SkipGramNegativeSamplingPyTorch(vocab_size=vocab_size, embedding_dim=embed_dim)

print("CBOW Embeddings Shape:", cbow_model.embeddings.weight.shape)
print("SkipGram Target Embeddings Shape:", sgns_model.target_embeddings.weight.shape)
```

#### 2. Huấn luyện Word2Vec với Gensim:

```python
from src.models.word2vec_cbow_skipgram import train_gensim_word2vec

sentences = [
    ["học_sinh", "học", "môn", "toán"],
    ["sinh_học", "là", "môn_học", "tự_nhiên"],
    ["học_sinh", "yêu_thích", "môn", "sinh_học"]
]

# Huấn luyện Word2Vec Skip-gram (sg=1) với Gensim
w2v_model = train_gensim_word2vec(sentences, vector_size=100, sg=1)
if w2v_model:
    vector = w2v_model.wv["sinh_học"]
    similar_words = w2v_model.wv.most_similar("sinh_học")
    print("Vector 'sinh_học' Shape:", vector.shape)
    print("Từ tương đồng nhất với 'sinh_học':", similar_words)
```

---

### 3.3. FastText (Bojanowski et al., Facebook AI 2016)

FastText cải tiến Word2Vec bằng cách biểu diễn một từ bằng **tổng các vector n-gram ký tự (Subword n-grams)** (ví dụ từ `where` với $n=3$: `<wh, whe, her, ere, re>`).
- **Ưu điểm:** Giải quyết triệt để vấn đề từ OOV.
