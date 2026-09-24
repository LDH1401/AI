# Natural Language Processing (NLP) Repository & Documentation

Kho tài liệu lý thuyết toàn diện và Codebase chuẩn Production về **Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing - NLP)**, từ Nền tảng NLTK, spaCy, Tokenization (BPE, WordPiece, Unigram), Vector Space Models (Word2Vec CBOW/Skip-gram), Kiến trúc Transformer Deep-Dive, Encoder Models (BERT/PhoBERT), Decoder Models (GPT/LLMs), Prompt Engineering, Fine-tuning (LoRA, QLoRA 4-bit), Alignment (RLHF, DPO), Document QA & RAG, **MoE (DeepSeek-V3, Mixtral)**, **State Space Models (Mamba-2)** đến **Reasoning LLMs (DeepSeek-R1, GRPO Loss, LLM Agents)**.

---

## 📚 Chuỗi Tài liệu Lý thuyết Toàn diện (13 Chương)

1. [Chương 1: Nền tảng NLP & Xử lý Chuỗi Văn bản (NLTK & spaCy)](docs/01_nlp_fundamentals.md)
2. [Chương 2: Tokenization & Tách từ Nâng cao (BPE, WordPiece, Unigram)](docs/02_tokenization_bpe_wordpiece.md)
3. [Chương 3: Vector Biểu diễn Từ (Word2Vec CBOW & Skip-gram)](docs/03_word_embeddings_word2vec.md)
4. [Chương 4: Kiến trúc Transformer Deep-Dive (Self-Attention & Positional Encodings)](docs/04_transformer_architecture.md)
5. [Chương 5: Mô hình Pretrained Encoder (BERT & PhoBERT)](docs/05_bert_phobert_encoder.md)
6. [Chương 6: Mô hình Causal Decoder & Generative LLMs (GPT Series)](docs/06_gpt_decoder_llm.md)
7. [Chương 7: Kỹ thuật Prompt Engineering & Retrieval-Augmented Generation (RAG)](docs/07_rag_retrieval_augmented_generation.md)
8. [Chương 8: Tinh chỉnh Mô hình Tham số Hiệu quả (PEFT: LoRA & QLoRA 4-bit)](docs/08_peft_lora_qlora.md)
9. [Chương 9: Căn chỉnh Mô hình theo Ý muốn Con người (RLHF & DPO)](docs/09_rlhf_and_dpo_alignment.md)
10. [Chương 10: Nền tảng Toán học: Lý thuyết Thông tin & Ước lượng Hợp lý Cực đại (MLE)](docs/10_information_theory_and_mle.md)
11. [Chương 11: Hỏi Đáp trên Tài liệu (Document QA & Document RAG)](docs/11_document_question_answering.md)
12. [Chương 12: Kiến trúc LLM Hiện đại: Mixture of Experts (MoE), DeepSeek-V3 & Mamba](docs/12_modern_llm_architectures_moe_ssm_mamba.md)
13. [Chương 13: Mô hình LLM Suy luận (Reasoning LLMs: DeepSeek-R1, GRPO & LLM Agents)](docs/13_reasoning_llms_grpo_deepseek_r1_and_llm_agents.md)

---

## 🛠️ Cấu trúc Kho lưu trữ (Project Repository Structure)

```
g:/Speech Processing/NLP/
├── docs/ (13 chương lý thuyết NLP, MoE, Mamba, DeepSeek-R1 & GRPO)
├── src/
│   ├── models/
│   │   ├── bpe_tokenizer.py
│   │   ├── word2vec_cbow.py
│   │   ├── transformer_block.py
│   │   ├── phobert_classifier.py
│   │   ├── rag_pipeline.py
│   │   ├── lora_linear.py
│   │   ├── document_qa_rag.py
│   │   └── deepseek_grpo_trainer.py # GRPO Loss Trainer cho LLM Reasoning
```
