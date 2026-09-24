# Implementation Plan - Comprehensive NLP Repository & Framework

Xây dựng kho tài liệu và codebase chuẩn Production cho **Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing - NLP)** toàn diện từ Nền tảng Xử lý Văn bản, TF-IDF, Word Embeddings (Word2Vec, GloVe), RNN/LSTM, Transformer Architecture, BERT/PhoBERT, GPT/LLM, RAG, Fine-tuning LoRA/QLoRA và liên kết sâu sắc với **Lý thuyết Thông tin & MLE (Perplexity, Cross-Entropy, NLL)**.

## Cấu trúc thư mục dự kiến
```
g:/Speech Processing/NLP/
├── .gemini/
│   ├── implementation_plan.md
│   └── todo.md
├── docs/
│   ├── 01_nlp_fundamentals.md
│   ├── 02_text_preprocessing_and_tokenization.md
│   ├── 03_vector_space_models_tfidf_and_word_embeddings.md
│   ├── 04_sequential_models_rnn_lstm_gru.md
│   ├── 05_transformer_architecture_deep_dive.md
│   ├── 06_bert_and_encoder_models.md
│   ├── 07_gpt_and_decoder_autoregressive_models.md
│   ├── 08_core_nlp_tasks.md
│   ├── 09_rag_and_llm_fine_tuning.md
│   └── 10_information_theory_mle_in_nlp.md
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_classifier.py
│   │   ├── bert_qa.py
│   │   └── transformer_seq2seq.py
│   ├── __init__.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── utils.py
│   ├── train.py
│   └── infer.py
├── requirements.txt
└── README.md
```
