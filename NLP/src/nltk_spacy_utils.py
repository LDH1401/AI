"""
Module Tiện ích Tích hợp NLTK & spaCy cho Tiền xử lý, POS Tagging, NER, Dependency Parsing, Stemming & Lemmatization.
"""

from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False


class NLTKPipeline:
    """
    Pipeline tiện ích bọc các thuật toán NLTK truyền thống.
    """
    def __init__(self):
        # Tải tự động dữ liệu NLTK nếu chưa có
        for resource in ["punkt", "averaged_perceptron_tagger", "wordnet"]:
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tách văn bản thành danh sách các câu."""
        return sent_tokenize(text)

    def tokenize_words(self, text: str) -> List[str]:
        """Tách câu thành danh sách các từ."""
        return word_tokenize(text)

    def get_pos_tags(self, words: List[str]) -> List[Tuple[str, str]]:
        """Gán nhãn Từ loại (Part-of-Speech Tagging - NN, VB, JJ, RB)."""
        return pos_tag(words)

    def stem_words(self, words: List[str]) -> List[str]:
        """Cắt gốc từ bằng Porter Stemmer (ví dụ: 'running' -> 'run')."""
        return [self.stemmer.stem(w) for w in words]

    def lemmatize_words(self, words: List[str], pos: str = "v") -> List[str]:
        """Biến đổi từ về nguyên mẫu bằng WordNet Lemmatizer."""
        return [self.lemmatizer.lemmatize(w, pos=pos) for w in words]


class SpacyPipeline:
    """
    Pipeline tiện ích bọc spaCy cho Dependency Parsing, NER và Lemmatization.
    """
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None
        if HAS_SPACY:
            try:
                self.nlp = spacy.load(model_name)
                print(f"--> Loaded spaCy model '{model_name}'.")
            except Exception as e:
                print(f"[Warning] Không thể load spaCy model '{model_name}': {e}")

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Phân tích toàn diện văn bản với spaCy: Token, Lemma, POS, Dependency, NER.
        """
        if self.nlp is None:
            return {"error": "spaCy model not loaded."}

        doc = self.nlp(text)
        
        tokens_info = []
        for token in doc:
            tokens_info.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": token.head.text
            })

        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

        return {
            "tokens": tokens_info,
            "entities": entities
        }
