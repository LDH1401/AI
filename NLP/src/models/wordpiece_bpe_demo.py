"""
Thuật toán Tách từ phụ (Subword Tokenization) từ đầu:
1. Byte-Pair Encoding (BPE).
2. WordPiece Tokenization (với tiền tố ##).
"""

import collections
from typing import Dict, List, Tuple, Set


class BPETokenizerFromScratch:
    """
    Triển khai Thuật toán Byte-Pair Encoding (BPE) từ đầu trong Python.
    """
    def __init__(self, num_merges: int = 10):
        self.num_merges = num_merges
        self.merges = {}
        self.vocab = set()

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Đếm tần suất xuất hiện của các cặp ký tự bi-grams liên tiếp."""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], v_in: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Gộp cặp bi-gram được chọn thành 1 subword duy nhất."""
        v_out = {}
        bigram = pair
        for word in v_in:
            w_out = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    w_out.append(bigram[0] + bigram[1])
                    i += 2
                else:
                    w_out.append(word[i])
                    i += 1
            v_out[tuple(w_out)] = v_in[word]
        return v_out

    def fit(self, corpus: List[str]):
        """Huấn luyện BPE trên tập văn bản thô."""
        # 1. Khởi tạo từ điển ký tự đơn lẻ kèm ký tự kết thúc từ '</w>'
        vocab = collections.defaultdict(int)
        for sentence in corpus:
            for word in sentence.split():
                chars = tuple(list(word) + ["</w>"])
                vocab[chars] += 1

        print("--> Khởi tạo BPE Vocab ban đầu:", len(vocab))

        # 2. Thực hiện num_merges lần gộp cặp bi-gram có tần suất cao nhất
        for i in range(self.num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            print(f"  [BPE Merge {i+1}] Best Pair: {best_pair} -> '{best_pair[0] + best_pair[1]}'")

    def tokenize_word(self, word: str) -> List[str]:
        """Phân tách 1 từ thành chuỗi các Subwords đã học."""
        tokens = list(word) + ["</w>"]
        while True:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            can_merge = [p for p in pairs if p in self.merges]
            if not can_merge:
                break
            # Chọn merge xuất hiện sớm nhất trong lịch sử
            pair = min(can_merge, key=lambda p: list(self.merges.keys()).index(p))
            tokens = list(self._merge_vocab(pair, {tuple(tokens): 1}).keys())[0]
        return tokens


class WordPieceTokenizerFromScratch:
    """
    Triển khai Thuật toán WordPiece Tokenization (dùng tiền tố ## cho subword tiếp theo).
    """
    def __init__(self, vocab: Set[str], max_input_chars_per_word: int = 100):
        self.vocab = vocab
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize_word(self, word: str) -> List[str]:
        """Tách từ theo quy tắc Greedy Longest-Match-First của WordPiece."""
        if len(word) > self.max_input_chars_per_word:
            return ["[UNK]"]

        output_tokens = []
        start = 0
        is_bad = False

        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1

            if cur_substr is None:
                is_bad = True
                break

            output_tokens.append(cur_substr)
            start = end

        if is_bad:
            return ["[UNK]"]
        return output_tokens
