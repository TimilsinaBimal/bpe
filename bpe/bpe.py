import json
import os

from .regex import RegexTokenizer
from .utils import find_pairs, replace_pair


class BytepairEncoding:
    def __init__(self, regex_model: str = "gpt4", model_path: str = "./models"):
        self.regex = RegexTokenizer()
        self.regex_model = regex_model
        self.model_path = model_path
        os.makedirs(self.model_path, exist_ok=True)

    def _read_file(self, filepath: str):
        with open(filepath) as f:
            text = f.read()
        return text

    def _save_model(self, vocab, merge_dict):
        # Convert bytes to strings in vocab for JSON serialization
        serializable_vocab = {str(k): list(v) if isinstance(v, bytes) else v for k, v in vocab.items()}

        # Convert tuple keys to string representation in merge_dict for JSON
        # serialization
        serializable_merge_dict = {f"{k[0]},{k[1]}": v for k, v in merge_dict.items()}

        # Save vocab
        with open(f"{self.model_path}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

        # Save merge dictionary
        with open(f"{self.model_path}/merge_dict.json", "w", encoding="utf-8") as f:
            json.dump(serializable_merge_dict, f, ensure_ascii=False, indent=2)

    def _load_model(self):
        with open(f"{self.model_path}/vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        # Convert strings back to bytes in vocab
        vocab = {int(k): bytes(v) for k, v in vocab.items()}
        with open(f"{self.model_path}/merge_dict.json", encoding="utf-8") as f:
            merge_dict = json.load(f)

        merge_dict = {tuple(map(int, k.split(","))): v for k, v in merge_dict.items()}

        return vocab, merge_dict

    def train(self, filepath: str, vocab_size: int):
        text = self._read_file(filepath)
        regex_text = self.regex.match_pattern(text, self.regex_model)

        vocab = {idx: bytes([idx]) for idx in range(256)}
        idx = 256

        tokens = [list(t.encode("utf-8")) for t in regex_text]
        merge_dict = {}
        while idx <= vocab_size:
            pairs = {}
            for token in tokens:
                find_pairs(token, pairs)

            # find max pair
            token_pair = max(pairs, key=pairs.get)

            if not token_pair:  # if no pair occurs
                print("No more pairs to merge")
                break
            merge_dict[token_pair] = idx
            vocab[idx] = vocab[token_pair[0]] + vocab[token_pair[1]]
            print(f"Token Pair: {token_pair}")
            tokens = [replace_pair(token, token_pair, idx) for token in tokens]
            idx += 1

        self._save_model(vocab, merge_dict)
        return vocab, merge_dict

    def encode(self, text: str) -> list:
        # load vocab and merge dict
        _, merge_dict = self._load_model()

        regex_text = self.regex.match_pattern(text, self.regex_model)
        tokens = [list(t.encode("utf-8")) for t in regex_text]
        while True and len(tokens) > 1:
            # get merge_pair
            merge_pairs = {}
            for token in tokens:
                find_pairs(token, merge_pairs)
            # find max pair
            if not merge_pairs:
                break
            merge_pair = min(merge_pairs, key=lambda x: merge_dict.get(x, float("inf")))
            # if no merge pair found
            if merge_dict.get(merge_pair) is None:
                break
            # get new token
            new_token = merge_dict[merge_pair]
            # replace pair
            tokens = [replace_pair(token, merge_pair, new_token) for token in tokens]
        # flatten the tokens
        tokens = [item for sublist in tokens for item in sublist]
        return tokens

    def decode(self, tokens: list) -> str:
        vocab, _ = self._load_model()
        tokens = b"".join(vocab.get(token) for token in tokens)
        text = tokens.decode("utf-8", errors="replace")
        return text
