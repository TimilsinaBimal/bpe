import argparse
import json

from colorama import Back, Fore, Style, init

from bpe.bpe import BytepairEncoding


def sum_of_digits(number):
    while True:
        number = sum(int(digit) for digit in str(number))

        if len(str(number)) == 1:
            return number


def visualize_tokens(tokens: list):
    # Initialize colorama
    init()

    # Define colors using colorama
    highlight_colors = {
        "0": Back.RED,
        "1": Back.YELLOW,
        "2": Back.GREEN,
        "3": Back.BLUE,
        "4": Back.MAGENTA,
        "5": Back.CYAN,
        "6": Back.WHITE,
        "7": Back.BLACK,
        "8": Back.LIGHTRED_EX,
        "9": Back.LIGHTGREEN_EX,
    }
    reset_color = Style.RESET_ALL
    vocab = load_vocab()
    for idx, token in enumerate(tokens):
        decoded_text = vocab[token].decode("utf-8", errors="replace")
        single_digit_token = sum_of_digits(token)
        background_color = highlight_colors.get(str(single_digit_token), reset_color)
        foreground_color = Fore.BLACK if background_color != Back.BLACK else Fore.WHITE
        end_sep = " " if idx < len(tokens) - 1 else "\n"
        print(
            f"{background_color}{foreground_color}{decoded_text}{reset_color}",
            end=end_sep,
        )


def train_bpe(file_path: str = "input.txt", vocab_size: int = 1000):
    return bpe.train(filepath=file_path, vocab_size=vocab_size)


def encode(text: str) -> list:
    return bpe.encode(text)


def decode(tokens: list) -> str:
    return bpe.decode(tokens)


def load_vocab(model_path: str = "models/vocab.json"):
    with open(model_path, encoding="utf-8") as f:
        vocab = json.load(f)
    vocab = {int(k): bytes(v) for k, v in vocab.items()}
    return vocab


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description="Bytepair Encoding")
    parser.add_argument("--train", action="store_true", help="Train BPE model")
    parser.add_argument("--encode", type=str, help="Encode text")
    parser.add_argument("--decode", type=str, help="Decode tokens. Comma seperated values eg: 12,13 -> [12, 13]")
    parser.add_argument("--file", type=str, help="File path")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocab size, default:1000")
    parser.add_argument("--visualize", action="store_true", default=True, help="Visualize token strings.")

    args = parser.parse_args()
    bpe = BytepairEncoding()

    if args.train:
        if not args.file:
            raise ValueError("File path is required for training")
        train_bpe(args.file, args.vocab_size)

    elif args.encode:
        tokens = encode(args.encode)
        if args.visualize:
            visualize_tokens(tokens)
        print(tokens)

    elif args.decode:
        tokens = args.decode.split(",")
        tokens = [int(token) for token in tokens]
        text = decode(tokens)
        print(text)
