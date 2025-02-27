def find_pairs(tokens: list, pairs: dict = {}) -> dict:
    for pair in zip(tokens, tokens[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs


def find_max_occuring_pair(pairs: dict) -> tuple[int] | None:
    max_pairs = max(pairs, key=pairs.get)
    return max_pairs if pairs[max_pairs] > 1 else None  # if no pairs occurs return None


def replace_pair(tokens: list, pair: tuple, new_token: int) -> list:
    results = []
    idx = 0
    while idx < len(tokens) - 1:
        token_pair = (tokens[idx], tokens[idx + 1])
        if pair == token_pair:
            results.append(new_token)  # replace pair with new token
            idx += 2  # skip two steps as idx+1 is already replaced

        else:
            results.append(tokens[idx])  # if don't match just append the token
            idx += 1

    if idx == len(tokens) - 1:  # if last pair doesn't match, append the last token
        results.append(tokens[idx])
    return results


def bytepair_encoding(tokens: list, vocab_size: int, max_token: int = 256):
    if vocab_size < 256:
        raise ValueError("vocab_size must be greater than 256")
    new_token = max_token
    merge_dict = {}
    while new_token <= vocab_size:
        token_pair = find_max_occuring_pair(tokens)
        if not token_pair:  # if no pair occurs
            print("No more pairs to merge")
            break
        merge_dict[token_pair] = new_token
        print(f"Token Pair: {token_pair}")
        tokens = replace_pair(tokens, token_pair, new_token)
        new_token += 1
    return tokens, merge_dict
