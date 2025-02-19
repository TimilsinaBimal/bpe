def find_pairs(tokens: list) -> dict:
    pairs = {}
    for (t1, t2) in zip(tokens, tokens[1:]):
        pair = (t1, t2)
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs


def find_max_occuring_pair(tokens:list) -> tuple:
    pairs = find_pairs(tokens)
    max_pairs = max(pairs, key=pairs.get) 
    return max_pairs if pairs[max_pairs] > 1 else None # if no pairs occurs return None