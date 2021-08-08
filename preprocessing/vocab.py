def tokenize(lines, mode="word"):
    """
    From list of strings to list of lists of tokens;
    :param lines: list of strings;
    :param mode: "words" or "char", former tokenizes to words, latter to chars;
    :return: list of lists of tokens;
    """
    if mode == "word":
        return [line.split() for line in lines]
    elif mode == "char":
        return [list(line) for line in lines]
    else:
        print(f"Unrecognised mode: {mode}")


def count_corpus(corpus):
    """
    :param corpus: list of lists of tokens (2D) or list of tokens (1D);
    :return: dictionary of <token, count> pairs;
    """
    freqs = dict()
    if len(corpus) == 0:
        return freqs
    if isinstance(corpus[0], list):
        for line in corpus:
            for token in line:
                if token not in freqs:
                    freqs[token] = 1
                else:
                    freqs[token] += 1
    else:
        for token in corpus:
            if token not in freqs:
                freqs[token] = 1
            else:
                freqs[token] += 1
    return freqs


class Vocab:
    def __init__(self, tokens=None, reserved_tokens=None, min_freq=0):
        """
        :param tokens: 1D or 2D list of tokens;
        :param reserved_tokens: list of reserved tokens other than <unk>;
        :param min_freq: take tokens that appear more than min_freq times;
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        token_counts = count_corpus(tokens)
        self._sorted_tokens = sorted([item for item in token_counts.items()
                                      if item[1] > min_freq], key=lambda x: x[1], reverse=True)
        self._idx_to_token = ["<unk>"] + reserved_tokens
        self._token_to_idx = {self._idx_to_token[i]: i for i in range(len(self._idx_to_token))}
        for token, _ in self._sorted_tokens:
            if token not in self._token_to_idx:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def __len__(self):
        return len(self._idx_to_token)

    def _tokens_to_index(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx.get(tokens, 0)  # idx if in dict else idx_of_unk
        return [self._tokens_to_index(token) for token in tokens]

    def __getitem__(self, tokens):
        idxs = self._tokens_to_index(tokens)
        return idxs if isinstance(idxs, list) else [idxs]

    def _idxs_to_tokens(self, idxs):
        if not isinstance(idxs, (list, tuple)):
            return self._idx_to_token[idxs] if (idxs >= 0 and idxs < len(self._idx_to_token)) else "<unk>"
        return [self._idxs_to_tokens(idx) for idx in idxs]

    def convert_to_token(self, idxs):
        tokens = self._idxs_to_tokens(idxs)
        return tokens if isinstance(tokens, list) else [tokens]
