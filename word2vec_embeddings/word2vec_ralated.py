import numpy as np
import torch
import random
import sys
sys.path.append("../preprocessing")
import vocab as voc
import random


class NegativeSampleGenerator:
    def __init__(self, vocab):
        """
        Sampling indexes from vocab;
        :param vocab: a vocab.Vocab object;
        """
        self.candidates = []
        self.taken = 0
        self.vocab_size = len(vocab)
        self.big_draw = max(1, self.vocab_size // 5)
        # no need to sample special tokens such as <unk>, <pad> etc...
        self.population = list(range(vocab.num_special_tokens, self.vocab_size))

    def sample(self, num_negatives):
        """
        Sample num_negatives from the vocab;
        To make it more efficient for large vocabs, a fifth of the vocab is sampled in self.candidates
        and these are exhausted we make another sample;
        :param num_negatives: how many negatives to sample;
        :return: list of indexes from the vocab of len num_negatives;
        """
        result = []
        if len(self.candidates) == 0 or self.taken + num_negatives > len(self.candidates):
            self.candidates = random.sample(self.population, self.big_draw)
            self.taken = 0
        if num_negatives > self.big_draw:
            times = num_negatives // self.big_draw
            for _ in range(times):
                result += random.sample(self.population, self.big_draw)
            self.candidates = random.sample(self.population, self.big_draw)
            if num_negatives % self.big_draw == 0:
                return result
            num_negatives %= self.big_draw
        self.taken += num_negatives
        return result + self.candidates[self.taken-num_negatives:self.taken]
