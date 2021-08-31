import numpy as np
import torch
import random
import sys
sys.path.append("../preprocessing")
import vocab as voc
import random
import math
import itertools


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


def subsample(tokens, vocab, t=1e-5):
    """
    This is the subsampling mechanism proposed by Mikolov et. al. for discarding very frequent words
    that carry little meaning (e.g. "the", "an", "a" etc...);
    :param tokens: a list of tokenised sentences (i.e. a list of lists);
    :param vocab: a vocab.Vocab object;
    :param t: this is a heuristic for discarding very frequent words; Depends on sample size and the construction
    of your corpus (since higher t leads to lower probability to discard any given word);
    :return: a list of lists of token_ids based on the mapping from vocab;
    """
    counts = voc.count_corpus(tokens)
    available_tokens = sum(counts.values())
    def discard(token, counts, available_tokens):
        """
        Simple utility function returns True if U(0, 1) <= max( 1 - sqrt(t/f(w)), 0)
        where f(w) = num_occurs(w) / corpus_size; and False otherwise;
        :param token: string object;
        :param counts: dict object with (string, int) pairs;
        :param available_tokens: len of corpus;
        :return: boolean based on the logic defined in the docstring;
        """
        return random.random() <= max(1 - math.sqrt(t / counts[token] * available_tokens), 0)
    samples = [[vocab[token] for token in line if discard(token, counts, available_tokens)] for line in tokens]


def _get_triplets(sentence, negatives_sampler, num_negatives, window_size):
    """
    Gets centers, contexts, negatives per a sentence which is of the form List[int] (i.e. each element of sentence
    is a token_id);
    :param sentence: len(sentence) > 2;
    :param negatives_sampler: object of type NegativeSampleGenerator;
    :param num_negatives: int type, how many negatives to sample;
    :param window_size: int type, the size of context window;
    :return: zipped triplet centers, contexts, negatives;
    """
    contexts = []
    negatives = []
    centers = sentence
    for i in range(len(sentence)):
        curr_contexts = [sentence[j] for j in range(max(0, i - window_size), min(len(sentence) - 1, i + window_size))
                         if j != i]
        contexts.append(curr_contexts)
        negatives.append(negatives_sampler.sample(num_negatives))
    return zip(centers, contexts, negatives)
