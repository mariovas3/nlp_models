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
        # self.taken is an index pointer in self.candidates;
        self.taken = 0
        self.vocab_size = len(vocab)
        self.big_draw = max(1, self.vocab_size // 5)
        # no need to sample special tokens such as <unk>, <pad> etc...
        self.population = list(range(vocab.num_special_tokens, self.vocab_size))

    def sample(self, num_negatives):
        """
        Sample num_negatives from the vocab;
        To make it more efficient for large vocabs, a fifth of the vocab is sampled in self.candidates
        and when these are exhausted we draw another sample;
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
        return random.random() <= max(1 - math.sqrt( t / (counts[token] / available_tokens) ), 0)
    return [[vocab[token][0] for token in line if not discard(token, counts, available_tokens)] for line in tokens]


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
    fast_check = set(sentence)
    for i in range(len(sentence)):
        curr_contexts = [sentence[j] for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1))
                         if j != i]
        contexts.append(curr_contexts)
        curr_negatives = []
        # the negatives should be random words that are not in the current sentence;
        while len(curr_negatives) < num_negatives:
            curr_negatives += [neg for neg in negatives_sampler.sample(num_negatives - len(curr_negatives))
                               if neg not in fast_check]
        negatives.append(curr_negatives)
    return zip(centers, contexts, negatives)


def get_triplets_from_corpus(tokens, negatives_sampler, num_negatives, max_window_size):
    """
    Gets an itertools.chain object comprised of zipped (centers, contexts, negatives) tuples
    coming from _get_triplets();
    :param tokens: tokens[0] is a tokenised sentence (i.e. list of ints);
    :param negatives_sampler: NegativeSampleGenerator object;
    :param num_negatives: int type, number of negatives to sample;
    :param max_window_size: int type; max context window size;
    :return: itertools.chain object;
    """
    triplets = None
    for sentence in tokens:
        if len(sentence) <= 2:
            continue
        window_size = random.randint(1, max_window_size)
        new_zipped_triplets = _get_triplets(sentence, negatives_sampler, num_negatives, window_size)
        if triplets is None:
            triplets = itertools.chain(new_zipped_triplets)
        else:
            triplets = itertools.chain(triplets, new_zipped_triplets)
    return triplets
