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
    def __init__(self, vocab, sampling_weights):
        """
        Sampling indexes from vocab according to the distribution corresponding to self.sampling_weights;
        :param vocab: a vocab.Vocab object;
        """
        self.candidates = []
        # self.taken is an index pointer in self.candidates;
        self.taken = 0
        self.vocab_size = len(vocab)
        self.big_draw = max(1, self.vocab_size // 5)
        # no need to sample special tokens such as <unk>, <pad> etc...
        self.population = list(range(vocab.num_special_tokens, self.vocab_size))
        # sampling weights for sampling negative examples;
        self.sampling_weights = sampling_weights

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
            self.candidates = random.choices(self.population, weights=self.sampling_weights, k=self.big_draw)
            self.taken = 0
        if num_negatives > self.big_draw:
            times = num_negatives // self.big_draw
            for _ in range(times):
                result += random.choices(self.population, weights=self.sampling_weights, k=self.big_draw)
            self.candidates = random.choices(self.population, weights=self.sampling_weights, k=self.big_draw)
            if num_negatives % self.big_draw == 0:
                return result
            num_negatives %= self.big_draw
        self.taken += num_negatives
        return result + self.candidates[self.taken-num_negatives:self.taken]


def get_negative_sampler(counts, vocab):
    """
    Sorts out the sampling distribution for sampling negatives according to the Mikolov et. al. paper;
    i.e. proportional to word_frequency ** 0.75;
    :param counts: dict object of (token, frequency) pairs;
    :param vocab: vocab.Vocab object;
    :return: NegativeSampleGenerator object;
    """
    sampling_weights = [counts[vocab.convert_to_token(token_id)[0]] ** 0.75
                        for token_id in range(vocab.num_special_tokens, len(vocab))]
    return NegativeSampleGenerator(vocab, sampling_weights)


def subsample(tokens, vocab, t=1e-5):
    """
    This is the subsampling mechanism proposed by Mikolov et. al. for discarding very frequent words
    that carry little meaning (e.g. "the", "an", "a" etc...);
    :param tokens: a list of tokenised sentences (i.e. a list of lists);
    :param vocab: a vocab.Vocab object;
    :param t: this is a heuristic for discarding very frequent words; Depends on sample size and the construction
    of your corpus (since higher t leads to lower probability to discard any given word);
    :return: a list of lists of token_ids based on the mapping from vocab; and counts of frequencies as a dict object;
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
    return ([[vocab[token][0] for token in line if not discard(token, counts, available_tokens)]
             for line in tokens], counts)


def _get_triplets(sentence, negatives_sampler, num_negatives, window_size):
    """
    Gets centers, contexts, negatives per a sentence which is of the form List[int] (i.e. each element of sentence
    is a token_id);
    :param sentence: len(sentence) > 2;
    :param negatives_sampler: object of type NegativeSampleGenerator;
    :param num_negatives: int type, how many negatives to sample;
    :param window_size: int type, the size of context window;
    :return: tuple centers, contexts, negatives;
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
        # there should be num_negatives per context word;
        K = num_negatives * len(curr_contexts)
        while len(curr_negatives) < K:
            curr_negatives += [neg for neg in negatives_sampler.sample(K - len(curr_negatives))
                               if neg not in fast_check]
        negatives.append(curr_negatives)
    return centers, contexts, negatives


def get_triplets_from_corpus(tokens, negatives_sampler, num_negatives, max_window_size):
    """
    Gets a tuple of lists object comprised of (centers, contexts, negatives);
    :param tokens: tokens[0] is a tokenised sentence (i.e. list of ints);
    :param negatives_sampler: NegativeSampleGenerator object;
    :param num_negatives: int type, number of negatives to sample;
    :param max_window_size: int type; max context window size;
    :return: tuple of 3 lists;
    """
    triplets = None
    all_centers, all_contexts, all_negatives = [], [], []
    for sentence in tokens:
        if len(sentence) < 2:
            continue
        window_size = random.randint(1, max_window_size)
        center, context, negative = _get_triplets(sentence, negatives_sampler, num_negatives, window_size)
        all_centers += center
        all_contexts += context
        all_negatives += negative
    return all_centers, all_contexts, all_negatives


def _get_batches(ccn_iter):
    """
    This is the collate_fn for the torch.utils.data.DataLoader object;
    :param ccn_iter: comes from a torch.utils.data.Dataset.__getitem__ method;
    :return: tuple of torch.tensor objects;
    """
    max_len = -1
    for center, context, negative in ccn_iter:
        max_len = max(max_len, len(context) + len(negative))

    centers, contexts_and_negatives, coefficients, mask_pads = [], [], [], []
    for center, contexts, negatives in ccn_iter:
        centers += [center]
        temp_len = len(contexts) + len(negatives)
        contexts_and_negatives += [contexts + negatives + [0] * (max_len - temp_len)]
        coefficients += [[1] * len(contexts) + [-1] * len(negatives) + [0] * (max_len - temp_len)]
        mask_pads += [[1] * temp_len + [0] * (max_len - temp_len)]
    return (torch.tensor(centers).reshape(-1, 1), torch.tensor(contexts_and_negatives),
            torch.tensor(coefficients, dtype=torch.float32), torch.tensor(mask_pads, dtype=torch.float32))


class _EmbeddingsDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives, **kwargs):
        """
        Custom dataset object for word2vec embeddings;
        :param centers: list of ints;
        :param contexts: list of lists of ints;
        :param negatives: list of lists of ints;
        :param kwargs: passed to base class constructor;
        """
        super(_EmbeddingsDataset, self).__init__(**kwargs)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx], self.negatives[idx]

    def __len__(self):
        return len(self.centers)
