import torch
from torchtext.datasets import WikiText2
import sys
sys.path.append("../BERT")
sys.path.append("../preprocessing")
import vocab as voc
import top100GutenbergScraper as scraper
from bert_data_processing import get_nltk_tokenizer
import re
import word2vec_NEG as w2v_neg
import random
import numpy as np
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def get_sentences_from_books(file_name, num_books=100, truncate=True):
    """
    Downloads num_books from Project Gutenberg, and saves them in ../data/file_name where each line
    is a paragraph. Then uses the nltk punkt tokenizer to tokenize (makes a paragraph to a list of sentences);
    Then return tokenized sentences (each being a list of lowercased words);
    :param file_name: where to save/look for the text data;
    :param num_books: how many books to download from Project Gutenberg;
    :param truncate: mode of writing to ../data/file_name;
    :return: list of tokenized sentences (each sentence as a list of lowercased words);
    """
    scraper.save_gutenberg_books(file_name, num_books, truncate)
    tokenizer = get_nltk_tokenizer()
    with open(f"../data/{file_name}", 'r') as f:
        lines = f.readlines()
    lines = [tokenizer.tokenize(paragraph) for paragraph in lines]
    return [re.sub("[^A-Za-z]+", ' ', line).strip().lower().split()
            for paragraph in lines for line in paragraph
            if "<book" not in line and len(line.split()) > 2]


def get_iter_and_vocab_neg(file_name, num_books=100, truncate=True, reserved_tokens=["<pad>"], vocab_min_freq=0,
                         t=1e-5, num_negatives=5, max_window_size=4, shuffle=True, batch_size=512):
    """
    Returns a torch.utils.data.DataLoader and vocab.Vocab object;
    :param file_name: where to save/look for data it should be just the name of the file, not path;
    :param num_books: number of books to scrape from Project Gutenberg;
    :param truncate: mode of writing to file_name;
    :param reserved_tokens: reserved_tokens for vocab.Vocab constructor;
    :param vocab_min_freq: min_freq for vocab.Vocab constructor;
    :param t: heuristic for subsampling, defaults to 1e-5 following from Mikolov et. al.
    :param num_negatives: negative samples per context word;
    :param max_window_size: max size of context window;
    :return: torch.utils.data.DataLoader and vocab.Vocab objects;
    """
    sentences = get_sentences_from_books(file_name, num_books, truncate)
    vocab = voc.Vocab(sentences, reserved_tokens=reserved_tokens, min_freq=vocab_min_freq)
    subsampled, counts = w2v_neg.subsample(sentences, vocab, t=t)
    negatives_sampler = w2v_neg.get_negative_sampler(counts, vocab)
    centers, contexts, negatives = w2v_neg.get_triplets_from_corpus(subsampled, negatives_sampler, num_negatives,
                                                                    max_window_size)
    dataset = w2v_neg._EmbeddingsDataset(centers, contexts, negatives)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=w2v_neg._get_batches)
    return loader, vocab


def get_wikitext2_data_neg(dir, reserved_tokens=["<pad>"], vocab_min_freq=0,
                         t=1e-5, num_negatives=5, max_window_size=4, shuffle=True, batch_size=512):
    """
    Returns torch.utils.data.DataLoader object and vocab.Vocab object for the word2vec training following the
    negative sampling procedure for the WikiText2 data set;
    :param dir: Where to store the files/data from the WikiText2 dataset;
    :param reserved_tokens: to be passed to vocab.Vocab constructor;
    :param vocab_min_freq: to be passed to vocab.Vocab constructor;
    :param t: heuristic to be passed to w2v_neg.subsample function;
    :param num_negatives: how many negatives to sample per context word;
    :param max_window_size: size of context window;
    :param shuffle: boolean, to be passed to constructor of torch.utils.data.DataLoader;
    :param batch_size: to be passed to constructor of torch.utils.data.DataLoader;
    :return: torch.utils.data.DataLoader, vocab.Vocab;
    """
    tokenizer = get_nltk_tokenizer()
    train_iter = WikiText2(dir, split="train")
    train_sentences = [tokenizer.tokenize(paragraph) for paragraph in train_iter]

    train_sentences = [re.sub("[^A-Za-z]+", ' ', str(sentence)).strip().lower().split()
                       for sentence in train_sentences]

    train_tokens = [[token for token in sentence if len(token) > 2 and "unk" not in token]
                       for sentence in train_sentences]
    vocab = voc.Vocab(train_tokens, reserved_tokens=reserved_tokens, min_freq=vocab_min_freq)
    subsampled, counts = w2v_neg.subsample(train_tokens, vocab, t=t)
    negatives_sampler = w2v_neg.get_negative_sampler(counts, vocab)
    centers, contexts, negatives = w2v_neg.get_triplets_from_corpus(subsampled, negatives_sampler, num_negatives,
                                                                    max_window_size)
    dataset = w2v_neg._EmbeddingsDataset(centers, contexts, negatives)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                         collate_fn=w2v_neg._get_batches)
    return loader, vocab
