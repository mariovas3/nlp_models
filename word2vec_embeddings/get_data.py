import torch
import sys
sys.path.append("../BERT")
sys.path.append("../preprocessing")
import vocab as voc
import top100GutenbergScraper as scraper
from bert_data_processing import get_nltk_tokenizer
import re
import word2vec_ralated as w2v


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


def get_iter_and_vocab(file_name, num_books=100, truncate=True, reserved_tokens=["<pad>"], vocab_min_freq=0,
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
    subsampled, counts = w2v.subsample(sentences, vocab, t=t)
    negatives_sampler = w2v.get_negative_sampler(counts, vocab)
    centers, contexts, negatives = w2v.get_triplets_from_corpus(subsampled, negatives_sampler, num_negatives,
                                                                max_window_size)
    dataset = w2v._EmbeddingsDataset(centers, contexts, negatives)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=w2v._get_batches)
    return loader, vocab