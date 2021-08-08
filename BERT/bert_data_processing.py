import sys
sys.path.append("../preprocessing")
import vocab as voc
import torch
from torch.utils.data import Dataset, DataLoader
import nltk.data
import re
import random
import bert
import top100GutenbergScraper as scraper
import os


def get_nltk_tokenizer():
    """
    Gets the nltk punkt tokenizer so that I can split the sentences in each paragraph of the Project Gutenberg books;
    :return: <nltk_punkt_tokenizer>;
    """
    if not os.path.exists("../data/tokenizers/punkt/english.pickle"):
        if not os.path.exists("../data"):
            os.mkdir("../data")
        nltk.download("punkt", download_dir="../data/")
    tokenizer = nltk.data.load("../data/tokenizers/punkt/english.pickle")
    return tokenizer


def tokenize_books(lines, tokenizer):
    """
    From list of lists of sentences to list of lists of lists, where most inner list is a tokenised sentence;
    :param lines: list of strings, each string is a paragraph;
    :param tokenizer: <nltk_punkt_tokenizer>
    :return: list of lists of lists; dim(1) is a list of sentences, dim(2) is a tokenised sentence;
    """
    lines = [tokenizer.tokenize(paragraph) for paragraph in lines]
    random.shuffle(lines)
    return [[re.sub("[^A-Za-z\'\']+", ' ', sentence).strip().lower().split() for sentence in paragraph]
             for paragraph in lines if ("<book_" not in paragraph and len(paragraph) >= 2)]


def _nsp_sentence_pairs(first, second, paragraphs):
    """
    Decide if two sentences follow each other w.p. 50% or sample a second sentence and say they don't follow each
    other. This is for the Next Sentence Prediction task of BERT pretraining;
    :param first: first tokenised sentence;
    :param second: second tokenised sentence;
    :param paragraphs: list of lists of lists; most inner list is a tokenised sentence;
    second most inner is a list of tokenised sentences (so a paragraph);
    :return: <first_sentence>, <original_or_new_second_sentence>, <is_next>
    """
    is_next = 1
    if random.random() < 0.5:
        second = random.choice(random.choice(paragraphs))
        is_next = 0
    return first, second, is_next


def _nsp_data_one_paragraph(paragraph, paragraphs, max_bert_seq_len):
    """
    Get nsp data from one paragraph;
    :param paragraph: list of tokenised sentences;
    :param paragraphs: list of paragraphs;
    :param max_bert_seq_len: maximum length of a bert style sequence (i.e. has special tokens like "<cls>" etc...);
    :return: List[Tuple(<bert_style_seqs>, <segments>, <is_next>)]
    """
    nsp_data = []
    for i in range(len(paragraph) - 1):
        first, second, is_next = _nsp_sentence_pairs(paragraph[i], paragraph[i+1], paragraphs)
        if len(first) + len(second) + 3 < max_bert_seq_len:
            tokens, segments = bert.get_tokens_and_segments(first, second)
            nsp_data.append((tokens, segments, is_next))
    return nsp_data


def _mask_bert_seq(tokens, non_special_indeces, num_masks_allowed, vocab):
    """
    Masks a bert sequence;
    :param tokens: <bert_sequence> type; i.e. <cls>first<sep>[second<sep>] - format
    :param non_special_indeces: indexes of non-special indexes from tokens; special indexes
    are <cls> and <sep>; these are not to be predicted;
    :param num_masks_allowed: originally 15% of sequence;
    :param vocab: just a vocab; see vocab.Vocab for more info;
    :return: (<masked_sequence>, <list_maskedPosition_originalLabel_pairs>)
    """
    to_mask = [token for token in tokens]
    random.shuffle(non_special_indeces)
    positions_and_labels = []
    for idx in non_special_indeces:
        if len(positions_and_labels) == num_masks_allowed:
            break
        mask = None
        if random.random() < 0.8:
            mask = "<mask>"
        else:
            if random.random() < 0.5:
                mask = tokens[idx]
            else:
                mask = vocab.convert_to_token(random.randint(0, len(vocab) - 1))[0]
        # append masked position and original token at that position
        positions_and_labels.append((idx, tokens[idx]))
        to_mask[idx] = mask
    return to_mask, positions_and_labels


def _mlm_data_from_seq(tokens, vocab):
    """
    Gets data for BERT Masked Language Modelling pretraining task;
    :param tokens: <bert_style_sequence>;
    :param vocab: vocab.Vocab object;
    :return: <idxs_for_masked_seq>, <positions_of_masks>, <idxs_for_original_labels_of_masks>
    """
    non_special_indeces = []
    for i in range(len(tokens)):
        if tokens[i] not in ["<cls>", "<sep>"]:
            non_special_indeces.append(i)
    num_masks_allowed = max(1, round((len(tokens) - 3) * 0.15))
    masked_seq, masked_positions_labels = _mask_bert_seq(tokens, non_special_indeces,
                                                         num_masks_allowed, vocab)
    masked_positions_labels = sorted(masked_positions_labels, key=lambda x: x[0])
    positions = [item[0] for item in masked_positions_labels]
    labels = [item[1] for item in masked_positions_labels]
    return vocab[masked_seq], positions, vocab[labels]


def _get_bert_pretraining_inputs(pretrain_tuples, max_len, vocab):
    """
    Pads the inputs for pretraining;
    :param pretrain_tuples: Typle(masked_ids, positions_of_masks, original_labels_of_masks, segments, is_next);
    :param max_len: max len of bert_style_seq;
    :param vocab: vocab.Vocab object;
    :return: (all_token_idxs, all_segments, attention_masks, all_positions_of_masks,
            all_mlm_weights, all_label_idxs, nsp_labels) padded up to max_len;
    """
    max_masks_allowed = round(max_len * 0.15)
    all_token_idxs, all_positions_of_masks, all_label_idxs = [], [], []
    all_segments, all_mlm_weights, nsp_labels, attention_masks = [], [], [], []
    for masked_token_idxs, positions_of_masks, idxs_of_labes, segments, is_next in pretrain_tuples:
        all_token_idxs.append(torch.tensor(
            masked_token_idxs + vocab["<pad>"] * (max_len - len(masked_token_idxs)), dtype=torch.long))
        all_positions_of_masks.append(torch.tensor(
            positions_of_masks + [0] * (max_masks_allowed - len(positions_of_masks)), dtype=torch.long))
        all_label_idxs.append(torch.tensor(
            idxs_of_labes + [0] * (max_masks_allowed - len(idxs_of_labes)), dtype=torch.long))
        all_segments.append(torch.tensor(
            segments + [0] * (max_len - len(segments)), dtype=torch.long))
        all_mlm_weights.append(torch.tensor(
            [1.] * len(idxs_of_labes) + [0.] * (max_masks_allowed - len(idxs_of_labes)), dtype=torch.float32))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
        attention_masks.append(torch.tensor(len(masked_token_idxs), dtype=torch.float32))
    return (all_token_idxs, all_segments, attention_masks, all_positions_of_masks,
            all_mlm_weights, all_label_idxs, nsp_labels)


class GutenbergDataset(Dataset):
    def __init__(self, paragraphs, max_len, min_freq):
        """
        :param paragraphs: list of lists of lists; most inner list is a tokenised sentence;
        second most inner list is a list of tokenised sentences;
        :param max_len: max len of bert_style_sequence (includes special tokens like "<cls" and "<sep>");
        :param min_freq: minimum frequency for a token to be in the vocabulary;
        """
        super(GutenbergDataset, self).__init__()
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = voc.Vocab(sentences, reserved_tokens=["<pad>", "<mask>", "<cls>", "<sep>"], min_freq=min_freq)
        pretrain_tuples = []
        for paragraph in paragraphs:
            pretrain_tuples += _nsp_data_one_paragraph(paragraph, paragraphs, max_len)
        pretrain_tuples = [(_mlm_data_from_seq(tokens, self.vocab) + (segments, is_next))
                           for tokens, segments, is_next in pretrain_tuples]
        (self.all_token_idxs, self.all_segments, self.attention_masks, self.all_positions_of_masks,
         self.all_mlm_weights, self.all_label_idxs, self.nsp_labels) = _get_bert_pretraining_inputs(pretrain_tuples,
                                                                                            max_len, self.vocab)

    def __len__(self):
        return len(self.nsp_labels)

    def __getitem__(self, idx):
        return (self.all_token_idxs[idx], self.all_segments[idx], self.attention_masks[idx],
                self.all_positions_of_masks[idx], self.all_mlm_weights[idx],
                self.all_label_idxs[idx], self.nsp_labels[idx])


def get_gutenberg_loader_and_vocab(batch_size, max_len, file_name, num_books, truncate, min_freq):
    """
    Wraps a DataLoader on the Gutenberg data based on the provided parameters and returns a vocab.Vocab object
    based on that data;
    :param batch_size: batch_size;
    :param max_len: max length of bert sequence;
    :param file_name: where to store gutenberg books;
    :param num_books: choose between 1 and 100 books to scrape;
    :param truncate: to truncate or not the file where the books will be stored;
    :param min_freq: minimum frequency of token to add to vocabulary;
    :return: torch.utils.data.DataLoader, vocab.Vocab
    """
    scraper.save_gutenberg_books(file_name, num_books, truncate)
    nltk_tokenizer = get_nltk_tokenizer()
    with open(f"../data/{file_name}", 'r') as books:
        lines = books.readlines()
    paragraphs = tokenize_books(lines, nltk_tokenizer)
    pretrain_dataset = GutenbergDataset(paragraphs, max_len, min_freq)
    pretrain_loader = DataLoader(pretrain_dataset, shuffle=True, batch_size=batch_size)
    return pretrain_loader, pretrain_dataset.vocab
