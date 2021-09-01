import sys
sys.path.append("../BERT")
import top100GutenbergScraper as scraper
from bert_data_processing import get_nltk_tokenizer
import re


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
    return [re.sub("[^A-Za-z\']+", ' ', line).strip().lower().split()
            for paragraph in lines for line in paragraph
            if "<book" not in line and len(line.split()) > 2]
