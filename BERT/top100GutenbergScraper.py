import requests
from bs4 import BeautifulSoup
import os
import re


def _get_web_pages_of_books(num_books=100):
    """
    Gets the links to the pages of the 100 top books yesterday on project Gutenberg;
    :return: list of URLs to homepages for each of the "Top 100 EBooks yesterday" from project Gutenberg;
    """
    various_charts = "https://www.gutenberg.org/browse/scores/top"
    base_url = "https://www.gutenberg.org"
    with requests.get(various_charts) as charts_page:
        soup1 = BeautifulSoup(charts_page.content, "html.parser")
    return [base_url + h.get("href") for h in soup1.select("ol li a")[:min(abs(num_books), 100)]]


def _get_links_to_text(books_menu_pages):
    """
    Stores the "Read this book online:HTML" pages of each book in links_to_text;
    :param books_menu_pages: list of urls to menu pages of books;
    :return: list of urls to the "Read this book online:HTML" page for each book;
    """
    base_url = "https://www.gutenberg.org"
    links_to_text = []
    for menu_page in books_menu_pages:
        with requests.get(menu_page) as r:
            soup = BeautifulSoup(r.content, "html.parser")
        links_to_text.append(base_url + soup.select("tr td a")[0].get("href"))
    return links_to_text


def _scrape_books(links_to_text, file_name, truncate=True):
    """
    Scrape books from the links in links_to_text and save them in
    ../data/<file_name>
    :param links_to_text: urls to HTML of books;
    :param file_name: where to store the books;
    :param truncate: whether to truncate if the path already exists, defaults to True in which
    case it truncates; If False, it appends;
    :return: void;
    """
    if not os.path.exists("../data"):
        os.mkdir("../data")
    save_path = os.path.join("../data/", file_name)

    # select mode based on truncate;
    mode = 'w' if truncate else 'a'

    # print some info;
    print(f"downloading {len(links_to_text)} books...")

    for url in links_to_text:
        # open request
        with requests.get(url) as f:
            soup = BeautifulSoup(f.content, "html.parser")

        # save to file;
        # before the content of a book it has the following indicator:
        # <book_firstName_secondName_authorName>
        # a regex is there to get rid of ',.' characters and replace with '\s';
        with open(save_path, mode) as books:
            books.write(f"<book_{'_'.join(re.sub('[,.]+', ' ', soup.title.get_text()).strip().lower().split())}>\n")
            for p in soup.select("body p"):
                books.write("".join(p.get_text().splitlines()))
                books.write('\n')
            books.write('\n')
        # stop truncation after first book;
        if mode == 'w':
            mode = 'a'


def save_gutenberg_books(file_name, num_books=100, truncate=True):
    """
    Saves books in ../data/file_name;
    :param file_name: name of file to store the books;
    :param num_books: choose how many books to scrape [1, 100];
    :param truncate: boolean, if True, truncates the contents of file_name, else appends to file_name;
    :return: void;
    """
    carry_on = None
    if os.path.exists(f"../data/{file_name}"):
        while carry_on not in ['n', 'y']:
            carry_on = input(
                "A file with that name already exists, if truncate is True I will overwrite it. Continue [y/n]:")
        if carry_on == 'n':
            return
    _scrape_books(_get_links_to_text(_get_web_pages_of_books(num_books)), file_name, truncate)
