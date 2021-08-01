import requests
from bs4 import BeautifulSoup
import os


def _get_web_pages_of_books(num_books=100):
    """
    Gets the links to the pages of the 100 top books yesterday on project Gutenberg;
    :return: URLs to the "Top 100 EBooks yesterday" from project Gutenberg;
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
    :return: list of urls to the "Read this book online:HTML" page of each book;
    """
    base_url = "https://www.gutenberg.org"
    links_to_text = []
    for menu_page in books_menu_pages:
        with requests.get(menu_page) as r:
            soup = BeautifulSoup(r.content, "html.parser")
        links_to_text.append(base_url + soup.select("tr td a")[0].get("href"))
    return links_to_text


def _scrape_books(links_to_text, file_name):
    """
    Scrape books from the links in links_to_text and save them in
    ../data/<file_name>
    :param links_to_text: urls to HTML of books;
    :param file_name: where to store the books;
    :return: -1 if ../data/file_name exists, 1 otherwise;
    """
    if os.path.exists(f"../data/{file_name}"):
        return -1
    if not os.path.exists("../data"):
        os.mkdir("../data")
    save_path = os.path.join("../data/", file_name)

    print(f"downloading {len(links_to_text)} books...")
    i = 0
    for url in links_to_text:
        # open request
        with requests.get(url) as f:
            soup = BeautifulSoup(f.content, "html.parser")

        # save to file;
        with open(save_path, 'a') as books:
            books.write(f"<book_{i}>\n")
            for p in soup.select("body p"):
                books.write(p.get_text())
            books.write('\n')
        i += 1
    return 0


def save_gutenberg_books(file_name, num_books=100):
    """
    Saves books in ../data/file_name;
    :param file_name: name of file to store the books;
    :param num_books: choose how many books to scrape [1, 100];
    :return: -1 if path already exists, 1 otherwise;
    """
    return _scrape_books(_get_links_to_text(_get_web_pages_of_books(num_books)), file_name)


if __name__ == "__main__":
    save_gutenberg_books("gutenberg_books.txt", 5)
