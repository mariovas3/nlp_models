## Will contain NLP models implemented with PyTorch.
### Currently has:
* Implementation of BERT for pretraining;
  * In the BERT directory there is a demo jupyter notebook demonstrating how to pretrain BERT;
  * The data used are scraped from Project Gutenberg books;
  * More on the data acquisition you can find in the BERT/top100GutenbergScraper.py file;
  * The data processing is implemented in the BERT/bert_data_processing.py file;
  * The implementation of the model itself (*BertModel*) can be found in BERT/bert.py;
  * The vocabulary for the data is implemented in preprocessing/vocab.py;