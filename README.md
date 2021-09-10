## Will contain NLP models implemented with PyTorch.
### Currently has:
* Implementation of BERT for pretraining;
  * In the `BERT` directory there is a demo jupyter notebook demonstrating how to pretrain BERT;
  * The data used are scraped from Project Gutenberg books;
  * More on the data acquisition you can find in the `BERT/top100GutenbergScraper.py` file;
  * The data processing is implemented in the `BERT/bert_data_processing.py` file;
  * The implementation of the model itself (*BertModel*) can be found in `BERT/bert.py`;
  * The vocabulary for the data is implemented in `preprocessing/vocab.py`;
* Implementation of `word2vec` embeddings following a negative sampling training procedure (Mikolov et. al.).
  * The embeddings are trained on the `WikiText2` dataset;
  * There is a demo notebook that demonstrates how the embeddings are trained;
    * At the end of the notebook there is an **insightful plot** of a random sample of embeddings in 2D (obtained with TSNE);
    * The demo notebook is located in `word2vec_embeddings/w2v_NEG_WikiText2.ipynb`;
  * The main data wrangling is contained in `word2vec_embeddings/word2vec_NEG.py` (sampling centers, contexts and negatives);
  * The data loading is contained in `word2vec_embeddings/get_data.py`;