## Will contain NLP models implemented with PyTorch.
### Currently has:
* Pretraining of BERT and using it for a downstream text classification task;
  * In the `BERT` directory there is a demo jupyter notebook demonstrating how to pretrain BERT and use it for classifying articles from the AG_NEWS dataset. I achieved **90%** accuracy on a balanced test set with relatively little computational demand. The notebook also explores an interesting idea of how to represent the articles so that you can pretrain BERT on them.
  * In the same directory there is also a scraper of Project Gutenberg books which is likely to be more useful if you want to pretrain BERT on more data. In the demo notebook though, I only pretrain BERT on the *training* part of the AG_NEWS dataset.
  * If you are interested in the Project Gutenberg scraper though, you can find more details in the `BERT/top100GutenbergScraper.py` file;
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
