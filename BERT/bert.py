import torch
from torch import nn
import transformer as tr


def get_tokens_and_segments(tokens1, tokens2=None):
    """
    Get bert-style tokens and segments from 1 or 2 lists of tokens;
    :param tokens1: list of tokens for first sequence
    :param tokens2: optional list of tokens for second sequence;
    :return: (bert_style_tokens, segments)
    """
    tokens= ["<cls>"] + tokens1 + ["<sep>"]
    segments = [0] * len(tokens)
    if tokens2 is not None:
        tokens += tokens2 + ["<sep>"]
        segments += [1] * (len(tokens2) + 1)
    return tokens, segments


class BertEncoder(nn.Module):
    def __init__(self, key_dim, query_dim, value_dim, hidden_dim, num_heads,
                 norm_dim, ffn_input, ffn_hidden, num_layers, vocab_size, pos_encoding_size,
                 dropout=0, with_bias=True, **kwargs):
        super(BertEncoder, self).__init__(**kwargs)
        self.word_embeds = nn.Embedding(vocab_size, hidden_dim)
        self.segment_embeds = nn.Embedding(2, hidden_dim)
        # make it (1, pos_encoding_size, hidden_dim) so you can add it to word and segment embedding batches;
        self.pos_encodings = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros((pos_encoding_size, hidden_dim), dtype=torch.float32).unsqueeze(0)), requires_grad=True)
        self.encoders = nn.Sequential()
        for i in range(num_layers):
            self.encoders.add_module(
                f"encoder_block_{i}",
                tr.EncoderBlock(key_dim, query_dim, value_dim, hidden_dim, num_heads,
                                norm_dim, ffn_input, ffn_hidden, dropout, with_bias, **kwargs)
            )

    def forward(self, token_ids, segments, attention_masks):
        """
        BertEncoder forward pass;
        :param token_ids: token_ids.shape = (batch_size, <indeces_from_vocab>)
        :param segments: segments.shape = (batch_size, segments)
        :param attention_masks: attention_masks.shape = (batch_size);
        for each batch tells me the position from which padding of token_ids begins;
        :return: encodings of tokens in token ids of shape (batch_size, token_ids.shape[1], hidden_dim)
        """
        embeds = self.word_embeds(token_ids) + self.segment_embeds(segments)
        embeds += self.pos_encodings.data[:, :embeds.shape[1], :]
        print(embeds.shape)
        for mod in self.encoders:
            embeds = mod(embeds, attention_masks)
        return embeds


class MLM(nn.Module):
    def __init__(self, mlm_input, mlm_hiddens, vocab_size, **kwargs):
        """
        This is for Masked Language Modelling (needed to pretrain BERT);
        :param mlm_input: usually the same as hidden_dim from an encoder block;
        :param mlm_hiddens: some intermediate dimension;
        :param vocab_size: This needs to predict masked words, so there are vocab_size possibilities;
        """
        super(MLM, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(mlm_input, mlm_hiddens),
            nn.ReLU(),
            nn.Linear(mlm_hiddens, vocab_size)
        )

    def forward(self, X, masked_positions):
        """
        :param X: the encodings of the input tokens, X.shape = (batch_size, num_queries, hidden_dim)
        :param masked_positions: has the indeces of the masks in each batch,
        masked_positions.shape = (batch_size, masks_per_batch)
        :return: self.net(<masked_tokens_of_X>);
        """
        # extract the masked_positions from X
        batch_size = masked_positions.shape[0]
        masks_per_batch = masked_positions.shape[1]
        # get (batch_idx, position) pair for each masked_position;
        batch_idx = torch.repeat_interleave(torch.arange(0, batch_size), masks_per_batch)
        X_masked = X[batch_idx, masked_positions.reshape(-1)].reshape((batch_size, masks_per_batch, -1))
        return self.net(X_masked)


class NSP(nn.Module):
    def __init__(self, nsp_input, nsp_hidden, **kwargs):
        """
        This is for the Next Sentence Prediction task for BERT pretraining;
        No need to give output dimension since a pair of sentences can either follow
        each other or not i.e. it is a binary classification problem;
        :param nsp_input: generally the dimension of the <cls> representation;
        :param nsp_hidden: some intermediate dimension;
        :param kwargs: to be passed to the constructor of nn.Module;
        """
        super(NSP, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(nsp_input, nsp_hidden),
            nn.ReLU(),
            nn.Linear(nsp_hidden, 2)
        )

    def forward(self, cls_token):
        return self.net(cls_token)
