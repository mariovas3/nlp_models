import torch
from torch import nn


def masked_softmax(X, thresholds, value = -1e6):
    """
    Text seqs of variable lengths, so there is going to be some padding, hence take only the
    relevant tokens not the <pad> tokens when doing softmax;
    :param X: X.shape = (batch_size, num_queries, num_keys=num_values)
    :param thresholds: 1D or 2D torch.Tensor saying from which position to stop counting;
    :param value: replace X entries with that value before softmax;
    :return: same dimension torch.Tensor as X, but undergone a more careful softmax procedure;
    """
    if thresholds is None:
        return nn.functional.softmax(X, dim=2)

    og_shape = X.shape
    if thresholds.dim() == 1:
        thresholds = torch.repeat_interleave(thresholds, og_shape[1])
    else:
        thresholds = thresholds.reshape(-1)

    def replacer(X, thresholds, value):
        # X.shape = (batch_size * num_queries, num_keys)
        # some broadcasting magic...
        mask = torch.arange(X.size(-1))[None, :] < thresholds[:, None]
        X[~mask] = value
        return X
    X = replacer(X.reshape(-1, og_shape[2]), thresholds, value).reshape(og_shape)
    return nn.functional.softmax(X, dim=2)



class LearnableDotProdAttention(nn.Module):
    def __init__(self, query_dim, key_dim, dropout=0, **kwargs):
        """
        Performs Learnable attention like so:
        dim1_softmax(Q @ W @ K) @ V, where W is the learnable weight matrix
        :param query_dim: dimension of queries;
        :param key_dim: dimension of keys;
        :param dropout: passed to nn.Dropout and applied to attention_weights, dropout defaults to 0;
        :param kwargs: kwargs passed to the constructor of nn.Module;
        """
        super(LearnableDotProdAttention, self).__init__(**kwargs)
        self.W = nn.Parameter(nn.init.xavier_uniform_(
            torch.zeros((query_dim, key_dim), dtype=torch.float32)), requires_grad=True)
        # want to save the attention_weights for performing attention_pooling of values;
        self.attention_weights = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, thresholds=None):
        # returns the pooled attention which has the shape:
        # (batch_size, num_queries, values_dim)

        # attention_weights.shape = (batch_size, num_queries, num_keys=num_values)
        self.attention_weights = masked_softmax(
            torch.bmm(queries @ self.W, keys.permute(0, 2, 1)), thresholds, value=-1e-6)
        return torch.bmm(self.dropout(self.attention_weights), values)


def fix_dim(X, num_heads):
    """
    :param X: X.shape = (batch_size, num_elements, rep_dim)
    :param num_heads: heads of MHAttention
    :return: X.shape = (batch_size * num_heads, num_elements, rep_dim // num_heads)
    """
    X = X.reshape(X.size(0), X.size(1), num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.size(2), X.size(3))


def reverse_dim(X, num_heads):
    """
    Reverse action of fix_dim;
    :param X: X.shape = (batch_size * num_heads, num_queries, rep_dim)
    :param num_heads: heads in MHAttention
    :return: X.shape = (batch_size, num_queries, rep_dim * num_heads)
    """
    X = X.reshape(-1, num_heads, X.size(1), X.size(2))
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.size(0), X.size(1), -1)


class MHAttention(nn.Module):
    def __init__(self, key_dim, query_dim, value_dim, rep_dim, num_heads,
                 dropout=0, with_bias=True, **kwargs):
        """

        :param key_dim: dimension of keys
        :param query_dim: dimension of queries
        :param value_dim: dimension of values
        :param rep_dim: must be divisible by num_heads;
        :param num_heads: heads for Multi-Head Attention;
        :param dropout: pass to nn.Dropout() defaults to 0;
        :param kwargs: to pass to constructor of nn.Module;
        """
        super(MHAttention, self).__init__(**kwargs)
        # this is to parallelise for gpus;
        self.num_heads = num_heads
        self.attention_pooling = LearnableDotProdAttention(rep_dim // num_heads, rep_dim // num_heads, dropout)
        self.W_q = nn.Linear(query_dim, rep_dim, bias=with_bias)
        self.W_k = nn.Linear(key_dim, rep_dim, bias=with_bias)
        self.W_v = nn.Linear(value_dim, rep_dim, bias=with_bias)
        self.W_o = nn.Linear(rep_dim, rep_dim, bias=with_bias)

    def forward(self, queries, keys, values, thresholds=None):
        # make queries, keys and valued of shape:
        # (batch_size * num_heads, num_q or num_k, rep_dim // num_heads)
        # for parallel comp;
        Q = fix_dim(self.W_q(queries), self.num_heads)
        K = fix_dim(self.W_k(keys), self.num_heads)
        V = fix_dim(self.W_v(values), self.num_heads)

        if thresholds is not None:
            thresholds = torch.repeat_interleave(thresholds, self.num_heads)
        encodings = reverse_dim(self.attention_pooling(Q, K, V, thresholds), self.num_heads)
        return self.W_o(encodings)
