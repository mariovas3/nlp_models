import torch
from torch import nn


def masked_softmax(X, attention_masks, value=-1e6):
    """
    Text seqs of variable lengths, so there is going to be some padding, hence take only the
    relevant tokens not the <pad> tokens when doing softmax;
    :param X: X.shape = (batch_size, num_queries, num_keys=num_values)
    :param attention_masks: 1D or 2D torch.Tensor saying from which position to stop counting;
    :param value: replace entries of X with that value before softmax;
    :return: same dimension torch.Tensor as X, but undergone a more careful softmax procedure;
    """
    if attention_masks is None:
        return nn.functional.softmax(X, dim=2)

    og_shape = X.shape
    if attention_masks.dim() == 1:
        # for each query mask the interaction with the same keys >attention_masks;
        attention_masks = torch.repeat_interleave(attention_masks, og_shape[1])
    else:
        # or you have specified which query-key interaction to mask for each query;
        attention_masks = attention_masks.reshape(-1)

    def replacer(X, attention_masks, value):
        # X.shape = (batch_size * num_queries, num_keys)
        # some broadcasting magic...
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = torch.arange(X.size(-1), device=device)[None, :] < attention_masks[:, None]
        X[~mask] = value
        return X
    X = replacer(X.reshape(-1, og_shape[2]), attention_masks, value).reshape(og_shape)
    return nn.functional.softmax(X, dim=2)



class LearnableDotProdAttention(nn.Module):
    def __init__(self, query_dim, key_dim, dropout=0, **kwargs):
        """
        Performs Learnable attention like so:
        dim1_softmax(Q @ W @ K.T) @ V, where W is the learnable weight matrix
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

    def forward(self, queries, keys, values, attention_masks=None):
        # returns the pooled attention which has the shape:
        # (batch_size, num_queries, values_dim)

        # attention_weights.shape = (batch_size, num_queries, num_keys=num_values)
        # basically for each query gets a distribution over its interaction with each key and
        # some learnable weight;
        self.attention_weights = masked_softmax(
            torch.bmm(queries @ self.W, keys.permute(0, 2, 1)), attention_masks, value=-1e-6)
        # for each query get a weighted sum of values based on the distribution
        # specified in self.attention_weights;
        return torch.bmm(self.dropout(self.attention_weights), values)


def allocate_to_heads(X, num_heads):
    """
    :param X: X.shape = (batch_size, num_queries, hidden_dim)
    :param num_heads: count of heads of MHAttention
    :return: X.shape = (batch_size * num_heads, num_queries, hidden_dim // num_heads)
    """
    X = X.reshape(X.size(0), X.size(1), num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.size(2), X.size(3))


def combine_heads(X, num_heads):
    """
    Reverse action of allocate_to_heads;
    :param X: X.shape = (batch_size * num_heads, num_queries, hidden_dim_per_head)
    :param num_heads: count of heads in MHAttention
    :return: X.shape = (batch_size, num_queries, hidden_dim_per_head * num_heads)
    """
    X = X.reshape(-1, num_heads, X.size(1), X.size(2))
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.size(0), X.size(1), -1)


class MHAttention(nn.Module):
    def __init__(self, key_dim, query_dim, value_dim, hidden_dim, num_heads,
                 dropout=0, with_bias=True, **kwargs):
        """

        :param key_dim: dimension of keys
        :param query_dim: dimension of queries
        :param value_dim: dimension of values
        :param hidden_dim: must be divisible by num_heads;
        :param num_heads: heads for Multi-Head Attention;
        :param dropout: pass to nn.Dropout() defaults to 0;
        :param kwargs: to pass to constructor of nn.Module;
        """
        super(MHAttention, self).__init__(**kwargs)
        # this is to parallelise for gpus;
        self.num_heads = num_heads
        self.attention_pooling = LearnableDotProdAttention(hidden_dim // num_heads, hidden_dim // num_heads, dropout)
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=with_bias)
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=with_bias)
        self.W_v = nn.Linear(value_dim, hidden_dim, bias=with_bias)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=with_bias)

    def forward(self, queries, keys, values, attention_masks=None):
        # make queries, keys and valued of shape:
        # (batch_size * num_heads, num_q or num_k, hidden_dim // num_heads)
        # for parallel comp;
        Q = allocate_to_heads(self.W_q(queries), self.num_heads)
        K = allocate_to_heads(self.W_k(keys), self.num_heads)
        V = allocate_to_heads(self.W_v(values), self.num_heads)

        if attention_masks is not None:
            attention_masks = torch.repeat_interleave(attention_masks, self.num_heads)
        encodings = combine_heads(self.attention_pooling(Q, K, V, attention_masks), self.num_heads)
        # returns tensor of shape (batch_size, num_queries, hidden_dim);
        return self.W_o(encodings)


class AddNorm(nn.Module):
    def __init__(self, norm_dim, dropout=0, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.norm = nn.LayerNorm(norm_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, X_res):
        return self.norm(X+self.dropout(X_res))


class EncBlockFFN(nn.Module):
    def __init__(self, ffn_input, ffn_hidden, ffn_output, **kwargs):
        super(EncBlockFFN, self).__init__(**kwargs)
        self.net = nn.Sequential(
            nn.Linear(ffn_input, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, ffn_output)
        )

    def forward(self, X):
        return self.net(X)


class EncoderBlock(nn.Module):
    def __init__(self, key_dim, query_dim, value_dim, hidden_dim, num_heads,
                 norm_dim, ffn_input, ffn_hidden, dropout=0, with_bias=True, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MHAttention(key_dim, query_dim, value_dim, hidden_dim, num_heads,
                                     dropout, with_bias, **kwargs)
        self.add_norm1 = AddNorm(norm_dim, dropout)
        self.ffn = EncBlockFFN(ffn_input, ffn_hidden, hidden_dim)
        self.add_norm2 = AddNorm(norm_dim, dropout)

    def forward(self, X, attention_masks):
        # since it is Self-attention, query, key and value are all the same;
        # returns tensor of shape (batch_size, query_dim, hidden_dim);
        X_res = self.add_norm1(X, self.attention(X, X, X, attention_masks))
        return self.add_norm2(X_res, self.ffn(X_res))
