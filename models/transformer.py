import torch
import math

class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, embedding_size_src, embedding_size_tgt, num_heads, dim_feedforward, dropout,
                 num_encoder_layers, max_len, device):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding_size_src = embedding_size_src
        self.embedding_size_tgt = embedding_size_tgt
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len
        self.num_encoder_layers = num_encoder_layers
        self.device = device

        self.InputLayerEncoder = InputLayer(embedding_size_src, d_model, dropout, max_len)
        self.Encoder = Encoder(d_model, num_heads, dim_feedforward, dropout, num_encoder_layers)
        self.OutputLayer = OutputLayer(embedding_size_tgt, d_model)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src):
        # src  N x seq_len x embedding_size_src
        x = self.InputLayerEncoder(src)  #  N x seq_len x d_model
        memory = self.Encoder(x)  #  N x seq_len x d_model
        out = self.OutputLayer(memory)  # N x seq_len x embedding_size_tgt

        return out

    def predict(self, src, use_thres=True, thres=0.5, use_pd=False):
        self.eval()
        with torch.no_grad():
            y = self.forward(src)  #  N x seq_len x embedding_size_src

        return y
    

class Encoder(torch.nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # seq_len x N x d_model 
        out = self.Encoder(src)  # seq_len x N x d_model 
        out = out.permute(1, 0, 2)  #  N x seq_len x d_model
        
        return out
       

class InputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model, dropout, max_len):
        super(InputLayer, self).__init__()

        self.Linear = torch.nn.Linear(embedding_size, d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        x = self.Linear(src)
        x = self.ReLU(x)
        out = self.PositionalEncoding(x)

        return out


class OutputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model):
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        # y = torch.reshape(y, (decoder_out.shape[0], decoder_out.shape[1], 3, self.embedding_size // 3))

        # _h = y[:, :, 0, :]
        # _v = y[:, :, 1, :]
        # _o = y[:, :, 2, :]

        # h = _h
        # v = torch.sigmoid(_v)
        # o = torch.tanh(_o) * 0.5

        return y
    
    
    
class PositionalEncoding(torch.nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2!= 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)