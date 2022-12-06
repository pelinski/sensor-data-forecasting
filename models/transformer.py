import torch
import math
import torch.nn.functional as F


class TransformerEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        """Transformer Encoder constructor

        Args:
            d_model (int, optional): Size of the model internal embeddings. Defaults to 64.
            embedding_size_src (int): Embedding size of source. Defaults to 8.
            embedding_size_tgt (int): Embedding size of target. Defaults to 8.
            num_heads (int, optional): Number of heads in the multihead attention. Defaults to 16.
            dim_feedforward (int, optional): Size of the feedforward network. Defaults to 256.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            max_seq_len (int, optional): Maximum sequence length. Defaults to 16.
            num_encoder_layers (int, optional): Number of encoder layers. Defaults to 7.
        """
        super(TransformerEncoder, self).__init__()

        self.d_model = kwargs.get("d_model", 64)
        self.embedding_size_src = kwargs.get("embedding_size_src", 8)
        self.embedding_size_tgt = kwargs.get("embedding_size_tgt", 8)
        self.num_heads = kwargs.get("num_heads", 16)
        self.dim_feedforward = kwargs.get("dim_feedforward", 256)
        self.dropout = kwargs.get("dropout", 0.2)
        self.max_len = kwargs.get("seq_len", 16)
        self.num_encoder_layers = kwargs.get("num_encoder_layers", 7)
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.InputLayerEncoder = InputLayer(
            self.embedding_size_src, self.d_model, self.dropout, self.max_len)
        self.Encoder = Encoder(
            self.d_model, self.num_heads, self.dim_feedforward, self.dropout, self.num_encoder_layers)
        self.OutputLayer = OutputLayer(self.embedding_size_tgt, self.d_model)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src):
        """Transformer Encoder forward pass

        Args:
            src (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, embedding_size_src)

        Returns:
            out (torch.Tensor): Torch Tensor of shape (batch_size, seq_len, embedding_size_tgt)
        """
        x = self.InputLayerEncoder(src)  # (batch_size, seq_len, d_model)
        memory = self.Encoder(x)  # (batch_size, seq_len, d_model)
        out = self.OutputLayer(memory)

        return out

    def predict(self, src):
        """Transformer Encoder predict method

        Args:
            src (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, embedding_size_src)

        Returns:
            y (torch.Tensor): Output torch tensor of shape (batch_size, seq_len, embedding_size_tgt)
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(src)
        return y


class InputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model, dropout, max_len):
        """ Input Layer constructor

        Args:
            embedding_size (int): Embedding size of source. 
            d_model (int):  Size of the model internal embeddings. 
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length.

        """

        super(InputLayer, self).__init__()

        self.max_len = max_len

        self.Linear = torch.nn.Linear(embedding_size, d_model, bias=True)
        self.ReLU = torch.nn.ReLU()
        self.PositionalEncoding = PositionalEncoding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """Input layer forward pass

        Args:
            src (torch.Tensor): Torch tensor of shape (batch_size, seq_len, embedding_size_src)

        Returns:
            (torch.Tensor): Torch tensor of shape (batch_size, seq_len, d_model)
        """
        # x = F.pad(src, (0,0,0,0,0,self.max_len - src.shape[0])).shape # pad to max len

        x = self.Linear(src)  # (batch_size, seq_len, d_model)
        x = self.ReLU(x)  # (batch_size, seq_len, d_model)
        out = self.PositionalEncoding(x)  # (batch_size, seq_len, d_model)

        return out


class Encoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, num_encoder_layers):
        """Encoder constructor

        Args:
            d_model (int): Size of the model internal embeddings.
            num_heads (int): Number of heads in the multihead attention.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
            num_encoder_layers (int): Number of encoder layers.
        """
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        """Encoder forward pass

        Args:
            src (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_size_src)

        Returns:
            (torch.Tensor): Output tensor of shape (batch_size, seq_len, d_model)
        """
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, embedding_size_src)
        out = self.Encoder(src)  # (seq_len, batch_size, d_model)
        out = out.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        return out


class OutputLayer(torch.nn.Module):
    def __init__(self, embedding_size, d_model):
        """Output Layer constructor

        Args:
            embedding_size (int):  Embedding size of source. 
            d_model (int): Size of the model internal embeddings.
        """
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.Linear = torch.nn.Linear(d_model, embedding_size, bias=True)

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoder_out):
        """Output Layer forward pass

        Args:
            encoder_out (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, d_model)

        Returns:
            (torch.Tensor): Torch tensor of shape (batch_size, seq_len, embedding_size_tgt)
        """
        y = self.Linear(encoder_out)

        return y


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        """Positional encoding constructor

        Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

        Args:
            d_model (int): Model internal embeddings.
            max_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float)  # (max_len)
        position = position.unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Positional encoding forward pass

        Args:
            x (torch.Tensor): Torch tensor of shape (batch_size, seq_len, d_model)

        Returns:
            (torch.Tensor): Torch tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(0), :]  # (batch_size, seq_len, d_model)
        x = self.dropout(x)  # (batch_size, seq_len, d_model)
        return x
