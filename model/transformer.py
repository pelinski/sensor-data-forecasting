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
            out_size (int): Output sequence length. Defaults to 32. 
            num_heads (int, optional): Number of heads in the multihead attention. Defaults to 16.
            dim_feedforward (int, optional): Size of the feedforward network. Defaults to 256.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            seq_len (int, optional): Maximum sequence length. Defaults to 16.
            num_encoder_layers (int, optional): Number of encoder layers. Defaults to 7.
        """
        super(TransformerEncoder, self).__init__()

        self.d_model = kwargs.get("d_model", 64)
        self.embedding_size_src = kwargs.get("embedding_size_src", 8)
        self.embedding_size_tgt = kwargs.get("embedding_size_tgt", 8)
        self.out_size = kwargs.get("out_size", 32)
        self.num_heads = kwargs.get("num_heads", 16)
        self.dim_feedforward = kwargs.get("dim_feedforward", 256)
        self.dropout = kwargs.get("dropout", 0.1)
        self.seq_len = kwargs.get("seq_len", 16)
        self.num_encoder_layers = kwargs.get("num_encoder_layers", 7)
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.src_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            self.seq_len).to(self.device)

        self.in_linear = torch.nn.Linear(
            self.embedding_size_src, self.d_model, bias=True)
        self.pos_encoder = PositionalEncoding(
            self.d_model, self.seq_len, self.dropout)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_encoder_layers)
        self.out_linear = torch.nn.Linear(
            self.d_model, self.embedding_size_tgt)
        self.out_linear_proj = torch.nn.Linear(self.seq_len, self.out_size)

        self.init_weights(self.in_linear)
        self.init_weights(self.out_linear)
        self.init_weights(self.out_linear_proj)

    def init_weights(self, module, initrange=0.1):
        module.bias.data.zero_()
        module.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """Transformer Encoder forward pass

        Args:
            src (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, embedding_size_src)

        Returns:
            out (torch.Tensor): Torch Tensor of shape (batch_size, seq_len, embedding_size_tgt)
        """
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, embedding_size_src)

        src = self.in_linear(src)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)  # (seq_len, batch_size, d_model)
        memory = self.transformer_encoder(
            src, self.src_mask)  # (seq_len, batch_size, d_model)
        # (seq_len, batch_size, embedding_size_tgt)
        out = self.out_linear(memory)
        out = out.permute(1, 2, 0)  # (batch_size, embedding_size_tgt,seq_len)
        # (batch_size, embedding_size_tgt,out_size)
        out = self.out_linear_proj(out)
        # (batch_size, out_size, embedding_size_tgt)
        out = out.permute(0, 2, 1)

        return out

    def predict(self, src):
        """Transformer Encoder predict method

        Args:
            src (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, embedding_size_src)
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(src)
        return y


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        """Positional encoding constructor

        Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)

        Args:
            d_model (int): Model internal embeddings.
            seq_len (int): Maximum sequence length.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pos_enc = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(
            0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))   # (d_model/2)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pos_enc[:, 0::2] = torch.cos(position * div_term)
        else:
            pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)  # (1, seq_len, d_model)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        """Positional encoding forward pass

        Args:
            x (torch.Tensor): Torch tensor of shape (batch_size, seq_len, d_model)

        Returns:
            (torch.Tensor): Torch tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pos_enc  # (batch_size, seq_len, d_model)
        x = self.dropout(x)  # (batch_size, seq_len, d_model)

        return x
