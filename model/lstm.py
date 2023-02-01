import math
import torch
import torch.nn as nn


class CustomLSTM(nn.Module):  # short version using matrices
    # https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
    def __init__(self, **kwargs):
        """LSTM

        Source: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091

        Args:
            seq_len (int): Sequence length
            out_size (int): Output sequence length
            dropout (float): Dropout rate
            input_size (int): Input size of the network
            hidden_size (int): Size of the hidden state
        """
        super().__init__()
        self.seq_len = kwargs.get("seq_len", 10)
        self.out_size = kwargs.get("out_size", 30)
        self.input_size = kwargs.get("input_size", 2)
        self.hidden_size = kwargs.get("hidden_size", 4)
        self.dropout_p = kwargs.get("dropout", 0.2)

        # matrices containing weights for input, hidden and bias for each of the 4 gates
        self.W = nn.Parameter(torch.Tensor(
            self.input_size, self.hidden_size*4))
        self.U = nn.Parameter(torch.Tensor(
            self.hidden_size, self.hidden_size*4))

        self.bias = nn.Parameter(torch.Tensor(self.hidden_size*4))
        self.relu = nn.ReLU()
        self.dense = nn.Linear(
            self.hidden_size, self.out_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        #init weights in out linear layer
        self.dense.bias.data.zero_()
        self.dense.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None, return_states=False):
        """LSTM forward pass

        Args:
            x (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, input_size)
            init_states (torch.Tensor, optional): Initial states for output of the network (h_t) and the long-term memory (c_t). Defaults to None.
            return_states (bool, optional): Returns hidden_state, (h_t, c_t) if set to True, otherwise returns only hidden_state . Defaults to False.

        Returns:
            if return_states is True:
                hidden_state (torch.Tensor), (h_t (torch.Tensor), c_t (torch.Tensor)): Hidden state, network output and long-term memory
            if return_states is False:
                hidden_state (torch.Tensor): Hidden state
        """
        (batch_size, seq_size, _) = x.shape
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device, non_blocking=True),
                        torch.zeros(batch_size, self.hidden_size).to(x.device, non_blocking=True))
            # (batch_size, hidden_size), (batch_size, hidden_size)
        else:
            h_t, c_t = init_states

        hsz = self.hidden_size

        for t in range(seq_size):
            x_t = x[:, t, :]  # (batch_size, input_size)
            # batch the computations into a single matrix multiplication
            # @ is for matrix multiplication
            gates = x_t@self.W + h_t@self.U + self.bias

            # input gate (batch_size, hidden_size)
            i_t = torch.sigmoid(gates[:, :hsz])
            # forget gate (batch_size, hidden_size)
            f_t = torch.sigmoid(gates[:, hsz:hsz*2])
            # candidate gate (batch_size, hidden_size)
            g_t = torch.tanh(gates[:, hsz*2:hsz*3])
            # output gate (batch_size, hidden_size)
            o_t = torch.sigmoid(gates[:, hsz*3:])

            c_t = f_t * c_t + i_t * g_t  # (batch_size, hidden_size)
            h_t = o_t * torch.tanh(c_t)  # (batch_size, hidden_size)

            # h_t -->(1, batch_size, hidden_size)
            hidden_seq.append(h_t.unsqueeze(0))
            # hidden_seq is a list of sequence_length items, each of shape (1, batch_size, hidden_size)

        # reshape hidden_seq
        # (sequence_length, batch_size, hidden_size)
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0,
                                          1).contiguous()  # (batch_size, sequence_length, hidden_size). contiguous returns a tensor contiguous in memory

        # (batch_size, seq_len, hidden_size)
        # out = self.relu(hidden_seq) 
        
        # last predicted output contains information of the previous outputs
        out = hidden_seq[:,-1,:] # (batch_size, hidden_size)

        # project hidden_size into out_size
        out= self.dense(out)  # (batch_size, out_size)
        
        # out= self.relu(out)  # (batch_size, out_size) 
        # out= self.dropout(out) # (batch_size, out_size)
        
        out= out.unsqueeze(2) # (batch_size, out_size, 1) --> for compatibility with transformer

        return out

    def predict(self, x, init_states=None, return_states=False):
        """LSTM predict method

        Args:
            x (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, embedding_size_src)
        """
        self.eval()
        with torch.no_grad():
            y = self.forward(x, init_states, return_states)
        return y
