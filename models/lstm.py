# https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
import math
import torch
import torch.nn as nn

    
class CustomLSTM(nn.Module): # short version using matrices
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # matrices containing weights for input, hidden and bias for each of the 4 gates
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size*4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size*4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size*4))
        self.init_weights()
        
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self,x,init_states=None):
        """
        assumes x is of shape (batch_size, sequence_size, input_size)
        """
        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device, non_blocking=True),
                        torch.zeros(batch_size, self.hidden_size).to(x.device, non_blocking=True))
            # (batch_size, hidden_size), (batch_size, hidden_size)
        else:
            h_t, c_t = init_states
            
        hsz = self.hidden_size
        
        for t in range(seq_size):
            x_t = x[:, t, :] # (batch_size, input_size)
            # batch the computations into a single matrix multiplication
            gates = x_t@self.W + h_t@self.U + self.bias             # @ is for matrix multiplication
            
            i_t = torch.sigmoid(gates[:, :hsz]) # input gate (batch_size, hidden_size)
            f_t = torch.sigmoid(gates[:, hsz:hsz*2]) # forget gate (batch_size, hidden_size)
            g_t = torch.tanh(gates[:, hsz*2:hsz*3]) # candidate gate (batch_size, hidden_size)
            o_t = torch.sigmoid(gates[:, hsz*3:]) # output gate (batch_size, hidden_size)
            
            c_t = f_t * c_t + i_t * g_t  # (batch_size, hidden_size)
            h_t = o_t * torch.tanh(c_t)  # (batch_size, hidden_size)
            
            hidden_seq.append(h_t.unsqueeze(0))  # h_t -->(1, batch_size, hidden_size)
                # hidden_seq is a list of sequence_length items, each of shape (1, batch_size, hidden_size)

        # reshape hidden_seq
        hidden_seq = torch.cat(
            hidden_seq,
            dim = 0
        )  # (sequence_length, batch_size, hidden_size)
        hidden_seq = hidden_seq.transpose(0,
                                          1).contiguous()  #(batch_size, sequence_length, hidden_size). contiguous returns a tensor contiguous in memory

        return hidden_seq, (h_t, c_t)
    