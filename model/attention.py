import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import get_init_weights


class Attention(nn.Module):
    def __init__(self, seq_len, embedding_size, hidden_size, n_layers, num_directions, mode='basic'):
        super(Attention, self).__init__()
        self.mode = mode
        
        if mode == 'basic':
            self.fc = nn.Linear(embedding_size + hidden_size * n_layers * num_directions, seq_len, bias=False)
        elif mode == 'soft':
            self.W1 = get_init_weights((hidden_size, hidden_size))
#             self.W2 = get_init_weights((embedding_size, hidden_size))
            self.W3 = get_init_weights((hidden_size, seq_len))
            self.b = Parameter(torch.zeros(hidden_size))
            
        self.__init_layers()
            
            
    def __init_layers(self):
        if self.mode == 'basic':
            nn.init.xavier_normal_(self.fc.weight)
            

    def __original_fn(self, inputs, hiddens, embeddings):
        """
        inputs: (batch_size x seq_len x num_directions * hidden_size)
        """
        n_layers_dirs = hiddens.size(0)
        
        # (batch_size x hidden_size * n_layers * num_directions)
        hiddens = torch.cat([hiddens[i,:,:] for i in range(n_layers_dirs)], dim=1)
        
        # (batch_size x embedding_size + hidden_size * n_layers * num_directions)
        concat = torch.cat((embeddings, hiddens), dim=1)
        
        # (batch_size x hidden_size * n_layers * num_directions) -> (batch_size x 1 x seq_len)
        attn_weights = torch.softmax(self.fc(concat), dim=1).unsqueeze(1)

        # (batch_size x 1 x num_directions * hidden_size)
        attn_applied = torch.bmm(attn_weights, inputs) / attn_weights.size(2)
        
        # (batch_size x 1 x num_directions * hidden_size) -> (batch_size x hidden_size)
        return attn_applied.squeeze(1)
    

    def __soft_fn(self, inputs, hiddens):
        attn_weights = torch.softmax((hiddens @ self.W1 + self.b) @ self.W3, dim=1).unsqueeze(1)
        return (torch.bmm(attn_weights, inputs) / attn_weights.size(2)).squeeze(1)
    

    def forward(self, inputs, hiddens, embeddings=None):
        if self.mode == 'basic':
            return self.__original_fn(inputs, hiddens, embeddings)
        elif self.mode == 'soft':
            return self.__soft_fn(inputs, hiddens)
        raise 'Not implemented mode'