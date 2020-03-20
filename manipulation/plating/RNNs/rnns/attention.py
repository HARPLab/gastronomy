import torch
import torch.nn as nn
import torch.functional as F

class Attention(nn.Module):
    """
    Used the IBM github code as reference: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
    License:  <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__
    """

    def __init__(self, dims):
        """
        dims(int): The number of expected features in the output
        """
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dims*2, dims)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        """
        Inputs:
            output - (batch, output_len, dimensions): tensor containing the output 
                     features from the decoder
            context -(batch, input_len, dimensions): tensor containing features of 
                     the encoded input sequence

        Outputs:
            output - (batch, output_len, dimensions): tensor containing the 
                     attended output features from the decoder
            attn - (batch, output_len, input_len): tensor containing attention weights
        """
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn