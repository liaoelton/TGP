from typing import Dict
import torch
from torch import nn
from torch.nn import Embedding, functional


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        # num_class: int,
        device,
    ) -> None:
        super(SeqClassifier, self).__init__()
       # self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.dropout = dropout
        self.bidirectional = bidirectional
        # self.numClasses = num_class

        
      
        self.rnn = nn.LSTM(
           # embeddings.size(1),
            # 1,
            1, 
            hidden_size,
            num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True,
        ).to(device)
        num_directions = 2 if bidirectional else 1
        self.rnn_output_dim = num_directions * hidden_size
        self.hidden_state_dim = num_layers * num_directions * hidden_size
        self.classfier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.rnn_output_dim, 50)
        ).to(device)
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x, y = batch['program'], batch['output'] # [batch_size, max_len]
        #print(x)
        #x = self.embed(x)
       
        res = {}
        sh = x.shape
        packed_x = nn.utils.rnn.pack_padded_sequence(x.reshape((sh[0],-1,1)), batch['len'], batch_first=True , enforce_sorted=False)
        self.rnn.flatten_parameters()
        # for lstm
        x, (h, _) = self.rnn(packed_x)
        # for gru
        # x, h = self.rnn(packed_x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) # [batch_size, max_len, hid_dim]
        if self.bidirectional:
            h = torch.cat((h[-1], h[-2]), axis=-1) 
        else:
            h = h[-1]
        y_hat = [self.classfier(h)]
        if y is not None:
            l = nn.MSELoss()
            loss = l(y_hat[-1],y)
        else:
            loss  = 0
        res["loss"] = loss
        # res["y_pred"] = y_hat[-1].max(1, keepdim=True)[1].reshape(-1)
        return res


