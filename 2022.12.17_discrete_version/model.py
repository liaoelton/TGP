from typing import Dict
import torch
from torch import nn
from torch.nn import Embedding, functional
from torch.autograd import Variable

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        device,
    ) -> None:
        super(SeqClassifier, self).__init__()
       # self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.dropout = dropout
        self.bidirectional = bidirectional
       
        
        self.embed = nn.Embedding(10, 500 ).to(device)  # 10 token will appear , 100 dim
        
        self.rnn = nn.LSTM(
            #self.embed.size(1),
            self.embed.embedding_dim,
            hidden_size,
            num_layers,
            dropout = dropout,
            bidirectional = False,
            batch_first = True,
        ).to(device)

        self.attn = nn.MultiheadAttention(
            1000,
            10
        ).to(device)
        
        num_directions = 2 if bidirectional else 1
        self.rnn_output_dim = hidden_size *2
        self.hidden_state_dim = num_layers * num_directions * hidden_size
        self.classfier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear( self.rnn_output_dim, 3),
        ).to(device)
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        x, y = batch['program'], batch['output'] # [batch_size, max_len]
    
        #print( batch['program'] )
        
        x = self.embed(x.long())
        
        res = {}
        sh = x.shape
        #print(x.reshape((sh[0],-1,self.embed.embedding_dim)).shape)
        #
        
        packed_x = nn.utils.rnn.pack_padded_sequence(x.reshape((sh[0],-1,self.embed.embedding_dim)), batch['len'], batch_first=True , enforce_sorted=False)
        self.rnn.flatten_parameters()
        
       
        # for lstm
        x, (h, _) = self.rnn(packed_x)
        # for gru
        # x, h = self.rnn(packed_x)
       
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True) # [batch_size, max_len, hid_dim]
        #print(x.shape)
        # if self.bidirectional:
        #     h = torch.cat((h[-1], h[-2]), axis=-1) 
        # else:
        #     h = h[-1]
        #print(x.shape)
        #y_hat = self.classfier(x[ :,:32,:])
        """
        attn_output, attn_output_weights = self.attn(x,x,x)
        y_hat = [self.classfier(attn_output)]
        """
        #print(y_hat)
      
        if self.bidirectional:
            h = torch.cat((h[-1], h[-2]), axis=-1) 
        else:
            h = h[-1]

        y_hat = [self.classfier(h)]
        y_pred = y_hat[-1].max(1, keepdim=True)[1].reshape(-1)
        res["y_pred"] =  y_pred
 
        #print(y_pred.shape, y.shape)
        #print(y_pred , y_pred)
        #print(y_hat[-1],y)
         
        if y is not None:
           
            loss = functional.cross_entropy( y_hat[-1] , y)
            # for b in range(len(y_pred)):
            #     for i in range(len(y_pred[0])):
            #         loss += torch.abs(y_pred[b][i]- y[b][i])
            # #print(loss)
            # loss = Variable(torch.tensor(loss).float() ,requires_grad=True)
            # l1 = torch.nn.L1Loss()
            # loss = l1() 
            # print(loss)
            # loss.requires_grad_(True)
            # loss.float()
        else:
            print("QQ")
            loss  = 0
        res["loss"] = loss
        # res["y_pred"] = y_hat[-1].max(1, keepdim=True)[1].reshape(-1)
        return res


