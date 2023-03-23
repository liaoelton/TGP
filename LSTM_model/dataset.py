from typing import List, Dict

from torch.utils.data import Dataset

# from utils import Vocab, pad_to_len

import torch
# from transformers import BertTokenizer, BertModel
# Importing the relevant modules
# from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        max_len: int,
    ):
        self.data = data
        self.max_len = max_len

        self.token2idx = {
            -5 : 1,
            -4 : 2,
            -3 : 3,
            -2 : 4,
            -1 : 5,
            0  : 0,
            1  : 6,
            2  : 7,
            3  : 8, 
            4  : 9,
        }

    def _token2idx( self, program ):

        return [ self.token2idx[p]  for p in program]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    # @property
    # def num_classes(self) -> int:
    #     return len(self.label_mapping)

    

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # { ["input": [0,1,2,4,1] , "output": 12]      data #1
        #   ["input": [0,1,2,4,2] , "output": 14]  }   data #2
        samples.sort(key=lambda x: len(x['program']), reverse=True)
        batch = {}
        
        
    
        batch['program'] = []
        # concat input x
        # batch['program'] += np.linspace(-10, 10, num=50).tolist()
        batch['len'] = []
        lens = [len(s['program']) for s in samples]
       
        batch['len'] = torch.tensor([len(s['program']) for s in samples]) 
        batch['program'] = [s['program']+[0] * (max(lens) - len(s['program'])) for s in samples]
        batch['program'] = [ self._token2idx(b) for b in batch['program']]
        # print(batch['program'])
        batch['program'] = torch.tensor(batch['program'], dtype=torch.float)

        
        batch['output'] = [ abs( max(s['output']) - min(s['output'])) for s in samples    ]
        for i in range(len(batch['output'] )):
            if batch['output'][i] < 2:
                batch['output'][i] = 2
            elif batch['output'][i] > 2:
                batch['output'][i] = 0
            else:
                batch['output'][i] = 1
        # batch['id'] = [s['id'] for s in samples]
        if 'output' in samples[0].keys():
            batch['output'] = torch.tensor(batch['output'])
        else:
            batch['output'] = torch.zeros(len(samples), dtype=torch.long)

        return batch

    

 


