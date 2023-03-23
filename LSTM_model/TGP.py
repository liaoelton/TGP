import torch
from model import SeqClassifier

class TGPclass():
    def __init__(self) -> None:
        
        self.model = SeqClassifier(
            hidden_size = 256,
            num_layers = 5,
            dropout = 0.1,
            bidirectional = True,
            device = 'cuda'
        ).to('cuda')

   
        ckpt = torch.load("ckpt/output/best-model.pth")
        self.model.load_state_dict(ckpt)
    def predict(self , x):
        output_cls = self.model(x)
        return output_cls
