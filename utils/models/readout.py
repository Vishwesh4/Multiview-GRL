import torch

class Readout(torch.nn.Module):
    """
    Performs average mean pooling of positive as well as negative entities with mask separately
    """
    def __init__(self) -> None:
        super().__init__()
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, x):
        return self.sig(torch.mean(x,0))