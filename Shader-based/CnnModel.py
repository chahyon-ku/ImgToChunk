import torch

class CnnModel(torch.nn.Module):
    def __init__(self, d_out):
        super(CnnModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 16, 3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1, padding='same'),
        )

    def forward(self, x):
        return self.model(x)
