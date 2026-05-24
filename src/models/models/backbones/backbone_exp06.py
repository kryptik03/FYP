import torch
import torch.nn as nn

class DECCNN2D(nn.Module):
    """
    Lightweight 2D CNN designed to process (1, Freq, Time) STFT magnitude spectrograms.
    Extracts features for both SimCLR contrastive learning and DEC clustering.
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 16, embedding_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global Average Pooling to collapse spatial dimensions
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projector = nn.Sequential(
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 4, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, 1, Freq, Time)
        Returns:
            z: L2-normalized embedding of shape (Batch, embedding_dim)
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1) # Flatten
        z = self.projector(features)
        
        # L2 Normalize for DEC and SimCLR stability
        z = nn.functional.normalize(z, p=2, dim=1)
        return z
