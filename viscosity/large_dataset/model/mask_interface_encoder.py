import torch
import torch.nn as nn

class MaskInterfaceEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_points=64):
        """
        Encoder for fluid interface represented as a sequence of y-coordinates
        
        Args:
            embedding_dim: Output embedding dimension
            num_points: Number of x-axis points sampled along the interface
        """
        super().__init__()
        self.num_points = num_points
        
        # MLP to encode the interface points
        self.encoder = nn.Sequential(
            nn.Linear(num_points, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, interfaces):
        """
        Encode interface sequences
        
        Args:
            interfaces: Tensor of shape (B, T, num_points)
            
        Returns:
            Tensor of shape (B, T, embedding_dim)
        """
        B, T, _ = interfaces.shape
        
        # Process each time step
        embeddings = []
        for t in range(T):
            # Encode the interface vector
            embedding = self.encoder(interfaces[:, t])  # (B, embedding_dim)
            embeddings.append(embedding)
            
        # Stack along time dimension
        embeddings = torch.stack(embeddings, dim=1)  # (B, T, embedding_dim)
        
        return embeddings