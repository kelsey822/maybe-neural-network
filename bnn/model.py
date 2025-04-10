import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        # Variational parameters for weights and biases
        self.W1_mu = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W1_log_std = nn.Parameter(torch.full((hidden_dim, input_dim), -3.0))
        self.b1_mu = nn.Parameter(torch.zeros(hidden_dim))
        self.b1_log_std = nn.Parameter(torch.full((hidden_dim,), -3.0))
        
        self.W2_mu = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)
        self.W2_log_std = nn.Parameter(torch.full((output_dim, hidden_dim), -3.0))
        self.b2_mu = nn.Parameter(torch.zeros(output_dim))
        self.b2_log_std = nn.Parameter(torch.full((output_dim,), -3.0))
    
    def sample_weights(self):
        weights = {}
        for name in ['W1', 'b1', 'W2', 'b2']:
            mu = getattr(self, f"{name}_mu")
            log_std = getattr(self, f"{name}_log_std")
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            weights[name] = mu + std * eps
        return weights

    def forward(self, X, weights=None):
        if weights is None:
            weights = self.sample_weights()
        Z1 = F.linear(X, weights['W1'], weights['b1'])
        A1 = F.relu(Z1)
        Z2 = F.linear(A1, weights['W2'], weights['b2'])
        return F.log_softmax(Z2, dim=1)
    
    def kl_divergence(self):
        kl = 0
        for name in ['W1', 'b1', 'W2', 'b2']:
            mu = getattr(self, f"{name}_mu")
            log_std = getattr(self, f"{name}_log_std")
            std = torch.exp(log_std)
            kl += 0.5 * torch.sum(std**2 + mu**2 - 1 - 2 * log_std)
        return kl
