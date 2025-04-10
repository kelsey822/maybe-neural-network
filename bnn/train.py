import torch
import torch.nn.functional as F

def train_model(model, train_loader, optimizer, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # Flatten images (from [batch, 1, 28, 28] to [batch, 784])
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            weights = model.sample_weights()
            output = model(data, weights)
            
            nll = F.nll_loss(output, target)
            kl = model.kl_divergence() / len(train_loader.dataset)
            loss = nll + kl
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
