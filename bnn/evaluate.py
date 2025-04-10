import torch

def evaluate(model, test_loader, device, samples=10):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            probs_sum = torch.zeros(data.size(0), 10).to(device)
            for _ in range(samples):
                weights = model.sample_weights()
                output = model(data, weights)
                probs_sum += output.exp()  # convert log probs to probs

            avg_probs = probs_sum / samples
            pred = avg_probs.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
