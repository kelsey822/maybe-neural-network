import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from bnn.model import BayesianNN
from bnn.train import train_model
from bnn.evaluate import evaluate
from bnn.utils import (
    plot_prediction_confidence,
    plot_entropy_distribution,
    test_prediction,
    predict_with_uncertainty,
    visualize_prediction_uncertainty
)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, optimizer, etc.
model = BayesianNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_model(model, train_loader, optimizer, device, num_epochs=20)

# Evaluate the model
evaluate(model, test_loader, device, samples=10)

# Visualize prediction confidence across the test set
plot_prediction_confidence(model, test_loader, samples=30)
plot_entropy_distribution(model, test_loader, samples=30)

# Test prediction on a single image with entropy and visualization
test_prediction(index=7, model=model, test_dataset=test_dataset)

# Get a sample image and label
image, label = test_dataset[7]
image_input = image.view(1, -1).to(device)

# Get predictive distribution (mean and std)
mean_pred, std_pred = predict_with_uncertainty(model, image_input, n_iter=100)

# Visualize with error bars
visualize_prediction_uncertainty(image.squeeze(), mean_pred[0], std_pred[0], true_label=label)
