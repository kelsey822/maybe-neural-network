import matplotlib.pyplot as plt
import torch
import scipy.stats
import numpy as np

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_prediction_confidence(model, test_loader, samples=30):
    model.eval()
    confidences = []

    with torch.no_grad():
        for data, _ in test_loader:
            #data = data.to(device)
            data = data.view(data.size(0), -1).to(device)
            probs_sum = torch.zeros(data.size(0), 10).to(device)

            for _ in range(samples):
                weights = model.sample_weights()
                output = model(data, weights)
                probs_sum += output.exp()

            avg_probs = probs_sum / samples
            max_conf = avg_probs.max(dim=1).values
            confidences.extend(max_conf.cpu().numpy())

    plt.hist(confidences, bins=20, range=(0, 1), color='steelblue', edgecolor='black')
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Max Predicted Probability")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.show()

def plot_entropy_distribution(model, test_loader, samples=30):
    model.eval()
    entropies = []

    with torch.no_grad():
        for data, _ in test_loader:
            #data = data.to(device)
            data = data.view(data.size(0), -1).to(device)
            probs_sum = torch.zeros(data.size(0), 10).to(device)

            for _ in range(samples):
                weights = model.sample_weights()
                output = model(data, weights)
                probs_sum += output.exp()

            avg_probs = (probs_sum / samples).cpu().numpy()
            entropy = scipy.stats.entropy(avg_probs.T)  # shape: (batch,)
            entropies.extend(entropy)

    plt.hist(entropies, bins=30, color='darkorange', edgecolor='black')
    plt.title("Predictive Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.show()

def test_prediction(index, model, test_dataset, num_samples=30):
    model.eval()

    image, label = test_dataset[index]
    image = image.view(1, -1).to(device)  # shape: (1, 784)

    probs_sum = torch.zeros((1, 10)).to(device)
    with torch.no_grad():
        for _ in range(num_samples):
            weights = model.sample_weights()
            output = model(image, weights)  # log-probs
            probs_sum += output.exp()       # convert to probs

    avg_probs = probs_sum / num_samples
    entropy = scipy.stats.entropy(avg_probs.cpu().numpy()[0], base=2)
    prediction = torch.argmax(avg_probs, dim=1).item()

    print("Prediction:", prediction)
    print("Label:     ", label)
    print("Entropy:   {:.4f}".format(entropy))

    img = image.view(28, 28).cpu().numpy() * 255
    plt.gray()
    plt.imshow(img, interpolation='nearest')
    plt.show()


def predict_with_uncertainty(model, x, n_iter=100):
    model.train()  
    preds = []
    for _ in range(n_iter):
        pred = model(x)
        preds.append(pred.unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    mean_prediction = preds.mean(dim=0)
    std_prediction = preds.std(dim=0)
    return mean_prediction, std_prediction


def visualize_prediction_uncertainty(image, mean_pred, std_pred, true_label=None):
    probs = torch.softmax(mean_pred, dim=0).detach().cpu().numpy()

    errors = std_pred.detach().cpu().numpy()

    errors = np.clip(errors, 0.0, 0.25)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    if image.dim() == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    x = np.arange(len(probs))
    axs[1].bar(x, probs, yerr=errors, capsize=5)
    axs[1].set_xticks(x)
    axs[1].set_ylim([0, 1])  
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Predicted Probability")
    title = "Predictive Uncertainty"
    if true_label is not None:
        title += f" (True label: {true_label})"
    axs[1].set_title(title)

    plt.tight_layout()
    plt.show()




























