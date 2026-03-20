import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cnn_model import ECGCNN
from data.patient_data import load_patient_partial


def compute_ece(probs, labels, bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0
    bin_boundaries = np.linspace(0, 1, bins+1)

    for i in range(bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(labels)

    return ece


def fine_tune(seconds, record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_adapt, y_adapt, X_test, y_test = load_patient_partial(record, seconds)

    model = ECGCNN().to(device)
    model.load_state_dict(torch.load("global_model.pth", map_location=device))

    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    X_adapt = torch.tensor(X_adapt).to(device)
    y_adapt = torch.tensor(y_adapt).to(device)

    model.train()
    for _ in range(5):
        preds = model(X_adapt)
        loss = loss_fn(preds, y_adapt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return compute_ece(probs, y_test)


def run(record="233"):

    times = [10, 30, 60, 120, 300, 600]

    eces = []
    for t in times:
        e = fine_tune(t, record)
        eces.append(e)
        print(t, "sec -> ECE:", e)

    plt.plot(times, eces, marker='o')
    plt.xlabel("Personalization time (seconds)")
    plt.ylabel("Calibration Error (ECE)")
    plt.title(f"Overfitting vs Personalization Patient {record}")
    plt.show()


if __name__ == "__main__":
    run()
