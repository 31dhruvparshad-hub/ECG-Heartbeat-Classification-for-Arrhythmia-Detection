# noise_test.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cnn_model import ECGCNN
from models.personalize import personalize
from data.patient_data import load_patient


def add_noise(X, level):
    noise = np.random.normal(0, level, X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)



def accuracy(model, X, y, device):
    with torch.no_grad():
        preds = model(torch.tensor(X).to(device)).argmax(dim=1).cpu().numpy()
    return np.mean(preds == y)


def noise_experiment(record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_test, y_test = load_patient(record)

    # Load models
    global_model = ECGCNN().to(device)
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device))
    global_model.eval()

    personal_model, _, _ = personalize(record)
    personal_model.eval()

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

    g_acc = []
    p_acc = []

    for n in noise_levels:
        X_noisy = add_noise(X_test, n)

        g_acc.append(accuracy(global_model, X_noisy, y_test, device))
        p_acc.append(accuracy(personal_model, X_noisy, y_test, device))

    plt.figure(figsize=(6,4))
    plt.plot(noise_levels, g_acc, marker='o', label="Global")
    plt.plot(noise_levels, p_acc, marker='o', label="Personalized")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy")
    plt.title(f"Noise Robustness Patient {record}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    noise_experiment("233")
