# calibration.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from models.cnn_model import ECGCNN
from models.personalize import personalize
from data.patient_data import load_patient


def get_probs(model, X, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def plot_calibration(record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_test, y_test = load_patient(record)

    # Global
    global_model = ECGCNN().to(device)
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device))
    g_probs = get_probs(global_model, X_test, device)

    # Personalized
    personal_model, _, _ = personalize(record)
    p_probs = get_probs(personal_model, X_test, device)

    # confidence = max probability
    g_conf = np.max(g_probs, axis=1)
    p_conf = np.max(p_probs, axis=1)

    g_correct = (np.argmax(g_probs, axis=1) == y_test).astype(int)
    p_correct = (np.argmax(p_probs, axis=1) == y_test).astype(int)

    g_true, g_pred = calibration_curve(g_correct, g_conf, n_bins=10)
    p_true, p_pred = calibration_curve(p_correct, p_conf, n_bins=10)

    plt.figure(figsize=(6,6))
    plt.plot(g_pred, g_true, marker='o', label="Global")
    plt.plot(p_pred, p_true, marker='o', label="Personalized")
    plt.plot([0,1],[0,1],'--', label="Perfect reliability")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration Curve Patient {record}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_calibration("233")
