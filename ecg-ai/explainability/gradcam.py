import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cnn_model import ECGCNN
from models.personalize import personalize
from data.patient_data import load_patient


def saliency_map(model, beat, device):

    model.eval()
    beat = beat.clone().detach().to(device)
    beat.requires_grad = True

    output = model(beat)
    pred_class = output.argmax(dim=1)

    loss = output[0, pred_class]
    loss.backward()

    saliency = beat.grad.abs().squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)

    return saliency, pred_class.item()


def visualize(record="233", beat_index=50):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, X_test, _ = load_patient(record)

    beat = torch.tensor(X_test[beat_index:beat_index+1]).to(device)

    # global model
    global_model = ECGCNN().to(device)
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device))
    g_map, g_pred = saliency_map(global_model, beat.clone(), device)

    # personalized model
    personal_model, _, _ = personalize(record)
    p_map, p_pred = saliency_map(personal_model, beat.clone(), device)

    signal = X_test[beat_index]

    plt.figure(figsize=(10,4))

    plt.subplot(2,1,1)
    plt.title(f"Global focus (class {g_pred})")
    plt.plot(signal)
    plt.imshow(g_map[np.newaxis,:], aspect="auto", alpha=0.5, cmap="jet")

    plt.subplot(2,1,2)
    plt.title(f"Personalized focus (class {p_pred})")
    plt.plot(signal)
    plt.imshow(p_map[np.newaxis,:], aspect="auto", alpha=0.5, cmap="jet")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
