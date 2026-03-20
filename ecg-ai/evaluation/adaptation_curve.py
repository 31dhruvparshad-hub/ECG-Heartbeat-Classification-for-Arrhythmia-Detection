import torch
import numpy as np
import matplotlib.pyplot as plt
from models.cnn_model import ECGCNN
from data.patient_data import load_patient_partial


def fine_tune(seconds, record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_adapt, y_adapt, X_test, y_test = load_patient_partial(record, seconds)

    model = ECGCNN().to(device)
    model.load_state_dict(torch.load("global_model.pth", map_location=device))

    # freeze feature extractor
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    X_adapt = torch.tensor(X_adapt).to(device)
    y_adapt = torch.tensor(y_adapt).to(device)

    for _ in range(5):
        preds = model(X_adapt)
        loss = loss_fn(preds, y_adapt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    preds = np.argmax(probs, axis=1)
    acc = np.mean(preds == y_test)
    conf = np.mean(np.max(probs, axis=1))

    return acc, conf


def run(record="233"):

    times = [10, 20, 40, 60, 120]

    accs = []
    confs = []

    for t in times:
        a, c = fine_tune(t, record)
        accs.append(a)
        confs.append(c)
        print(t, "sec -> acc:", a, "conf:", c)

    plt.plot(times, accs, marker='o', label="Accuracy")
    plt.plot(times, confs, marker='o', label="Confidence")
    plt.xlabel("Personalization time (seconds)")
    plt.ylabel("Score")
    plt.title(f"Adaptation Curve Patient {record}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
