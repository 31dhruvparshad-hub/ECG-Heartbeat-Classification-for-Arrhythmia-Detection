import torch
from sklearn.metrics import classification_report
from models.cnn_model import ECGCNN
from models.personalize import personalize
from data.patient_data import load_patient
from config.settings import CLASSES


def evaluate(record):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load same patient test set once
    _, _, X_test, y_test = load_patient(record)
    X_test_tensor = torch.tensor(X_test).to(device)

    # ----- Global model -----
    global_model = ECGCNN().to(device)
    global_model.load_state_dict(torch.load("global_model.pth", map_location=device))
    global_model.eval()

    with torch.no_grad():
        global_preds = global_model(X_test_tensor).argmax(dim=1).cpu().numpy()

    print("\nGLOBAL MODEL RESULTS")
    print(classification_report(y_test, global_preds, target_names=CLASSES))

    # ----- Personalized model -----
    personalized_model, _, _ = personalize(record)
    personalized_model.eval()

    with torch.no_grad():
        personal_preds = personalized_model(X_test_tensor).argmax(dim=1).cpu().numpy()

    print("\nPERSONALIZED MODEL RESULTS")
    print(classification_report(y_test, personal_preds, target_names=CLASSES, zero_division=0))


if __name__ == "__main__":
    evaluate("233")
