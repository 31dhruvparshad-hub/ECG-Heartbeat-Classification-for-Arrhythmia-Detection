# dashboard.py (final improved v2)
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from models.cnn_model import ECGCNN
from models.personalize import personalize
from config.settings import CLASSES
from explainability.gradcam import saliency_map
from data.patient_split import DS2
from data.patient_data import load_patient
# custom calibration (multi-class reliability)
from sklearn.calibration import calibration_curve

# ---------------- UI STYLE (LIGHT THEME) ----------------
st.set_page_config(page_title="ECG AI Reliability Demo", layout="wide")

st.markdown("""
<style>
html, body {background-color:#f7f9fc;}
h1 {font-size:40px !important; color:#0a2540}
h2 {font-size:28px !important;}
h3 {font-size:22px !important;}
.stButton>button {background-color:#2e7d32;color:white;border-radius:8px;font-size:18px;padding:8px 18px;}
</style>
""", unsafe_allow_html=True)

st.title("🫀 ECG AI Trustworthiness Demonstrator")
st.caption("Global vs Personalized Arrhythmia Detection")

# ---------------- MODELS ----------------
@st.cache_resource
def load_global_model():
    device = torch.device("cpu")
    model = ECGCNN().to(device)
    model.load_state_dict(torch.load("global_model.pth", map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_personalized_model(patient):
    model, _, _ = personalize(patient)
    model.eval()
    return model


def predict_probs(model, X):
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(X)), dim=1).cpu().numpy()
    return probs


def explanation_plot(model, beat, title):
    device = torch.device("cpu")
    beat_tensor = torch.tensor(beat).unsqueeze(0).to(device)
    saliency, _ = saliency_map(model, beat_tensor.clone(), device)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(beat, linewidth=2)
    ax.imshow(saliency[np.newaxis,:], aspect="auto", alpha=0.5, cmap="plasma")
    ax.set_title(title)
    ax.tick_params(labelsize=12)
    return fig

# ---------------- DATA ----------------
patient = st.selectbox("Select Patient", DS2)
X_adapt, y_adapt, X_test, y_test = load_patient(patient)

colA, colB = st.columns([3,1])
beat_index = colA.slider("Select heartbeat", 0, len(X_test)-1, 50)
manual_index = colB.number_input("Type beat #", min_value=0, max_value=len(X_test)-1, value=beat_index, step=1)

if manual_index != beat_index:
    beat_index = manual_index

beat = X_test[beat_index]

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs(["ECG Viewer", "Diagnosis", "Explainability", "Reliability Metrics"])

# ---- TAB 1 ECG VIEWER ----
with tab1:
    st.subheader("Heartbeat Signal")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(beat, linewidth=2)
    ax.set_title(f"Patient {patient} - Beat {beat_index}")
    ax.tick_params(labelsize=12)
    st.pyplot(fig)

    st.write(f"Adaptation beats: {len(X_adapt)}")
    st.write(f"Evaluation beats: {len(X_test)}")

# ---- TAB 2 DIAGNOSIS ----
with tab2:
    st.subheader("AI Diagnosis")
    col1, col2 = st.columns(2)

    if col1.button("Run Global Diagnosis"):
        model = load_global_model()
        probs = predict_probs(model, beat[np.newaxis,:])
        pred = np.argmax(probs)
        col1.success(f"Global Prediction: {CLASSES[pred]} | Confidence: {probs[0][pred]:.2f}")

    if col2.button("Run Personalized Diagnosis"):
        model = load_personalized_model(patient)
        probs = predict_probs(model, beat[np.newaxis,:])
        pred = np.argmax(probs)
        col2.success(f"Personalized Prediction: {CLASSES[pred]} | Confidence: {probs[0][pred]:.2f}")

# ---- TAB 3 EXPLAINABILITY ----
with tab3:
    st.subheader("Model Attention")
    col1, col2 = st.columns(2)

    if col1.button("Explain Global"):
        st.pyplot(explanation_plot(load_global_model(), beat, "Global Attention"))

    if col2.button("Explain Personalized"):
        st.pyplot(explanation_plot(load_personalized_model(patient), beat, "Personalized Attention"))

# ---- TAB 4 RELIABILITY ----
with tab4:
    st.subheader("Performance Metrics")
    if st.button("Compute Metrics on Patient"):
        g_model = load_global_model()
        p_model = load_personalized_model(patient)

        g_probs = predict_probs(g_model, X_test)
        p_probs = predict_probs(p_model, X_test)

        g_pred = np.argmax(g_probs, axis=1)
        p_pred = np.argmax(p_probs, axis=1)

        labels_range = list(range(len(CLASSES)))

        st.write("### Global Model Metrics")
        st.text(classification_report(y_test, g_pred, labels=labels_range, target_names=CLASSES, zero_division=0))
        st.write("Accuracy:", accuracy_score(y_test, g_pred))
        st.write("Precision:", precision_score(y_test, g_pred, average='weighted', zero_division=0))
        st.write("Recall:", recall_score(y_test, g_pred, average='weighted', zero_division=0))
        st.write("F1:", f1_score(y_test, g_pred, average='weighted', zero_division=0))

        st.write("### Personalized Model Metrics")
        st.text(classification_report(y_test, p_pred, labels=labels_range, target_names=CLASSES, zero_division=0))
        st.write("Accuracy:", accuracy_score(y_test, p_pred))
        st.write("Precision:", precision_score(y_test, p_pred, average='weighted', zero_division=0))
        st.write("Recall:", recall_score(y_test, p_pred, average='weighted', zero_division=0))
        st.write("F1:", f1_score(y_test, p_pred, average='weighted', zero_division=0))

    if st.button("Show Calibration Curve"):
        def multiclass_calibration_and_ece(model, X, y, n_bins=10):
            with torch.no_grad():
                probs = torch.softmax(model(torch.tensor(X)), dim=1).cpu().numpy()
            ece = 0.0
            total = len(y)
            curves = []
            for c in range(len(CLASSES)):
                true = (y == c).astype(int)
                pred = probs[:, c]
                if true.sum() == 0:
                    continue
                # bins
                bins = np.linspace(0,1,n_bins+1)
                bin_ids = np.digitize(pred, bins) - 1
                accs = []
                confs = []
                weights = []
                for b in range(n_bins):
                    idx = bin_ids == b
                    if np.any(idx):
                        conf = pred[idx].mean()
                        acc = true[idx].mean()
                        w = idx.mean()
                        ece += abs(acc-conf)*w
                        accs.append(acc)
                        confs.append(conf)
                        weights.append(w)
                if len(confs)>1:
                    curves.append((np.array(confs), np.array(accs)))
            return curves, ece

        g_model = load_global_model()
        p_model = load_personalized_model(patient)

        g_curves, g_ece = multiclass_calibration_and_ece(g_model, X_test, y_test)
        p_curves, p_ece = multiclass_calibration_and_ece(p_model, X_test, y_test)

        st.markdown(f"### Global ECE: **{g_ece:.3f}**  |  Personalized ECE: **{p_ece:.3f}**  |  Improvement: **{(g_ece-p_ece):.3f}**")

        fig, ax = plt.subplots(figsize=(7,7))
        for x,yc in g_curves:
            ax.plot(x, yc, color='tab:blue', alpha=0.5)
        for x,yc in p_curves:
            ax.plot(x, yc, color='tab:orange', alpha=0.5)
        ax.plot([0,1],[0,1],'k--',label='Perfect')
        ax.plot([],[],color='tab:blue',label='Global')
        ax.plot([],[],color='tab:orange',label='Personalized')
        ax.legend()
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.set_title('Reliability Diagram with ECE')
        st.pyplot(fig)
