import numpy as np
from config.settings import PROCESSED_DATA_DIR, SAMPLING_RATE, PERSONALIZATION_SECONDS
from data.build_dataset import label_to_id


def load_patient(record):
    beats = np.load(PROCESSED_DATA_DIR / f"{record}_beats.npy", allow_pickle=True)
    labels = np.load(PROCESSED_DATA_DIR / f"{record}_labels.npy", allow_pickle=True)

    X, y = [], []
    for b, l in zip(beats, labels):
        if l in label_to_id:
            X.append(b)
            y.append(label_to_id[l])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # number of beats roughly in first 60 sec
    # avg heart rate ≈ 75 bpm → ~75 beats
    adapt_count = int(PERSONALIZATION_SECONDS * 75 / 60)

    X_adapt = X[:adapt_count]
    y_adapt = y[:adapt_count]

    X_test = X[adapt_count:]
    y_test = y[adapt_count:]

    return X_adapt, y_adapt, X_test, y_test

def load_patient_partial(record, seconds):

    beats = np.load(PROCESSED_DATA_DIR / f"{record}_beats.npy", allow_pickle=True)
    labels = np.load(PROCESSED_DATA_DIR / f"{record}_labels.npy", allow_pickle=True)

    X, y = [], []
    for b, l in zip(beats, labels):
        if l in label_to_id:
            X.append(b)
            y.append(label_to_id[l])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    beats_per_sec = 75 / 60
    adapt_count = int(seconds * beats_per_sec)

    X_adapt = X[:adapt_count]
    y_adapt = y[:adapt_count]

    X_test = X[adapt_count:]
    y_test = y[adapt_count:]

    return X_adapt, y_adapt, X_test, y_test
