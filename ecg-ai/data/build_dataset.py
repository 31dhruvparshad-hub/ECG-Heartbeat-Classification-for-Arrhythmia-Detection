import numpy as np
from config.settings import PROCESSED_DATA_DIR, CLASSES
from data.patient_split import DS1, DS2

label_to_id = {c: i for i, c in enumerate(CLASSES)}
id_to_label = {i: c for c, i in label_to_id.items()}


def load_group(records):
    X, y = [], []

    for r in records:
        beats = np.load(PROCESSED_DATA_DIR / f"{r}_beats.npy", allow_pickle=True)
        labels = np.load(PROCESSED_DATA_DIR / f"{r}_labels.npy", allow_pickle=True)

        for b, l in zip(beats, labels):
            if l in label_to_id:
                X.append(b)
                y.append(label_to_id[l])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def build_train_test():
    print("Loading DS1 (training patients)...")
    X_train, y_train = load_group(DS1)

    print("Loading DS2 (test patients)...")
    X_test, y_test = load_group(DS2)

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    build_train_test()
