# heartbeat_segment.py
import numpy as np
from pathlib import Path
from config.settings import PRE_R_PEAK, POST_R_PEAK, BEAT_LENGTH, PROCESSED_DATA_DIR
from data.mitbih_loader import load_record
from data.patient_split import map_label


def extract_beats(record_name):
    signal, r_peaks, labels = load_record(record_name)

    beats = []
    beat_labels = []

    for peak, sym in zip(r_peaks, labels):

        mapped = map_label(sym)
        if mapped is None:
            continue

        start = peak - PRE_R_PEAK
        end = peak + POST_R_PEAK

        if start < 0 or end >= len(signal):
            continue

        beat = signal[start:end]

        # Normalize each beat (important!)
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        beats.append(beat)
        beat_labels.append(mapped)

    return np.array(beats), np.array(beat_labels)


def save_processed(record_name):
    beats, labels = extract_beats(record_name)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DATA_DIR / f"{record_name}_beats.npy", beats)
    np.save(PROCESSED_DATA_DIR / f"{record_name}_labels.npy", labels)

    print(f"{record_name}: saved {len(beats)} beats")


if __name__ == "__main__":
    save_processed("100")
