from pathlib import Path
from data.mitbih_loader import RECORDS
from data.heartbeat_segment import save_processed
from config.settings import PROCESSED_DATA_DIR


def already_processed(record):
    beats_file = PROCESSED_DATA_DIR / f"{record}_beats.npy"
    labels_file = PROCESSED_DATA_DIR / f"{record}_labels.npy"
    return beats_file.exists() and labels_file.exists()


def build_dataset():
    print("Processing MIT-BIH records (resume enabled)...\n")

    for rec in RECORDS:
        if already_processed(rec):
            print(f"{rec} already done, skipping")
            continue

        try:
            save_processed(rec)
        except Exception as e:
            print(f"Error processing {rec}: {e}")

    print("\nDataset creation complete!")


if __name__ == "__main__":
    build_dataset()
