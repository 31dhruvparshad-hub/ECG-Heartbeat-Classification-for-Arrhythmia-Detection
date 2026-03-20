from pathlib import Path
from config.settings import PROCESSED_DATA_DIR
from data.patient_split import DS1, DS2

def check_group(name, records):
    print(f"\n{name}")
    total = 0
    for r in records:
        f = PROCESSED_DATA_DIR / f"{r}_beats.npy"
        if f.exists():
            total += 1
        else:
            print("Missing:", r)
    print("Available records:", total, "/", len(records))

if __name__ == "__main__":
    check_group("DS1 (training patients)", DS1)
    check_group("DS2 (test patients)", DS2)
