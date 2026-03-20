# mitbih_loader.py
import wfdb
from pathlib import Path
from config.settings import RAW_DATA_DIR

# MIT-BIH record list (all patients)
RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

def download_record(record_name):
    """Download a single record from PhysioNet if not already present"""
    record_path = RAW_DATA_DIR / record_name
    if not record_path.with_suffix('.dat').exists():
        print(f"Downloading record {record_name}...")
        wfdb.dl_database('mitdb', dl_dir=str(RAW_DATA_DIR), records=[record_name])

def load_record(record_name):
    """Load ECG signal and annotation together"""
    download_record(record_name)

    record_path = RAW_DATA_DIR / record_name

    # ECG signal
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal[:,0]  # use MLII lead

    # Annotation (heartbeat labels + R peaks)
    annotation = wfdb.rdann(str(record_path), 'atr')

    r_peaks = annotation.sample
    labels = annotation.symbol

    return signal, r_peaks, labels


if __name__ == "__main__":
    sig, peaks, labs = load_record("100")
    print("Signal length:", len(sig))
    print("Number of beats:", len(peaks))
    print("First 10 labels:", labs[:10])
