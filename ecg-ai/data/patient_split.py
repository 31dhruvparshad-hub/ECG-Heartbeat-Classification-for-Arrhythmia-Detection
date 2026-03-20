# patient_split.py
# Mapping MIT-BIH annotations to our 5 classes

LABEL_MAP = {
    'N': 'N',      # Normal
    'L': 'LBBB',
    'R': 'RBBB',
    'A': 'PAC',
    'a': 'PAC',
    'J': 'PAC',
    'S': 'PAC',
    'V': 'PVC',
    'E': 'PVC'
}

VALID_LABELS = ['N', 'PAC', 'PVC', 'RBBB', 'LBBB']


def map_label(symbol):
    """Convert MIT-BIH symbol to our class or None if unwanted"""
    if symbol in LABEL_MAP:
        return LABEL_MAP[symbol]
    return None

# -------- Inter-patient DS1 / DS2 split (De Chazal standard) --------

DS1 = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220','223','230'
]

DS2 = [
    '100','103','105','111','113','117','121','123','200','202','210','212',
    '213','214','219','221','222','228','231','232','233','234'
]
