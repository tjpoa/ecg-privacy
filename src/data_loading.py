import wfdb

def load_record(record_path):
    """
    Load ECG record from WFDB format.
    """
    
    record = wfdb.rdrecord(str(record_path))

    signal = record.p_signal
    fs = record.fs
    leads = record.sig_name

    return signal, fs, leads