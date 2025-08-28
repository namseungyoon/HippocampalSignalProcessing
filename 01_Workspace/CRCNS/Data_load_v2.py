import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch       # Notch filter for 60Hz

matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# Bandpass for theta (4-12Hz)
def bandpass_filter(data, fs=1250, lowcut=4, highcut=12, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

# Notch filter for 60Hz
def notch_filter(data, fs=1250, freq=60.0, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

def load_eeg(filename, n_channels=1):
    """
    .eeg 파일을 불러오는 함수 (v2)
    Args:
        filename (str): .eeg 파일의 전체 경로 또는 상대 경로
        n_channels (int): 채널 수 (기본값 1)
    Returns:
        lfp (np.ndarray): LFP 신호 (shape: [n_samples, n_channels])
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {filename}")
    data = np.fromfile(filename, dtype=np.int16)
    if n_channels > 1:
        lfp = data.reshape(-1, n_channels)
    else:
        lfp = data.reshape(-1, 1)
    return lfp

def load_res(filename):
    """
    .res.N 파일 로드 (spike 발생 시간)
    Args:
        filename (str): .res.N 파일 경로
    Returns:
        spike_times (np.ndarray): spike 발생 샘플 인덱스 (shape: [n_spikes,])
    """
    return np.loadtxt(filename, dtype=int)

def load_spk(filename, n_samples_per_spike):
    """
    .spk.N 파일 로드 (spike waveform)
    Args:
        filename (str): .spk.N 파일 경로
        n_samples_per_spike (int): spike 1개당 샘플 수
    Returns:
        waveforms (np.ndarray): spike별 waveform (shape: [n_spikes, n_samples_per_spike])
    """
    data = np.fromfile(filename, dtype=np.int16)
    n_spikes = data.size // n_samples_per_spike
    return data.reshape((n_spikes, n_samples_per_spike))

def load_fet(filename):
    """
    .fet.N 파일 로드 (spike feature)
    Args:
        filename (str): .fet.N 파일 경로
    Returns:
        features (np.ndarray): spike별 feature 벡터 (shape: [n_spikes, n_features])
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_features = int(lines[0])
    features = np.array([[int(val) for val in line.strip().split()] for line in lines[1:]], dtype=int)
    return features

def load_clu(filename):
    """
    .clu.N 파일 로드 (클러스터 정보)
    Args:
        filename (str): .clu.N 파일 경로
    Returns:
        n_clusters (int): 클러스터 개수
        cluster_ids (np.ndarray): spike별 클러스터 ID (shape: [n_spikes,])
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    n_clusters = int(lines[0])
    cluster_ids = np.array([int(x) for x in lines[1:]], dtype=int)
    return n_clusters, cluster_ids

if __name__ == "__main__":
    eeg_path = "/Users/namseungyoon/Desktop/ETRI/PROJECT/Project_2025_2026_HIPPO/03_Research_Development/99_DATASET/CRCNS_DATASET/hc-3/ec013.379/"
    eeg_file = "ec013.379.eeg"

    fs = 1250   # LFP 샘플링 레이트 (Hz)

    lfp = load_eeg(eeg_path+eeg_file, n_channels=1)
    nf_lfp = notch_filter(lfp[:,0])
    bf_lfp = bandpass_filter(lfp[:,0])
    time_index = np.arange(lfp.shape[0]) / fs
    start_time_index = 2000 * fs
    end_time_index = 2001 * fs

    cutoff = 300
    order = 4
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq

    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    low_lfp = filtfilt(b, a, lfp[:,0])

    plt.figure(figsize=(8, 16))

    # 1. 원본 LFP
    plt.subplot(4, 1, 1)
    plt.plot(time_index[start_time_index:end_time_index], lfp[start_time_index:end_time_index,0], color='gray', linewidth=0.5)
    # plt.plot(time_index[start_time_index:end_time_index], low_lfp[start_time_index:end_time_index], color='green', linewidth=0.7)

    # plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('원본 LFP')

    # 2. 노치 필터 LFP
    plt.subplot(4, 1, 2)
    plt.plot(time_index[start_time_index:end_time_index], nf_lfp[start_time_index:end_time_index], color='red', linewidth=0.7)
    # plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('노치 필터 LFP')

    # 3. 밴드패스 필터 LFP
    plt.subplot(4, 1, 3)
    plt.plot(time_index[start_time_index:end_time_index], bf_lfp[start_time_index:end_time_index], color='blue', linewidth=0.7)
    # plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('밴드패스 필터 LFP (4-12Hz) Theta 리듬 영역')

    # 4. 로우패스 필터 LFP (300Hz)
    plt.subplot(4, 1, 4)
    plt.plot(time_index[start_time_index:end_time_index], low_lfp[start_time_index:end_time_index], color='green', linewidth=0.7)
    # plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('로우패스 필터 LFP (300Hz)')

    plt.tight_layout()
    plt.show()