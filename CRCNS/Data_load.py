#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRCNS 신경과학 데이터 파일 로더

각종 신경 데이터 파일(.clu.N, .fet.N, .spk.N, .res.N, .eeg, .whl, .xml, .threshold.N, .m1m2.N, .mm.N, .m1v 등)을 읽어들이는 함수 모음
각 함수는 파일 포맷별로 데이터 구조와 반환값을 명확히 주석으로 설명함
"""

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import h5py
import os
import matplotlib.pyplot as plt

# 데이터셋 폴더 경로 상수
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'DATASET')

def get_dataset_path(filename):
    """
    DATASET 폴더 내 파일의 전체 경로 반환
    Args:
        filename (str): 파일명 또는 상대경로
    Returns:
        full_path (str): DATASET/ 하위의 전체 경로
    """
    return os.path.join(DATASET_PATH, filename)

# -----------------------------
# .clu.N: 클러스터링 결과 파일
# 각 spike가 어느 cluster(뉴런)에 속하는지 ID 제공 (N: 전극 번호)
# 첫 줄: 클러스터 개수, 이후 각 줄: spike별 cluster ID (int)
def load_clu(filename_or_path):
    """
    .clu.N 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .clu.N 파일명 또는 상대경로
    Returns:
        n_clusters (int): 클러스터 개수
        cluster_ids (np.ndarray): spike별 클러스터 ID (shape: [n_spikes,])
    """
    path = get_dataset_path(filename_or_path)
    with open(path, 'r') as f:
        lines = f.readlines()
    n_clusters = int(lines[0])
    cluster_ids = np.array([int(x) for x in lines[1:]], dtype=int)
    return n_clusters, cluster_ids

# -----------------------------
# .fet.N: spike feature 파일 (PCA 등)
# 첫 줄: feature 개수, 이후 각 줄: spike별 feature 벡터 (int)
def load_fet(filename_or_path):
    """
    .fet.N 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .fet.N 파일명 또는 상대경로
    Returns:
        features (np.ndarray): spike별 feature 벡터 (shape: [n_spikes, n_features])
    """
    path = get_dataset_path(filename_or_path)
    with open(path, 'r') as f:
        lines = f.readlines()
    n_features = int(lines[0])
    features = np.array([[int(val) for val in line.strip().split()] for line in lines[1:]], dtype=int)
    return features

# -----------------------------
# .spk.N: spike waveform 데이터 (raw spike 신호)
# 각 row: spike별 waveform (샘플 단위, int16)
def load_spk(filename_or_path, n_samples_per_spike):
    """
    .spk.N 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .spk.N 파일명 또는 상대경로
        n_samples_per_spike (int): spike 1개당 샘플 수
    Returns:
        waveforms (np.ndarray): spike별 waveform (shape: [n_spikes, n_samples_per_spike])
    """
    path = get_dataset_path(filename_or_path)
    data = np.fromfile(path, dtype=np.int16)
    n_spikes = data.size // n_samples_per_spike
    waveforms = data.reshape((n_spikes, n_samples_per_spike))
    return waveforms

# -----------------------------
# .res.N: spike 발생 시간 (샘플 단위, int)
def load_res(filename_or_path):
    """
    .res.N 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .res.N 파일명 또는 상대경로
    Returns:
        spike_times (np.ndarray): spike 발생 샘플 인덱스 (shape: [n_spikes,])
    """
    path = get_dataset_path(filename_or_path)
    spike_times = np.loadtxt(path, dtype=int)
    return spike_times

# -----------------------------
# .eeg: LFP 데이터 (1.25kHz 다운샘플, int16)
def load_eeg(filename_or_path, n_channels=1):
    """
    .eeg 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .eeg 파일명 또는 상대경로
        n_channels (int): 채널 수 (기본 1)
    Returns:
        lfp (np.ndarray): LFP 신호 (shape: [n_samples, n_channels])
    """
    path = get_dataset_path(filename_or_path)
    data = np.fromfile(path, dtype=np.int16)
    if n_channels > 1:
        lfp = data.reshape(-1, n_channels)
    else:
        lfp = data.reshape(-1, 1)
    return lfp

# -----------------------------
# .whl: 동물 위치(x, y) 정보 (비디오 기반)
def load_whl(filename_or_path):
    """
    .whl 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .whl 파일명 또는 상대경로
    Returns:
        pos (np.ndarray): 프레임별 (x, y) 위치 (shape: [n_frames, 2])
    """
    path = get_dataset_path(filename_or_path)
    pos = np.loadtxt(path, dtype=float, usecols=(0,1))
    return pos

# -----------------------------
# .xml: 세션 설정 정보 (채널, 샘플링 속도 등, Neuroscope)
def load_xml(filename_or_path):
    """
    .xml 파일 로드 (Neuroscope) (DATASET 폴더)
    Args:
        filename_or_path (str): .xml 파일명 또는 상대경로
    Returns:
        root (xml.etree.ElementTree.Element): XML 루트 노드
    """
    path = get_dataset_path(filename_or_path)
    tree = ET.parse(path)
    root = tree.getroot()
    return root

# -----------------------------
# .threshold.N: spike 감지 시 사용된 역치 정보 (float)
def load_threshold(filename_or_path):
    """
    .threshold.N 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): .threshold.N 파일명 또는 상대경로
    Returns:
        threshold (float): 감지 역치 값
    """
    path = get_dataset_path(filename_or_path)
    with open(path, 'r') as f:
        threshold = float(f.readline().strip())
    return threshold

# -----------------------------
# .m1m2.N, .mm.N: Klusters spike sorting 보조 파일 (hdf5 등)
def load_hdf5(filename_or_path):
    """
    .m1m2.N, .mm.N 등 HDF5 파일 로드 (DATASET 폴더)
    Args:
        filename_or_path (str): hdf5 파일명 또는 상대경로
    Returns:
        h5file (h5py.File): h5py 파일 객체
    """
    path = get_dataset_path(filename_or_path)
    h5file = h5py.File(path, 'r')
    return h5file

# -----------------------------
# .m1v: 동영상 파일 (행동 영상)
# 동영상은 OpenCV 등으로 프레임 단위로 읽을 수 있음
def load_video(filename_or_path):
    """
    .m1v 동영상 파일 로드 (OpenCV 필요) (DATASET 폴더)
    Args:
        filename_or_path (str): 동영상 파일명 또는 상대경로
    Returns:
        cap (cv2.VideoCapture): OpenCV VideoCapture 객체
    """
    import cv2
    path = get_dataset_path(filename_or_path)
    cap = cv2.VideoCapture(path)
    return cap

# -----------------------------
# 예시: 파일 존재 여부 체크 및 로드
if __name__ == "__main__":
    # 예시 파일명 (DATASET 폴더 기준)
    example_clu = 'example.clu.1'
    example_fet = 'example.fet.1'
    example_spk = 'example.spk.1'
    example_res = 'example.res.1'
    example_eeg = 'example.eeg'
    example_whl = 'example.whl'
    example_xml = 'example.xml'
    example_threshold = 'example.threshold.1'
    example_hdf5 = 'example.mm.1'
    example_video = 'example.m1v'

    # 각 파일별 로드 예시 (파일이 존재할 때만)
    if os.path.exists(get_dataset_path(example_clu)):
        n_clusters, cluster_ids = load_clu(example_clu)
        print(f"clu: {n_clusters} clusters, {cluster_ids.shape} IDs")
    if os.path.exists(get_dataset_path(example_fet)):
        features = load_fet(example_fet)
        print(f"fet: {features.shape}")
    if os.path.exists(get_dataset_path(example_spk)):
        waveforms = load_spk(example_spk, n_samples_per_spike=32)
        print(f"spk: {waveforms.shape}")
    if os.path.exists(get_dataset_path(example_res)):
        spike_times = load_res(example_res)
        print(f"res: {spike_times.shape}")
    if os.path.exists(get_dataset_path(example_eeg)):
        lfp = load_eeg(example_eeg, n_channels=1)
        print(f"eeg: {lfp.shape}")
    if os.path.exists(get_dataset_path(example_whl)):
        pos = load_whl(example_whl)
        print(f"whl: {pos.shape}")
    if os.path.exists(get_dataset_path(example_xml)):
        root = load_xml(example_xml)
        print(f"xml: {root.tag}")
    if os.path.exists(get_dataset_path(example_threshold)):
        threshold = load_threshold(example_threshold)
        print(f"threshold: {threshold}")
    if os.path.exists(get_dataset_path(example_hdf5)):
        h5file = load_hdf5(example_hdf5)
        print(f"hdf5: {list(h5file.keys())}")
    if os.path.exists(get_dataset_path(example_video)):
        cap = load_video(example_video)
        print(f"video: {cap.isOpened()}")

def list_sessions():
    """
    DATASET 폴더 내 실험(세션) 폴더 목록 반환
    Returns:
        session_dirs (list): 세션 폴더명 리스트
    """
    return [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]


def load_all_sessions():
    """
    DATASET 폴더 내 모든 세션 폴더를 순회하며 주요 신경 데이터 파일을 자동으로 로드
    (단, whl, xml, m1m2, mm, m1v 파일은 제외)
    Returns:
        sessions_data (dict): {세션명: {파일타입: 데이터}} 형태의 딕셔너리
    """
    sessions_data = {}
    session_dirs = list_sessions()
    for session in session_dirs:
        session_path = os.path.join(DATASET_PATH, session)
        files = os.listdir(session_path)
        session_data = {}
        # 주요 파일 자동 로드 (존재할 때만)
        # EEG
        eeg_file = f"{session}.eeg"
        if eeg_file in files:
            session_data['eeg'] = load_eeg(os.path.join(session, eeg_file))
        # whl, xml, m1m2, mm, m1v 파일은 분석에서 제외 (위치, 설정, 보조, 동영상 등)
        # 각 tetrode별 파일 (1~8)
        for tetrode in range(1, 9):
            for ext, loader, key in [
                (f"clu.{tetrode}", load_clu, 'clu'),
                (f"fet.{tetrode}", load_fet, 'fet'),
                (f"res.{tetrode}", load_res, 'res'),
                (f"spk.{tetrode}", lambda f: load_spk(f, n_samples_per_spike=32), 'spk'),
                (f"threshold.{tetrode}", load_threshold, 'threshold'),
                # (f"mm.{tetrode}", load_hdf5, 'mm'),      # mm: 보조파일, 제외
                # (f"m1m2.{tetrode}", load_hdf5, 'm1m2'),  # m1m2: 보조파일, 제외
            ]:
                fname = f"{session}.{ext}"
                if fname in files:
                    if key not in session_data:
                        session_data[key] = {}
                    session_data[key][tetrode] = loader(os.path.join(session, fname))
        # 기타 파일 (예: mpg, m1v 등)
        # m1v(동영상), mm, m1m2, whl, xml 등은 제외
        sessions_data[session] = session_data
    return sessions_data


def load_session(session_name):
    """
    특정 세션 폴더의 주요 신경 데이터 파일만 불러오는 함수
    (whl, xml, m1m2, mm, m1v 파일은 제외)
    Args:
        session_name (str): 세션 폴더명 (예: 'ec013.379')
    Returns:
        session_data (dict): {파일타입: 데이터, 'session_name': 세션명} 형태의 딕셔너리
    """
    session_path = os.path.join(DATASET_PATH, session_name)
    files = os.listdir(session_path)
    session_data = {}
    session_data['session_name'] = session_name  # 실험명 추가
    # EEG
    eeg_file = f"{session_name}.eeg"
    if eeg_file in files:
        session_data['eeg'] = load_eeg(os.path.join(session_name, eeg_file))
    # whl, xml, m1m2, mm, m1v 파일은 제외
    # 각 tetrode별 파일 (1~8)
    for tetrode in range(1, 9):
        for ext, loader, key in [
            (f"clu.{tetrode}", load_clu, 'clu'),
            (f"fet.{tetrode}", load_fet, 'fet'),
            (f"res.{tetrode}", load_res, 'res'),
            (f"spk.{tetrode}", lambda f: load_spk(f, n_samples_per_spike=32), 'spk'),
            (f"threshold.{tetrode}", load_threshold, 'threshold'),
            # (f"mm.{tetrode}", load_hdf5, 'mm'),      # mm: 보조파일, 제외
            # (f"m1m2.{tetrode}", load_hdf5, 'm1m2'),  # m1m2: 보조파일, 제외
        ]:
            fname = f"{session_name}.{ext}"
            if fname in files:
                if key not in session_data:
                    session_data[key] = {}
                session_data[key][tetrode] = loader(os.path.join(session_name, fname))
    # 기타 파일 (예: mpg 등)
    # m1v(동영상), mm, m1m2, whl, xml 등은 제외
    return session_data
    

# 사용 예시
if __name__ == "__main__":
    print("세션 폴더 목록:", list_sessions())
    # 모든 세션의 주요 데이터 자동 로드 예시
    session_data = load_session(list_sessions()[0])
    print("현재 실험명 : ", session_data['session_name'])
    print("Session Keys : ", session_data.keys())

    eeg = session_data["eeg"]
    
    print("LFP샘플수 : ", eeg.shape[0])   # 샘플수
    print("LFP채널수 : ", eeg.shape[1])   # 채널수
    sampling_rate = 1250  # Hz
    n_samples = eeg.shape[0]

    # 타임 인덱스 (초 단위)
    time_index = np.arange(n_samples) / sampling_rate
    print(time_index[:10])

    # --- 그래프 시각화 ---
    plt.figure(figsize=(12, 4))
    plt.plot(time_index[:1000], eeg[:1000, 0], label='Channel 0')
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title([session_data['session_name'], 'LFP'])
    plt.legend()
    plt.tight_layout()
    plt.show()
