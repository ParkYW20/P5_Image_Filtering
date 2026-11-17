"""
P5 영상 필터링 — 제공 유틸리티 (수정 불필요)
"""
import csv, os
import numpy as np
from PIL import Image

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def read_gray(path):
    """8비트 흑백 영상 → float 배열 (0~255)"""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def save_gray(path, img):
    Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8)).save(path)


def load_test_list(path=os.path.join(DATA, "test_list.csv")):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def psnr(ref, img):
    """PSNR [dB] (8비트 기준, 결과는 0~255로 잘라서 비교)"""
    mse = np.mean((np.asarray(ref, float) - np.clip(img, 0, 255)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def sharpness(img):
    """선명도 지표: 평균 기울기 크기 (중앙 차분). 에지가 또렷할수록 큼"""
    f = np.clip(np.asarray(img, float), 0, 255)
    gy, gx = np.gradient(f)
    return float(np.mean(np.hypot(gx, gy)))


def freq_response_db(kernel, n=256):
    """2D 커널의 주파수 응답 크기 [dB] (중앙이 DC, n x n 영점 채움 DFT)"""
    H = np.fft.fftshift(np.abs(np.fft.fft2(kernel, (n, n))))
    return 20 * np.log10(H + 1e-12)
