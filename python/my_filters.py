"""
P5 영상 필터링 — 학생 구현 파일 (TODO를 채우세요)

2D 컨볼루션:  y[m, n] = sum_i sum_j h[i, j] · x[m - i, n - j]
  - 출력 크기 = 입력 크기 ('same'),  커널 중심이 출력 화소에 오도록 정렬 (커널 크기는 홀수)
  - 경계 처리: 가장자리 화소 복제(edge replicate)  →  np.pad(x, r, mode="edge")
  - 컨볼루션이므로 커널을 뒤집어서(flip) 곱한다  (3장의 '접고, 밀고, 곱하고, 더하기'의 2차원 버전)

[직접 구현 요건] conv2d 는 scipy.signal.convolve2d / scipy.ndimage / cv2.filter2D 등을 쓰지 말고 직접 구현.
평가 스크립트가 scipy.ndimage.convolve(mode="nearest") 결과와 비교합니다.
"""
import numpy as np


# def conv2d(img, kernel):
#     """
#     2D 컨볼루션 (same 크기, edge 복제 경계).
#     힌트(빠른 방법): 패딩한 영상에서 커널 원소 개수만큼 '밀린 영상'을 잘라 가중합
#           for i, j in 커널 위치:  y += h_flipped[i, j] * xpad[i : i+H, j : j+W]
#     """
#     # TODO
#     raise NotImplementedError("conv2d")


# def box_kernel(k):
#     """k x k 평균 필터 (모든 원소 1/k²)"""
#     # TODO
#     raise NotImplementedError("box_kernel")


# def gaussian_kernel(sigma, size=None):
#     """
#     2D 가우시안 커널: h[i, j] ∝ exp(-(i² + j²) / (2σ²)),  합이 1이 되도록 정규화
#     size 가 None 이면 size = 2*ceil(3σ) + 1 (±3σ 까지 포함)
#     """
#     # TODO
#     raise NotImplementedError("gaussian_kernel")


# def unsharp_mask(img, sigma, alpha):
#     """
#     언샤프 마스킹(샤프닝): y = x + alpha · (x - blur(x)),   blur = 가우시안(sigma)
#     (x - blur(x)) 가 고주파 성분 → 이를 alpha 배 더해 에지를 강조.  결과는 0~255로 자르기
#     """
#     # TODO
#     raise NotImplementedError("unsharp_mask")


# # ------------------------------------------------------------------ 선택 과제
# def conv2d_separable(img, k1d):
#     """(선택) 분리 가능한 커널 h = k1d ⊗ k1d 를 1D 컨볼루션 두 번(가로, 세로)으로 → 연산량 K² → 2K"""
#     raise NotImplementedError("conv2d_separable (선택)")


# def median_filter(img, k):
#     """(선택) k x k 메디안 필터 (비선형, edge 복제 경계). 소금·후추 잡음에 강함"""
#     raise NotImplementedError("median_filter (선택)")


# 가중치 행렬 생성 코드 
def _odd(k):
    if not isinstance(k, (int, np.integer)) or k < 1 or k % 2 != 1:
        raise ValueError('kernel size must be a positive odd integer')
    return k

# 영환 파트
def conv2d(img, kernel):
    """Direct convolution: float64, same shape, edge replication, flipped kernel."""
    x, h = np.asarray(img, dtype=float), np.asarray(kernel, dtype=float)
    if x.ndim != 2 or h.ndim != 2:
        raise ValueError('expected 2D image and kernel')
    kh, kw = map(_odd, h.shape)
    p = np.pad(x, ((kh//2, kh//2), (kw//2, kw//2)), mode='edge')
    y = np.zeros_like(x)
    hf = h[::-1, ::-1]
    for i in range(kh):
        for j in range(kw):
            y += hf[i,j] * p[i:i+x.shape[0], j:j+x.shape[1]]
    return y

#지혜 파트
def box_kernel(k):
    """k x k 평균 필터 (모든 원소 1/k²)"""
    return np.ones((k, k), dtype=np.float64) / (k * k)    

#연우 파트
def gaussian_kernel(sigma, size=None):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')
    size = _odd(2*int(np.ceil(3*sigma))+1 if size is None else size)
    q = np.arange(size, dtype=float) - size//2
    xx, yy = np.meshgrid(q, q)
    h = np.exp(-(xx*xx + yy*yy)/(2*sigma*sigma))
    return h / h.sum()

#강민 파트
def unsharp_mask(img, sigma, alpha):
    """2D 흑백 영상 -> 같은 크기의 float64 영상, 최종 값 범위 0~255.

    sigma > 0, alpha >= 0. 입력 배열은 변경하지 않는다.
    x - blurred는 음수도 포함하므로 중간 클리핑을 하지 않는다.
    """
    x = np.asarray(img, dtype=np.float64)
    if x.ndim != 2 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("img must be a nonempty finite 2D grayscale array")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and >= 0")
    kernel = gaussian_kernel(sigma)
    blurred = conv2d(x, kernel)
    detail = x - blurred
    enhanced = x + alpha * detail
    return np.clip(enhanced, 0.0, 255.0) 


# 선택 과제 2개 
#지혜 파트
def conv2d_separable(img, k1d):
    """(선택) 분리 가능한 커널 h = k1d ⊗ k1d 를 1D 컨볼루션 두 번(가로, 세로)으로 → 연산량 K² → 2K"""
    img = np.asarray(img, dtype=float)
    k1d = np.asarray(k1d, dtype=float)

    H, W = img.shape
    K = len(k1d)
    pad = K // 2

    kernel = k1d[::-1]

    # 1. 가로 방향 Convolution
    padded = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    temp = np.zeros_like(img)

    for i in range(H):
        for j in range(W):
            temp[i, j] = np.sum(padded[i, j:j+K] * kernel)

    # 2. 세로 방향 Convolution
    padded = np.pad(temp, ((pad, pad), (0, 0)), mode="edge")
    result = np.zeros_like(img)

    for i in range(H):
        for j in range(W):
            result[i, j] = np.sum(padded[i:i+K, j] * kernel)

    return result

def median_filter(img, k):
    k = _odd(k)
    x = np.asarray(img, dtype=float)
    p = np.pad(x, k//2, mode='edge')
    return np.median(np.lib.stride_tricks.sliding_window_view(p, (k,k)), axis=(-2,-1))
