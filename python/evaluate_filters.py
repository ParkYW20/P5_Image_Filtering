"""
P5 영상 필터링 — 공통 평가 스크립트 (수정 불필요)

사용 예:
  python evaluate_filters.py                                  # data/test_list.csv 의 시험 영상 전체
  python evaluate_filters.py --list <csv> --dir <폴더>          # 다른 시험 세트 (교수 시연용 등)

지표: PSNR [dB] (깨끗한 원본 대비, 클수록 좋음),  선명도 = 평균 기울기 크기 (원본 대비 비율로도 표시)
"""
import argparse, importlib.util, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import img_utils as U


def load_impl(path):
    spec = importlib.util.spec_from_file_location("impl", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def opt(fn, *a):
    """선택 과제 함수: 미구현이면 None"""
    try:
        return fn(*a)
    except NotImplementedError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=os.path.join(HERE, "my_filters.py"))
    ap.add_argument("--list", default=os.path.join(U.DATA, "test_list.csv"))
    ap.add_argument("--dir", default=os.path.join(U.DATA, "images_test"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    M = load_impl(a.impl)

    # ---------------------------------------------------------- 1. 구현 검증
    print("\n--- 1. 구현 검증 ---")
    rng = np.random.default_rng(0); x = rng.uniform(0, 255, (64, 80)); h = rng.uniform(-1, 1, (3, 5))
    try:
        y = np.asarray(M.conv2d(x, h)); ref = ndimage.convolve(x, h, mode="nearest")
        e = np.max(np.abs(y - ref)) if y.shape == ref.shape else np.inf
        print(f"conv2d : 크기 {y.shape} (기대 {ref.shape}),  scipy 대비 최대 오차 = {e:.2e}")
        if e > 1e-6 and y.shape == ref.shape and np.max(np.abs(y - ndimage.correlate(x, h, mode='nearest'))) < 1e-6:
            print("         → 상관(correlation)과 일치합니다. 커널을 뒤집었는지 확인하세요.")
    except NotImplementedError as err:
        print(f"conv2d : 미구현({err}) → 이후 평가 불가"); return
    try:
        b = np.asarray(M.box_kernel(5)); g = np.asarray(M.gaussian_kernel(1.2))
        print(f"box_kernel(5)      : 크기 {b.shape}, 합 {b.sum():.6f}")
        print(f"gaussian_kernel(1.2): 크기 {g.shape} (기대 (9, 9)), 합 {g.sum():.6f}, 대칭 오차 {np.max(np.abs(g - g.T)):.1e}, "
              f"중앙값 {g[g.shape[0] // 2, g.shape[1] // 2]:.4f} (연속 근사 1/(2πσ²) = {1 / (2 * np.pi * 1.2 ** 2):.4f})")
    except NotImplementedError as err:
        print(f"커널 : 미구현({err}) → 이후 평가 불가"); return
    img = U.read_gray(os.path.join(U.DATA, "images_clean", "camera.png"))
    g2 = np.asarray(M.gaussian_kernel(2.0)); k1 = g2.sum(axis=1)
    t0 = time.perf_counter(); yd = np.asarray(M.conv2d(img, g2)); td = time.perf_counter() - t0
    t0 = time.perf_counter(); ys = opt(M.conv2d_separable, img, k1); ts = time.perf_counter() - t0
    print(f"속도 (512x512, 가우시안 σ=2 → {g2.shape[0]}x{g2.shape[1]}): 2D 직접 {td:.3f} s", end="")
    if ys is not None:
        print(f",  분리형 {ts:.3f} s (결과 차이 {np.max(np.abs(np.asarray(ys) - yd)):.1e}),  "
              f"곱셈 수/화소 {g2.size} → {2 * len(k1)}")
    else:
        print(",  분리형: (선택 과제) 미구현")
    md = opt(M.median_filter, x, 3)
    if md is not None:
        print(f"median_filter : scipy 대비 최대 오차 {np.max(np.abs(np.asarray(md) - ndimage.median_filter(x, 3, mode='nearest'))):.2e}")

    # ---------------------------------------------------------- 2·3. 잡음 제거 / 샤프닝
    tests = U.load_test_list(a.list)
    noise = [t for t in tests if t["degradation"] in ("gauss", "sp")]
    blur = [t for t in tests if t["degradation"] == "blur"]
    filt = [("box3", lambda f: M.conv2d(f, M.box_kernel(3))), ("box5", lambda f: M.conv2d(f, M.box_kernel(5))),
            ("box7", lambda f: M.conv2d(f, M.box_kernel(7))), ("gau0.8", lambda f: M.conv2d(f, M.gaussian_kernel(0.8))),
            ("gau1.2", lambda f: M.conv2d(f, M.gaussian_kernel(1.2))), ("gau2.0", lambda f: M.conv2d(f, M.gaussian_kernel(2.0))),
            ("med3", lambda f: opt(M.median_filter, f, 3)), ("med5", lambda f: opt(M.median_filter, f, 5))]
    print("\n--- 2. 블러링으로 잡음 제거: PSNR [dB] (원본 대비) ---")
    print(f"{'test image':<28}{'입력':>7}" + "".join(f"{n:>8}" for n, _ in filt))
    examples = {}
    for t in noise:
        clean = U.read_gray(os.path.join(U.DATA, "images_clean", t["clean_file"])); f = U.read_gray(os.path.join(a.dir, t["test_file"]))
        row = f"{t['test_file']:<28}{U.psnr(clean, f):>7.2f}"
        for n, fn in filt:
            out = fn(f)
            row += f"{U.psnr(clean, np.asarray(out)):>8.2f}" if out is not None else f"{'-':>8}"
            if out is not None and t["test_file"].startswith(("noise_gauss20_camera", "noise_sp5_camera")):
                examples[(t["test_file"], n)] = np.asarray(out)
        print(row)
    print("(med3/med5 = 선택 과제 메디안 필터)")

    print("\n--- 3. 샤프닝으로 블러 복원: 언샤프 마스킹 (mask σ = 2.0) ---")
    alphas = [0.5, 1.0, 2.0, 4.0]
    print(f"{'test image':<28}{'':>8}{'입력':>9}" + "".join(f"{'α=' + str(al):>9}" for al in alphas))
    for t in blur:
        clean = U.read_gray(os.path.join(U.DATA, "images_clean", t["clean_file"])); f = U.read_gray(os.path.join(a.dir, t["test_file"]))
        try:
            outs = [np.asarray(M.unsharp_mask(f, 2.0, al)) for al in alphas]
        except NotImplementedError as err:
            print(f"unsharp_mask: 미구현({err})"); break
        s0 = U.sharpness(clean)
        print(f"{t['test_file']:<28}{'PSNR':>8}{U.psnr(clean, f):>9.2f}" + "".join(f"{U.psnr(clean, o):>9.2f}" for o in outs))
        print(f"{'':<28}{'선명도비':>8}{U.sharpness(f) / s0:>9.2f}" + "".join(f"{U.sharpness(o) / s0:>9.2f}" for o in outs))
        if t["test_file"].startswith("blur_g2_text"):
            examples["sharp"] = (clean, f, outs)

    # ---------------------------------------------------------- 4. 그림
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ks = {"box 5x5": M.box_kernel(5), "gaussian σ=1.2": M.gaussian_kernel(1.2)}
    try:
        g = np.asarray(M.gaussian_kernel(2.0)); d = np.zeros_like(g); d[g.shape[0] // 2, g.shape[1] // 2] = 1
        ks["unsharp α=1 (δ+(δ-g))"] = d + 1.0 * (d - g)
    except NotImplementedError:
        pass
    wn = np.fft.fftshift(np.fft.fftfreq(256))
    for name, k in ks.items():
        H = U.freq_response_db(np.asarray(k)); axs[0].plot(wn, H[128, :], label=name)
    axs[0].set_xlabel("normalized frequency (cycles/pixel)"); axs[0].set_ylabel("dB"); axs[0].set_ylim(-60, 12)
    axs[0].set_title("frequency response (center row)"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    for ax, (name, k) in zip(axs[1:], list(ks.items())[:2]):
        ax.imshow(U.freq_response_db(np.asarray(k)), extent=[-0.5, 0.5, -0.5, 0.5], vmin=-60, vmax=0, cmap="viridis")
        ax.set_title(f"|H| 2D: {name}")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "freq_responses.png"), dpi=110); plt.close(fig)
    for tf in ("noise_gauss20_camera.png", "noise_sp5_camera.png"):
        cols = [n for n in ("box5", "gau1.2", "med3") if (tf, n) in examples]
        if not cols:
            continue
        f = U.read_gray(os.path.join(a.dir, tf)); cr = (slice(80, 240), slice(180, 340))
        fig, axs = plt.subplots(1, 1 + len(cols), figsize=(3.2 * (1 + len(cols)), 3.4))
        axs[0].imshow(f[cr], cmap="gray", vmin=0, vmax=255); axs[0].set_title("input")
        for ax, n in zip(axs[1:], cols):
            ax.imshow(examples[(tf, n)][cr], cmap="gray", vmin=0, vmax=255); ax.set_title(n)
        for ax in axs: ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"denoise_{tf}"), dpi=110); plt.close(fig)
    if "sharp" in examples:
        clean, f, outs = examples["sharp"]; cr = (slice(40, 220), slice(20, 260))
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
        for ax, (im, tt) in zip(axs, [(clean, "original"), (f, "blurred"), (outs[1], "unsharp α=1"), (outs[3], "unsharp α=4")]):
            ax.imshow(np.clip(im[cr], 0, 255), cmap="gray", vmin=0, vmax=255); ax.set_title(tt); ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "sharpen_text_shapes.png"), dpi=110); plt.close(fig)
    print(f"\n그래프·결과 영상 저장 → {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
