#!/usr/bin/env python3
import os
import sys
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Tuple

# ---------------- CONFIG ----------------
PRETRAIN_CKPT   = "../CheckpointsCondition/ckpt_1800.pt"
POSTTRAIN_CKPT  = "../Checkpoints_pert/ckpt_ft.pt"

# GT files (your paths)
GT_SMALL_Z = "/home/wek54nug/Denoising_ddpm/GT_small_photon_shower_z.h5"  # (2.5,5.9)
GT_LARGE_Z = "/home/wek54nug/Denoising_ddpm/GT_large_photon_shower_z.h5"  # (2.5,6.1)
GT_SMALL_XY = "/home/wek54nug/Denoising_ddpm/GT_small_photon_shower_xy.h5"  # (2.4,6.0)
GT_LARGE_XY = "/home/wek54nug/Denoising_ddpm/GT_large_photon_shower_xy.h5"  # (2.6,6.0)

OUT_DIR = "/home/wek54nug/Denoising_ddpm/bib_gen"
os.makedirs(OUT_DIR, exist_ok=True)

# Batch / image sizes
BATCH_SIZE = 10         # sampler batch (increase if GPU allows)
BATCH_SIZE_FD = 5000   # per-repeat paired FD sample size (tune)
N_REPEATS = 10           # number of repeated paired draws to concatenate (tune)
IMG_SIZE   = 32
POOL_K, POOL_S = 6, 6
ALPHA = 1e-2

# Energies (labels) and mapping to indices
ENERGY_LABELS = [1,10,20,30,40,50,60,70,80,90,100]
IDX_MAP = {e:i for i,e in enumerate(ENERGY_LABELS)}

# DDPM config (must match model you trained)
DDPM_CFG = {
    "T": 500, "channel": 32, "channel_mult": [1,2,2,2],
    "num_res_blocks": 2, "dropout": 0.15,
    "beta_1": 1e-4, "beta_T": 0.028, "eta": 0.0,
    "ddim_steps": 50,
}

# Sigmoid sharpness used in E_meas mask
SIGMOID_K = 1

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Add project root (if your DiffusionFreeGuidence package is in parent)
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import your model/sampler (adjust import paths if needed)
from DiffusionFreeGuidence.ModelCondition import UNet
from DiffusionFreeGuidence.DiffusionCondition import DDIMSampler

# ---------------- HELPERS ----------------
def set_seed(seed=123456789):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fmt_num(x):
    s = str(x)
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s

def pool_to_5x5(x):
    return F.avg_pool2d(x, POOL_K, POOL_S) * (POOL_K * POOL_S)

def load_bib_map_32(xy_cm, z_cm):
    candidates = []
    xy_s = fmt_num(xy_cm); z_s = fmt_num(z_cm)
    candidates.append(f"bib_map_{xy_s}x{xy_s}x{z_s}_32x32.npy")
    candidates.append(f"bib_map_{format(xy_cm,'.1f')}x{format(xy_cm,'.1f')}x{format(z_cm,'.1f')}_32x32.npy")
    candidates.append(f"bib_map_{xy_cm}x{xy_cm}x{z_cm}_32x32.npy")
    for fname in candidates:
        if os.path.exists(fname):
            bib_np = np.load(fname)
            return torch.from_numpy(bib_np).float().to(DEVICE).view(1,1,IMG_SIZE,IMG_SIZE)
    raise FileNotFoundError(f"Missing BIB for ({xy_cm},{z_cm}). Tried:\n  " + "\n  ".join(candidates))

def load_ddpm_sampler(checkpoint_path):
    model = UNet(
        T=DDPM_CFG["T"], num_energy_labels=len(ENERGY_LABELS),
        ch=DDPM_CFG["channel"], ch_mult=DDPM_CFG["channel_mult"],
        num_res_blocks=DDPM_CFG["num_res_blocks"], dropout=DDPM_CFG["dropout"]
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.",""): v for k,v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()
    sampler = DDIMSampler(
        model, beta_1=DDPM_CFG["beta_1"], beta_T=DDPM_CFG["beta_T"],
        T=DDPM_CFG["T"], eta=DDPM_CFG["eta"], ddim_steps=DDPM_CFG["ddim_steps"]
    ).to(DEVICE)
    return sampler

def find_and_load_images_from_h5(h5_path, energy_value):
    wanted_idx = IDX_MAP[energy_value]
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, 'r') as h5:
        if 'images_xz' in h5.keys():
            img_name = 'images_xz'
        elif 'images_yz' in h5.keys():
            img_name = 'images_yz'
        else:
            img_candidates = [k for k in h5.keys() if k.startswith('images')]
            if not img_candidates:
                return None
            img_name = img_candidates[0]
        labels = np.array(h5['labels_energy'])
        idxs = np.where(labels == wanted_idx)[0]
        if len(idxs) == 0:
            return None
        imgs_np = np.array(h5[img_name])
        imgs_sel = imgs_np[idxs]
    imgs_t = torch.from_numpy(imgs_sel.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)
    H,W = imgs_t.shape[-2], imgs_t.shape[-1]
    if (H,W) != (IMG_SIZE, IMG_SIZE):
        imgs_t = F.interpolate(imgs_t, size=(IMG_SIZE, IMG_SIZE), mode='area')
    return imgs_t

# --- delta-method helper for U SE ---
def U_and_se_from_m(m_array: torch.Tensor, alpha: float = ALPHA) -> Tuple[float,float,float]:
    """
    From an array of per-event MSE values (torch tensor), compute U=1/(alpha^2 + mean_m)
    and the standard error of U via delta-method.
    Returns (U_val, se_U, mean_m)
    """
    m_np = m_array.detach().cpu().numpy().astype(np.float64)
    N = m_np.size
    if N <= 1:
        mean_m = float(m_np.mean()) if N==1 else 0.0
        U_val = 1.0/(alpha**2 + mean_m)
        return float(U_val), 0.0, float(mean_m)
    mean_m = float(m_np.mean())
    var_m = float(m_np.var(ddof=1))
    var_mean_m = var_m / N
    gprime = 1.0 / (alpha**2 + mean_m)**2
    se_U = abs(gprime) * np.sqrt(var_mean_m)
    U_val = 1.0 / (alpha**2 + mean_m)
    return float(U_val), float(se_U), float(mean_m)

# --- compute U and per-event measures for GT images (returns per-event MSE array) ---
def compute_U_and_emeas_from_gt_images(gt_images_cpu: torch.Tensor, bib_5x5: torch.Tensor,
                                       E_true: torch.Tensor, T_fixed: torch.Tensor,
                                       rescale_to_E_true: bool = False, idxs=None):
    """
    Returns: U_scalar, mean_E_meas, mse_per_event (torch tensor), E_meas_per_event (torch tensor)
    This is the uncertainty-aware variant (does not require grads).
    """
    if idxs is None:
        N = gt_images_cpu.shape[0]
        idxs = np.random.choice(N, min(BATCH_SIZE_FD, N), replace=N < BATCH_SIZE_FD)
    imgs = gt_images_cpu[idxs].to(DEVICE)
    imgs.requires_grad_(False)
    E_true_batch = E_true[:len(idxs)].to(DEVICE)

    if rescale_to_E_true:
        sample_sums = imgs.view(len(idxs), -1).sum(dim=1)
        bib_total = bib_5x5.view(1,-1).sum()
        desired_signal = (E_true_batch - bib_total).clamp(min=1e-6)
        sample_sums = sample_sums.clamp(min=1e-6)
        scale = (desired_signal / sample_sums).view(len(idxs),1,1,1)
        scale = torch.clamp(scale, min=1e-6, max=1e6)
        imgs = imgs * scale

    down = pool_to_5x5(imgs)
    combined = down + bib_5x5
    sig_mask = torch.sigmoid(SIGMOID_K*(combined - T_fixed))
    E_meas_per_event = (combined * sig_mask).view(len(idxs), -1).sum(dim=1)
    mse_per_event = (E_meas_per_event - E_true_batch)**2
    mse_per_event = mse_per_event.detach()

    mse_mean = mse_per_event.mean()
    U = 1.0 / (ALPHA**2 + mse_mean)
    return U, float(E_meas_per_event.mean().item()), mse_per_event, E_meas_per_event.detach()

# --- FD evaluation with repeated paired sampling (returns delta-method SE) ---
def eval_central_fd_from_gt(gt_small_path, gt_large_path, cfg_small, cfg_large, eval_cfg,
                            rescale_gt=True, U_mean=None, U_std=None, energy_value=None,
                            n_repeats: int = None):
    """
    Repeated paired FD estimator:
    - Run n_repeats independent paired draws of paired indices (same idxs for small & large)
    - For each repeat compute per-event mse arrays and m_diff = m_large - m_small
    - Concatenate all m_diff arrays and compute mean & se from concatenation
    - Linearize U at midpoint mean_m_mid, then FD = g'(m_mid) * mean(m_diff) / denom
    - SE propagated accordingly
    """
    if n_repeats is None:
        n_repeats = N_REPEATS

    def normalize_xy(xy_cm):
        return (xy_cm - 1.0) / 4.0
    def normalize_z(z_cm):
        return (z_cm - 4.0) / 11.0

    delta_x = normalize_xy(cfg_large[0]) - normalize_xy(cfg_small[0])
    delta_z = normalize_z(cfg_large[1]) - normalize_z(cfg_small[1])

    eps = 1e-12
    if abs(delta_x) < eps: delta_x = 0.0
    if abs(delta_z) < eps: delta_z = 0.0

    bib_32 = load_bib_map_32(eval_cfg[0], eval_cfg[1])
    bib_5 = pool_to_5x5(bib_32).detach()
    mu_bib, sigma_bib = bib_5.mean(), bib_5.std()
    T_fixed = (mu_bib + sigma_bib).view(1,1,1,1)

    fd_sep = []
    bias_center_list = []
    fd_se_list = []

    U_small_vals = []
    U_small_ses = []
    U_large_vals = []
    U_large_ses = []

    is_xy_perturb = (cfg_small[0] != cfg_large[0])
    is_z_perturb  = (cfg_small[1] != cfg_large[1])

    if is_xy_perturb and not is_z_perturb:
        denom = delta_x
        axis_name = "xy"
    elif is_z_perturb and not is_xy_perturb:
        denom = delta_z
        axis_name = "z"
    else:
        if abs(delta_x) >= abs(delta_z) and abs(delta_x) > 0.0:
            denom, axis_name = delta_x, "xy"
        elif abs(delta_z) > 0.0:
            denom, axis_name = delta_z, "z"
        else:
            raise ValueError("Could not detect perturbation axis.")

    if abs(denom) < eps:
        denom = np.sign(denom) * eps if denom != 0.0 else eps

    energies = [energy_value] if energy_value is not None else ENERGY_LABELS
    for E in energies:
        gt_small_imgs = find_and_load_images_from_h5(gt_small_path, E)
        gt_large_imgs = find_and_load_images_from_h5(gt_large_path, E)

        if gt_small_imgs is None or gt_large_imgs is None:
            fd_sep.append(float("nan"))
            fd_se_list.append(float("nan"))
            bias_center_list.append(float("nan"))
            U_small_vals.append(float("nan"))
            U_small_ses.append(float("nan"))
            U_large_vals.append(float("nan"))
            U_large_ses.append(float("nan"))
            continue

        try:
            N_small = gt_small_imgs.shape[0]
            N_large = gt_large_imgs.shape[0]
            N_common = min(N_small, N_large)
            if N_common <= 0:
                raise ValueError("Empty GT arrays for FD")

            # per-repeat paired sample size (bounded by available common events)
            n_per_repeat = min(BATCH_SIZE_FD, N_common)

            # storage for repeats
            m_diff_list = []
            m_small_list = []
            m_large_list = []
            bias_reps = []

            for rep in range(n_repeats):
                replace_flag = False if n_per_repeat <= N_common else True
                idxs = np.random.choice(N_common, n_per_repeat, replace=replace_flag)

                U_small, Eme_small_mean, mse_per_event_small, Eme_small_per_event = compute_U_and_emeas_from_gt_images(
                    gt_small_imgs, bib_5,
                    torch.full((len(idxs),), float(E), device=DEVICE),
                    T_fixed, rescale_to_E_true=rescale_gt, idxs=idxs
                )
                U_large, Eme_large_mean, mse_per_event_large, Eme_large_per_event = compute_U_and_emeas_from_gt_images(
                    gt_large_imgs, bib_5,
                    torch.full((len(idxs),), float(E), device=DEVICE),
                    T_fixed, rescale_to_E_true=rescale_gt, idxs=idxs
                )

                m_small = mse_per_event_small.detach().cpu().numpy().astype(np.float64)
                m_large = mse_per_event_large.detach().cpu().numpy().astype(np.float64)
                m_diff = (m_large - m_small).astype(np.float64)

                m_diff_list.append(m_diff)
                m_small_list.append(m_small)
                m_large_list.append(m_large)
                bias_reps.append(0.5 * ((Eme_small_mean - E) + (Eme_large_mean - E)))

            # concatenate repeats
            all_m_diff = np.concatenate(m_diff_list, axis=0)
            all_m_small = np.concatenate(m_small_list, axis=0)
            all_m_large = np.concatenate(m_large_list, axis=0)

            N_eff = all_m_diff.size
            mean_m_diff = float(np.mean(all_m_diff)) if N_eff > 0 else 0.0
            se_mean_m_diff = float(all_m_diff.std(ddof=1) / np.sqrt(N_eff)) if N_eff > 1 else 0.0

            # compute U and mean_m from concatenated m arrays
            U_small_val, se_small, mean_m_small = U_and_se_from_m(torch.from_numpy(all_m_small).float(), alpha=ALPHA)
            U_large_val, se_large, mean_m_large = U_and_se_from_m(torch.from_numpy(all_m_large).float(), alpha=ALPHA)

            # midpoint for linearization
            mean_m_mid = 0.5 * (mean_m_small + mean_m_large)
            gprime = -1.0 / (ALPHA**2 + mean_m_mid)**2

            # linearized FD and propagated SE
            fd_val = float(gprime * mean_m_diff / denom)
            se_U_diff = abs(gprime) * se_mean_m_diff
            se_fd = float(se_U_diff / abs(denom)) if abs(denom) > 0 else float('nan')

            bias_center = float(np.mean(bias_reps)) if bias_reps else float('nan')

            fd_sep.append(float(fd_val))
            fd_se_list.append(float(se_fd))
            bias_center_list.append(float(bias_center))

            U_small_vals.append(float(U_small_val))
            U_small_ses.append(float(se_small))
            U_large_vals.append(float(U_large_val))
            U_large_ses.append(float(se_large))

        except Exception as ex:
            print(f"   FD FAILED for E={E}, cfg_small={cfg_small}, cfg_large={cfg_large}: {ex}")
            fd_sep.append(float('nan'))
            fd_se_list.append(float('nan'))
            bias_center_list.append(float('nan'))
            U_small_vals.append(float('nan'))
            U_small_ses.append(float('nan'))
            U_large_vals.append(float('nan'))
            U_large_ses.append(float('nan'))

    return {
        "fd_sep_z": fd_sep,
        "fd_se": fd_se_list,
        "bias_center": bias_center_list,
        "U_small": U_small_vals,
        "U_small_se": U_small_ses,
        "U_large": U_large_vals,
        "U_large_se": U_large_ses
    }

# --- MC autodiff eval with visible SE (returns mean & se arrays) ---
def eval_sampler_mc(sampler, config, mc_runs=30):
    """
    Monte-Carlo evaluation to produce mean & standard error for autodiff gradients.
    Returns dict with lists for ENERGY_LABELS in the same order.
    """
    xy_cm, z_cm = config
    bib_32 = load_bib_map_32(xy_cm, z_cm)
    bib_5 = pool_to_5x5(bib_32).detach()
    mu_bib, sigma_bib = bib_5.mean(), bib_5.std()
    T_fixed = (mu_bib + sigma_bib).view(1,1,1,1)

    norm_xy = (xy_cm - 1.0)/4.0
    norm_z = (z_cm - 4.0)/11.0

    # store per-run scalar gradients and per-run U
    gxy_runs = {E: [] for E in ENERGY_LABELS}
    gz_runs  = {E: [] for E in ENERGY_LABELS}
    bias_runs = {E: [] for E in ENERGY_LABELS}
    u_runs = {E: [] for E in ENERGY_LABELS}   # store per-run U values

    for run in range(mc_runs):
        seed = 12345 + run
        set_seed(seed)
        for E in ENERGY_LABELS:
            B = BATCH_SIZE
            xy = torch.full((B,),float(norm_xy),device=DEVICE,requires_grad=True)
            z  = torch.full((B,),float(norm_z),device=DEVICE,requires_grad=True)
            E_true = torch.full((B,),float(E),device=DEVICE)

            noise = torch.randn(B,1,IMG_SIZE,IMG_SIZE,device=DEVICE)
            mat = torch.zeros(B, device=DEVICE)

            samples = sampler(noise, torch.full((B,), IDX_MAP[E], dtype=torch.long, device=DEVICE),
                              xy, z, mat) * 0.5 + 0.5

            sample_sums = samples.view(B,-1).sum(dim=1)
            desired_signal = (E_true - bib_5.view(1,-1).sum()).clamp(min=1e-6)
            scale = (desired_signal / sample_sums).view(B,1,1,1)
            scale = torch.clamp(scale, min=1e-6, max=1e6)
            samples = samples * scale

            down = pool_to_5x5(samples)
            combined = down + bib_5
            sig_mask = torch.sigmoid(SIGMOID_K*(combined - T_fixed))
            E_meas = (combined*sig_mask).view(B,-1).sum(dim=1)   # per-event E_meas
            mse_run = float(((E_meas - E_true)**2).mean().item())  # scalar per-run mse
            U_run = 1.0 / (ALPHA**2 + mse_run)

            # store per-run U
            u_runs[E].append(U_run)

            # compute autodiff gradients using torch tensor U (so autograd works)
            mse = ((E_meas - E_true)**2).mean()
            U_tensor = 1.0 / (ALPHA**2 + mse)
            gxy_tensor, gz_tensor = torch.autograd.grad(U_tensor, [xy, z], retain_graph=False, allow_unused=False)
            gxy_scalar = gxy_tensor.sum().item()
            gz_scalar  = gz_tensor.sum().item()

            bias_runs[E].append(E_meas.mean().item() - E)
            gxy_runs[E].append(gxy_scalar)
            gz_runs[E].append(gz_scalar)
            torch.cuda.empty_cache()

    results = {"dx_mean": [], "dx_se": [], "dz_mean": [], "dz_se": [], "bias_mean": [], "bias_se": [], "U_mean": [], "U_se": []}
    for E in ENERGY_LABELS:
        arr_dx = np.array(gxy_runs[E], dtype=np.float64)
        arr_dz = np.array(gz_runs[E], dtype=np.float64)
        arr_bias = np.array(bias_runs[E], dtype=np.float64)
        arr_u = np.array(u_runs[E], dtype=np.float64)

        mean_dx = float(np.mean(arr_dx)) if arr_dx.size>0 else float('nan')
        se_dx = float(arr_dx.std(ddof=1)/np.sqrt(len(arr_dx))) if len(arr_dx) > 1 else 0.0
        mean_dz = float(np.mean(arr_dz)) if arr_dz.size>0 else float('nan')
        se_dz = float(arr_dz.std(ddof=1)/np.sqrt(len(arr_dz))) if len(arr_dz) > 1 else 0.0
        mean_bias = float(arr_bias.mean()) if arr_bias.size>0 else float('nan')
        se_bias = float(arr_bias.std(ddof=1)/np.sqrt(len(arr_bias))) if len(arr_bias) > 1 else 0.0

        mean_U = float(arr_u.mean()) if len(arr_u)>0 else 0.0
        se_U = float(arr_u.std(ddof=1)/np.sqrt(len(arr_u))) if len(arr_u)>1 else 0.0

        results["dx_mean"].append(mean_dx)
        results["dx_se"].append(se_dx)
        results["dz_mean"].append(mean_dz)
        results["dz_se"].append(se_dz)
        results["bias_mean"].append(mean_bias)
        results["bias_se"].append(se_bias)
        results["U_mean"].append(mean_U)
        results["U_se"].append(se_U)

    return results

# ---------------- MAIN ----------------
def main():
    set_seed()
    print("Loading pre-trained sampler from:", PRETRAIN_CKPT)
    sampler_pre = load_ddpm_sampler(PRETRAIN_CKPT)
    print("Loading post-trained sampler from:", POSTTRAIN_CKPT)
    sampler_post = load_ddpm_sampler(POSTTRAIN_CKPT)

    # point to evaluate
    eval_point = (2.5, 6.0)   # xy_cm, z_cm

    mc_runs = 100   # tune: increase for smaller MC error
    print(f"Running MC autodiff eval with mc_runs={mc_runs}, BATCH_SIZE={BATCH_SIZE} ...")
    pre_res = eval_sampler_mc(sampler_pre, eval_point, mc_runs=mc_runs)
    post_res = eval_sampler_mc(sampler_post, eval_point, mc_runs=mc_runs)

    # ---------------------------------------------------
    # FINITE DIFFERENCE (GT) FOR BOTH AXES (uncertainty-aware) using repeated paired FD
    # ---------------------------------------------------
    cfg_small_z = (2.5, 5.9)
    cfg_large_z = (2.5, 6.1)
    print(f"Computing FD (GT) for z-perturbation (w/ SE) with N_REPEATS={N_REPEATS} ...")
    fd_z_res = eval_central_fd_from_gt(GT_SMALL_Z, GT_LARGE_Z, cfg_small_z, cfg_large_z, eval_point, rescale_gt=False, n_repeats=N_REPEATS)
    fd_dz = np.array(fd_z_res["fd_sep_z"])
    fd_dz_se = np.array(fd_z_res["fd_se"])

    cfg_small_xy = (2.4, 6.0)
    cfg_large_xy = (2.6, 6.0)
    print(f"Computing FD (GT) for xy-perturbation (w/ SE) with N_REPEATS={N_REPEATS} ...")
    fd_xy_res = eval_central_fd_from_gt(GT_SMALL_XY, GT_LARGE_XY, cfg_small_xy, cfg_large_xy, eval_point, rescale_gt=False, n_repeats=N_REPEATS)
    fd_dx = np.array(fd_xy_res["fd_sep_z"])
    fd_dx_se = np.array(fd_xy_res["fd_se"])

    # ---------------------------------------------------
    # PLOT: pre vs post vs FD (both axes) with error bars + shaded 95% CI
    # ---------------------------------------------------
    x = np.array(ENERGY_LABELS)

    pre_dx = np.array(pre_res["dx_mean"])
    pre_dx_err = np.array(pre_res["dx_se"])
    post_dx = np.array(post_res["dx_mean"])
    post_dx_err = np.array(post_res["dx_se"])

    pre_dz = np.array(pre_res["dz_mean"])
    pre_dz_err = np.array(pre_res["dz_se"])
    post_dz = np.array(post_res["dz_mean"])
    post_dz_err = np.array(post_res["dz_se"])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5), sharey=False)

    # LEFT: ∂U/∂Dxy
    if pre_dx.size>0:
        ax1.fill_between(x, pre_dx - 1.96*pre_dx_err, pre_dx + 1.96*pre_dx_err, alpha=0.18)
    if fd_dx.size>0:
        ax1.fill_between(x, fd_dx - 1.96*fd_dx_se, fd_dx + 1.96*fd_dx_se, alpha=0.18)
    if post_dx.size>0:
        ax1.fill_between(x, post_dx - 1.96*post_dx_err, post_dx + 1.96*post_dx_err, alpha=0.12)

    ax1.errorbar(x, pre_dx, yerr=pre_dx_err, fmt='--o', label='DDPM pre-trained ∂U/∂Dxy', capsize=4)
    ax1.errorbar(x, fd_dx, yerr=fd_dx_se, fmt='-d', label='GT ∂U/∂Dxy', capsize=4)
    ax1.errorbar(x, post_dx, yerr=post_dx_err, fmt='--s', label='DDPM post-trained ∂U/∂Dxy', capsize=4)

    ax1.set_xlabel("Energy (GeV)")
    ax1.set_ylabel("Gradient value (∂U/∂Dxy)")
    ax1.set_title(f"Utility gradients at point {eval_point}")
    ax1.grid(True); ax1.legend(fontsize='small')

    # RIGHT: ∂U/∂Dz
    if pre_dz.size>0:
        ax2.fill_between(x, pre_dz - 1.96*pre_dz_err, pre_dz + 1.96*pre_dz_err, alpha=0.18)
    if fd_dz.size>0:
        ax2.fill_between(x, fd_dz - 1.96*fd_dz_se, fd_dz + 1.96*fd_dz_se, alpha=0.18)
    if post_dz.size>0:
        ax2.fill_between(x, post_dz - 1.96*post_dz_err, post_dz + 1.96*post_dz_err, alpha=0.12)

    ax2.errorbar(x, pre_dz, yerr=pre_dz_err, fmt='--o', label='DDPM pre-trained ∂U/∂Dz', capsize=4)
    ax2.errorbar(x, fd_dz, yerr=fd_dz_se, fmt='-d', label='GT ∂U/∂Dz', capsize=4)
    ax2.errorbar(x, post_dz, yerr=post_dz_err, fmt='--s', label='DDPM post-trained ∂U/∂Dz', capsize=4)

    ax2.set_xlabel("Energy (GeV)")
    ax2.set_ylabel("Gradient value (∂U/∂Dz)")
    ax2.set_title(f"Utility gradients at point {eval_point}")
    ax2.grid(True); ax2.legend(fontsize='small')

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, f"prepost_fd_gradients_with_uncert.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved figure:", outpath)

if __name__ == "__main__":
    main()
