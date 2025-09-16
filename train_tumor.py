# -*- coding: utf-8 -*-
"""
NF-1 tumor segmentation trainer (Mac MPS 版, + Focal-Tversky, EMA Teacher, TTA)
- MPS 优先；若不可用退回 CPU
- YAML 驱动；自动配对 data/{raw,masks_tumor}
- 支持 resume / auto_resume / init_ckpt（含可选的 allow_partial/strict_arch）
- Dice 阈值网格搜索（th_min..th_max..th_step）
- 半监督：EMA Teacher + TTA 伪标签 + warm-up；或退化为学生伪标签
- 验证可开启 TTA（tta_val: true）

YAML 关键项示例（放在 configs/train.yaml）：
------------------------------------------------
data_root: data
img_dir: raw
mask_dir: masks_tumor

img_size: 384
crop_mode: mix           # mix / lesion / random / none
lesion_p: 0.9

# 损失（三选一；默认 bce_dice）
loss_type: focal_tversky   # bce_dice | bce_tversky | focal_tversky | combo
tversky_alpha: 0.3
tversky_beta: 0.7
focal_gamma: 0.75
bce_w: 0.25
dice_w: 0.75

# 半监督
unlabeled_dir: unlabeled   # 相对 data_root；或写绝对路径
unlabeled_batch: 2
unlabeled_weight: 0.3
pseudo_thr_high: 0.9
pseudo_thr_low: 0.1
ema_decay: 0.999
ema_eval: true
ema_warmup_epochs: 5
tta_val: true
------------------------------------------------
"""

import os, math, random, argparse, time, copy
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import numpy as np
from PIL import Image

# safetensors（可选）
try:
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file
    HAVE_SAFETENSORS = True
except Exception:
    HAVE_SAFETENSORS = False

# YAML
try:
    import yaml
except Exception:
    raise SystemExit("未安装 PyYAML：请先 `pip install pyyaml`。")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------
# 小工具
# -----------------------
def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(min(float(p), 1 - eps), eps)
    return math.log(p / (1 - p))

def _rand_crop_pair(img: torch.Tensor, msk: torch.Tensor, ch: int):
    _, H, W = img.shape
    if ch >= H or ch >= W:
        return img, msk
    top = random.randint(0, H - ch)
    left = random.randint(0, W - ch)
    return img[:, top:top + ch, left:left + ch], msk[:, top:top + ch, left:left + ch]

def _lesion_crop_pair(img: torch.Tensor, msk: torch.Tensor, ch: int):
    _, H, W = img.shape
    pos = (msk[0] > 0).nonzero(as_tuple=False)  # [N,2] (y,x)
    if pos.numel() == 0:
        return _rand_crop_pair(img, msk, ch)
    yx = pos[random.randint(0, pos.shape[0] - 1)]
    cy, cx = int(yx[0]), int(yx[1])
    top = max(0, min(cy - ch // 2, H - ch))
    left = max(0, min(cx - ch // 2, W - ch))
    return img[:, top:top + ch, left:left + ch], msk[:, top:top + ch, left:left + ch]


# -----------------------
# 数据读取与配对
# -----------------------
IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MSK_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def make_pairs(img_dir: Path, mask_dir: Path) -> Tuple[List[Path], List[Path]]:
    if not img_dir.exists():
        raise FileNotFoundError(f"未找到图像目录：{img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"未找到掩膜目录：{mask_dir}")

    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not imgs:
        raise FileNotFoundError(f"未在 {img_dir} 找到图像文件；支持扩展名：{sorted(IMG_EXTS)}")

    mask_map = {}
    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in MSK_EXTS:
            mask_map[p.stem] = p

    img_paths, mask_paths = [], []
    for ip in imgs:
        mp = mask_map.get(ip.stem, None)
        if mp is not None:
            img_paths.append(ip)
            mask_paths.append(mp)

    if not img_paths:
        raise FileNotFoundError(
            f"在 {img_dir} 与 {mask_dir} 没有找到同名 (image, mask) 对。请确保掩膜与图像同名（仅后缀不同）。"
        )
    return img_paths, mask_paths

def pil_to_array_gray(im: Image.Image) -> np.ndarray:
    """转单通道 float32 [0,1]"""
    if im.mode != "L":
        im = im.convert("L")
    arr = np.asarray(im, dtype=np.float32)
    if arr.max() > 1.5:
        arr /= 255.0
    return arr

def _resize_pair(im: Image.Image, mk: Image.Image, size: int):
    im = im.resize((size, size), Image.BILINEAR)
    mk = mk.resize((size, size), Image.NEAREST)
    return im, mk

def load_image_mask(img_path: Path, msk_path: Path, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    im = Image.open(img_path)
    mk = Image.open(msk_path)
    im, mk = _resize_pair(im, mk, size)
    im = pil_to_array_gray(im)  # [H,W] -> [0,1]
    mk = np.asarray(mk, dtype=np.float32)
    mk = (mk > 0).astype(np.float32)
    im_t = torch.from_numpy(im)[None, ...]  # [1,H,W]
    mk_t = torch.from_numpy(mk)[None, ...]
    return im_t, mk_t


class SegDataset(Dataset):
    def __init__(
        self,
        img_paths: List[Path],
        mask_paths: List[Path],
        img_size=512,
        flip_p=0.5,
        jitter=0.2,
        rotate90_p=0.5,
        gamma_jitter=0.25,
        crop_mode="mix",   # "none" | "lesion" | "random" | "mix"
        crop_size=256,
        lesion_p=0.8,
        cache_images=True,
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.size = int(img_size)
        self.flip_p = float(flip_p)
        self.jitter = float(jitter)
        self.rotate90_p = float(rotate90_p)
        self.gamma_jitter = float(gamma_jitter)
        self.crop_mode = str(crop_mode)
        self.crop_size = int(crop_size)
        self.lesion_p = float(lesion_p)
        self.cache_images = bool(cache_images)

        self._cache: List[Tuple[torch.Tensor, torch.Tensor]] | None = None
        if self.cache_images:
            self._cache = []
            for ip, mp in zip(self.img_paths, self.mask_paths):
                img, msk = load_image_mask(ip, mp, self.size)
                self._cache.append((img, msk))

    def __len__(self):
        return len(self.img_paths)

    def _get_base(self, i):
        if self._cache is not None:
            return self._cache[i][0].clone(), self._cache[i][1].clone()
        return load_image_mask(self.img_paths[i], self.mask_paths[i], self.size)

    def __getitem__(self, i):
        img, msk = self._get_base(i)

        # 轻增强
        if self.flip_p > 0 and random.random() < self.flip_p:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2]); msk = torch.flip(msk, dims=[2])
            else:
                img = torch.flip(img, dims=[1]); msk = torch.flip(msk, dims=[1])

        if self.rotate90_p > 0 and random.random() < self.rotate90_p:
            k = random.randint(0, 3)
            if k:
                img = torch.rot90(img, k, dims=[1, 2])
                msk = torch.rot90(msk, k, dims=[1, 2])

        if self.jitter > 0:
            fac = 1.0 + (random.random() * 2 - 1) * self.jitter
            img = torch.clamp(img * fac, 0.0, 1.0)

        if self.gamma_jitter > 0:
            g = 1.0 + (random.random() * 2 - 1) * self.gamma_jitter
            img = torch.clamp(img ** g, 0.0, 1.0)

        # 病灶感知/随机裁剪
        if self.crop_mode != "none" and self.crop_size > 0:
            ch = min(self.crop_size, self.size)
            use_lesion = (self.crop_mode == "lesion") or (self.crop_mode == "mix" and random.random() < self.lesion_p)
            img_c, msk_c = (_lesion_crop_pair(img, msk, ch) if use_lesion else _rand_crop_pair(img, msk, ch))
            if ch != self.size:
                img = F.interpolate(img_c.unsqueeze(0), size=(self.size, self.size), mode="bilinear",
                                    align_corners=False).squeeze(0)
                msk = F.interpolate(msk_c.unsqueeze(0), size=(self.size, self.size), mode="nearest").squeeze(0)
            else:
                img, msk = img_c, msk_c

        return img, msk


# ====== 未标注数据集 ======
class UnlabeledDataset(Dataset):
    """未标注图像：仅输出 img，增强策略与有标注保持一致（去掉基于掩膜的病灶裁剪）"""
    def __init__(self, img_paths: List[Path], img_size=512,
                 flip_p=0.5, jitter=0.2, rotate90_p=0.5, gamma_jitter=0.25,
                 crop_mode="mix", crop_size=256, lesion_p=0.8, cache_images=True):
        self.img_paths = img_paths
        self.size = int(img_size)
        self.flip_p = float(flip_p)
        self.jitter = float(jitter)
        self.rotate90_p = float(rotate90_p)
        self.gamma_jitter = float(gamma_jitter)
        self.crop_mode = str(crop_mode)
        self.crop_size = int(crop_size)
        self.lesion_p = float(lesion_p)
        self.cache_images = bool(cache_images)

        self._cache: List[torch.Tensor] | None = None
        if self.cache_images:
            self._cache = []
            for ip in self.img_paths:
                im = Image.open(ip)
                im = im.resize((self.size, self.size), Image.BILINEAR)
                arr = pil_to_array_gray(im)
                self._cache.append(torch.from_numpy(arr)[None, ...])

    def __len__(self):
        return len(self.img_paths)

    def _get_base(self, i):
        if self._cache is not None:
            return self._cache[i].clone()
        im = Image.open(self.img_paths[i])
        im = im.resize((self.size, self.size), Image.BILINEAR)
        arr = pil_to_array_gray(im)
        return torch.from_numpy(arr)[None, ...]

    def __getitem__(self, i):
        img = self._get_base(i)

        # 轻增强
        if self.flip_p > 0 and random.random() < self.flip_p:
            img = torch.flip(img, dims=[2] if random.random() < 0.5 else [1])

        if self.rotate90_p > 0 and random.random() < self.rotate90_p:
            k = random.randint(0, 3)
            if k:
                img = torch.rot90(img, k, dims=[1, 2])

        if self.jitter > 0:
            fac = 1.0 + (random.random() * 2 - 1) * self.jitter
            img = torch.clamp(img * fac, 0.0, 1.0)

        if self.gamma_jitter > 0:
            g = 1.0 + (random.random() * 2 - 1) * self.gamma_jitter
            img = torch.clamp(img ** g, 0.0, 1.0)

        # 随机裁剪
        if self.crop_mode != "none" and self.crop_size > 0:
            ch = min(self.crop_size, self.size)
            img_c, _ = _rand_crop_pair(img, img.new_zeros(1, *img.shape[-2:]), ch)
            if ch != self.size:
                img = F.interpolate(img_c.unsqueeze(0), size=(self.size, self.size),
                                    mode="bilinear", align_corners=False).squeeze(0)
            else:
                img = img_c
        return img


# -----------------------
# 模型：轻量 UNet
# -----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="gn", groups=8, drop=0.0):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(groups, c)) if norm == "gn" else (lambda c: nn.BatchNorm2d(c))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, base=16, depth=4, norm="gn", drop=0.05):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_ch
        for c in chs:
            self.downs.append(DoubleConv(c_in, c, norm=norm, drop=drop))
            self.pools.append(nn.MaxPool2d(2))
            c_in = c
        self.center = DoubleConv(chs[-1], chs[-1] * 2, norm=norm, drop=drop)

        self.ups = nn.ModuleList()
        c = chs[-1] * 2
        for dc in reversed(chs):
            self.ups.append(nn.ConvTranspose2d(c, dc, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(c if dc == chs[-1] else dc * 2, dc, norm=norm, drop=drop))
            c = dc
        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        feats = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            feats.append(x)
            x = pool(x)
        x = self.center(x)
        for i in range(0, len(self.ups), 2):
            up = self.ups[i]
            dc = self.ups[i + 1]
            x = up(x)
            skip = feats[-(i // 2 + 1)]
            # pad if needed
            if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
                dh = skip.shape[-2] - x.shape[-2]
                dw = skip.shape[-1] - x.shape[-1]
                x = F.pad(x, [0, max(0, dw), 0, max(0, dh)])
                x = x[:, :, : skip.shape[-2], : skip.shape[-1]]
            x = torch.cat([skip, x], dim=1)
            x = dc(x)
        return self.out_conv(x)


# -----------------------
# 损失 & 指标
# -----------------------
def soft_dice_prob(prob: torch.Tensor, target: torch.Tensor, eps=1e-6):
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean()

# === Tversky / Focal-Tversky / Combo ===
def tversky_loss(prob: torch.Tensor, target: torch.Tensor, alpha=0.3, beta=0.7, eps=1e-6):
    tp = (prob * target).sum(dim=(1,2,3))
    fp = (prob * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - prob) * target).sum(dim=(1,2,3))
    t = (tp + eps) / (tp + alpha*fp + beta*fn + eps)
    return (1 - t).mean()

def focal_tversky_loss(prob: torch.Tensor, target: torch.Tensor, alpha=0.3, beta=0.7, gamma=0.75):
    lt = tversky_loss(prob, target, alpha=alpha, beta=beta)
    return torch.pow(lt, gamma)

def combo_loss(logits: torch.Tensor, target: torch.Tensor, bce_w=0.25, t_alpha=0.3, t_beta=0.7):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    tv = tversky_loss(prob, target, alpha=t_alpha, beta=t_beta)
    return bce_w*bce + (1.0-bce_w)*tv

def compute_loss(logits: torch.Tensor, target: torch.Tensor, cfg: Dict[str, Any], pos_weight_t: torch.Tensor):
    loss_type = str(cfg.get("loss_type", "bce_dice")).lower()
    prob = torch.sigmoid(logits)

    if loss_type == "focal_tversky":
        alpha = float(cfg.get("tversky_alpha", 0.3))
        beta  = float(cfg.get("tversky_beta",  0.7))
        gamma = float(cfg.get("focal_gamma",   0.75))
        return focal_tversky_loss(prob, target, alpha=alpha, beta=beta, gamma=gamma)

    if loss_type == "bce_tversky":
        alpha = float(cfg.get("tversky_alpha", 0.3))
        beta  = float(cfg.get("tversky_beta",  0.7))
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_t)
        tv  = tversky_loss(prob, target, alpha=alpha, beta=beta)
        w_bce = float(cfg.get("bce_w", 0.3)); w_tv = float(cfg.get("dice_w", 0.7))
        return w_bce*bce + w_tv*tv

    if loss_type == "combo":
        return combo_loss(logits, target,
                          bce_w=float(cfg.get("bce_w", 0.25)),
                          t_alpha=float(cfg.get("tversky_alpha", 0.3)),
                          t_beta=float(cfg.get("tversky_beta", 0.7)))

    # 默认：bce + dice
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_t)
    dice_soft = soft_dice_prob(prob, target)
    return float(cfg.get("bce_w", 0.3)) * bce + float(cfg.get("dice_w", 0.7)) * (1 - dice_soft)

def compute_pos_weight(dloader: DataLoader, cap=50.0, max_batches=50, device="cpu"):
    """估计像素级正样本比例 -> pos_weight = (1-p)/p (上限cap)"""
    pos, tot, n = 0.0, 0.0, 0
    for _, y in dloader:
        y = y.to(device)
        pos += y.sum().item()
        tot += y.numel()
        n += 1
        if n >= max_batches:
            break
    p = max(pos / max(1.0, tot), 1e-6)
    raw = (1.0 - p) / p
    return min(cap, raw), p


# -----------------------
# EMA / TTA
# -----------------------
def copy_model(model: nn.Module):
    ema = copy.deepcopy(model)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema

@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    esd = ema.state_dict()
    msd = model.state_dict()
    for k in esd.keys():
        if k in msd and esd[k].dtype == msd[k].dtype:
            esd[k].lerp_(msd[k], 1.0 - decay)
    ema.load_state_dict(esd, strict=False)

@torch.no_grad()
def predict_proba_tta(net, x: torch.Tensor):
    # 5 个视图：原图 + Hflip + 3*旋转
    probs = []
    p = torch.sigmoid(net(x)); probs.append(p)
    x1 = torch.flip(x, dims=[-1]); probs.append(torch.sigmoid(net(x1)).flip(dims=[-1]))
    x2 = torch.rot90(x, 1, dims=[-2,-1]); probs.append(torch.rot90(torch.sigmoid(net(x2)), -1, dims=[-2,-1]))
    x3 = torch.rot90(x, 2, dims=[-2,-1]); probs.append(torch.rot90(torch.sigmoid(net(x3)), -2, dims=[-2,-1]))
    x4 = torch.rot90(x, 3, dims=[-2,-1]); probs.append(torch.rot90(torch.sigmoid(net(x4)), -3, dims=[-2,-1]))
    return torch.stack(probs, dim=0).mean(0)

def cosine_warmup(epoch, max_w, warmup_epochs=5):
    if warmup_epochs <= 0:
        return max_w
    if epoch <= warmup_epochs:
        return max_w * epoch / max(1, warmup_epochs)
    return max_w


# -----------------------
# 训练（MPS 友好）
# -----------------------
def run_epoch(net, loader, optimizer, device, cfg, pos_weight_t, grad_clip=0.0,
              unl_iter=None, ema_model=None, ema_decay=0.0,
              epoch=1, warmup_epochs=5):
    net.train()
    losses = []
    lambda_u_max = float(cfg.get("unlabeled_weight", 0.0))
    lam_u = cosine_warmup(epoch, lambda_u_max, warmup_epochs=warmup_epochs)
    th_h = float(cfg.get("pseudo_thr_high", 0.9))
    th_l = float(cfg.get("pseudo_thr_low", 0.1))

    for x, y in loader:
        x = x.to(device, non_blocking=False)
        y = y.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)

        # 有标注监督损失
        logits = net(x)
        loss_sup = compute_loss(logits, y, cfg, pos_weight_t)

        # 无标注：EMA Teacher + TTA 伪标签
        loss_unsup = x.new_zeros(())
        if (unl_iter is not None) and (lam_u > 0.0):
            try:
                x_u = next(unl_iter)
            except StopIteration:
                x_u = None
            if x_u is not None:
                x_u = x_u.to(device, non_blocking=False)
                with torch.no_grad():
                    if ema_model is not None:
                        p_teacher = predict_proba_tta(ema_model, x_u)
                    else:
                        p_teacher = torch.sigmoid(net(x_u))
                    conf_mask = ((p_teacher >= th_h) | (p_teacher <= th_l)).float()
                    pseudo = (p_teacher > 0.5).float()
                logits_u = net(x_u)
                bce_pix = F.binary_cross_entropy_with_logits(logits_u, pseudo, reduction="none")
                denom = conf_mask.sum().clamp_min(1.0)
                loss_unsup = (bce_pix * conf_mask).sum() / denom

        loss = loss_sup + lam_u * loss_unsup
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        # EMA 每步更新
        if ema_model is not None and ema_decay > 0:
            ema_update(ema_model, net, decay=ema_decay)

        losses.append(float(loss.item()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_once(net, loader, device, cfg: Dict[str, Any], pos_weight_t: torch.Tensor,
                  th_min: float, th_max: float, th_step: float):
    net.eval()
    eps = 1e-6
    thresholds = np.arange(th_min, th_max + 1e-9, th_step).astype(np.float32).tolist()

    # ✅ MPS 不支持 float64，统一用 float32
    dtype_accum = torch.float32

    total_loss = 0.0
    n_batches = 0

    inter = torch.zeros(len(thresholds), dtype=dtype_accum, device=device)
    pred_sum = torch.zeros(len(thresholds), dtype=dtype_accum, device=device)
    tgt_sum = torch.zeros((), dtype=dtype_accum, device=device)

    pred_pos_list, prob_mean_list = [], []
    empty_gt_cnt, sample_cnt = 0, 0

    use_tta = bool(cfg.get("tta_val", False))

    for x, y in loader:
        x = x.to(device); y = y.to(device)

        logits = net(x)  # 用于 val_loss（不做 TTA）
        loss = compute_loss(logits, y, cfg, pos_weight_t)
        total_loss += float(loss.item()); n_batches += 1

        # 用 TTA 的概率来做 Dice/阈值搜索（若开启）
        prob = predict_proba_tta(net, x) if use_tta else torch.sigmoid(logits)

        pred_pos_list.append((prob > 0.5).float().mean().item())
        prob_mean_list.append(prob.mean().item())
        empty_gt_cnt += (y.sum(dim=(1, 2, 3)) == 0).sum().item()
        sample_cnt += y.size(0)

        tgt_sum += y.sum(dtype=dtype_accum)
        for i, th in enumerate(thresholds):
            pred = (prob > th).to(torch.float32)
            inter[i] += (pred * y).sum(dtype=dtype_accum)
            pred_sum[i] += pred.sum(dtype=dtype_accum)

    union = pred_sum + tgt_sum + eps
    dice_all = (2.0 * inter + eps) / union
    best_idx = int(torch.argmax(dice_all).item())
    best_dice = float(dice_all[best_idx].item())
    best_thr = float(thresholds[best_idx])
    val_loss = float(total_loss / max(1, n_batches))

    metrics = {
        "pred_pos@0.5": float(np.mean(pred_pos_list)) if pred_pos_list else 0.0,
        "prob_mean": float(np.mean(prob_mean_list)) if prob_mean_list else 0.0,
        "empty_gt_ratio": float(empty_gt_cnt / max(1, sample_cnt)),
    }
    return val_loss, best_dice, best_thr, metrics


# -----------------------
# 保存/加载（严格匹配可配置）
# -----------------------
def save_ckpt(path, model, optimizer, scheduler, epoch, best_dice, best_thr, args_dict):
    os.makedirs(Path(path).parent, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_dice": best_dice,
        "best_thr": best_thr,
        "args": args_dict,
    }
    torch.save(ckpt, path)

def _load_state(model: nn.Module, state_dict: Dict[str, torch.Tensor],
                allow_partial: bool, strict_arch: bool):
    model_state = model.state_dict()
    matched, skipped = {}, []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    if (not allow_partial) or strict_arch:
        if skipped:
            raise RuntimeError(f"[严格匹配] 发现 {len(skipped)} 个不匹配参数，终止加载（设置 allow_partial:true 或 strict_arch:false 可放宽）。")
        model.load_state_dict(state_dict, strict=True)
        return 0

    # allow_partial
    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)
    if skipped:
        print(f"[warn] 部分加载：跳过 {len(skipped)} 个不匹配参数")
    return len(skipped)

def _torch_load_any(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_any_weights(path, model, optimizer=None, scheduler=None,
                     map_location="cpu", allow_partial=False, strict_arch=True):
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext == ".safetensors":
        if not HAVE_SAFETENSORS:
            raise SystemExit("未安装 safetensors：`pip install safetensors` 后再用 .safetensors")
        sd = safe_load_file(path, device=map_location)
        _load_state(model, sd, allow_partial=allow_partial, strict_arch=strict_arch)
        return "model-only(safetensors)", None

    obj = _torch_load_any(path, map_location=map_location)

    if isinstance(obj, dict) and "model" in obj:
        _load_state(model, obj["model"], allow_partial=allow_partial, strict_arch=strict_arch)
        if optimizer is not None and obj.get("optimizer") is not None:
            optimizer.load_state_dict(obj["optimizer"])
        if scheduler is not None and obj.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(obj["scheduler"])
            except Exception as e:
                print(f"[warn] 调度器状态恢复失败：{e}")
        return "full", obj

    if isinstance(obj, dict):
        _load_state(model, obj, allow_partial=allow_partial, strict_arch=strict_arch)
        return "model-only", None

    raise RuntimeError(f"无法识别的权重格式：{path}")


# -----------------------
# YAML 配置读取
# -----------------------
DEFAULT_CFG_REL = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"

def load_config(path_from_cli: str | None) -> dict:
    if path_from_cli:
        cfg_path = Path(path_from_cli)
    else:
        cfg_path = DEFAULT_CFG_REL
    if not cfg_path.exists():
        raise SystemExit(f"未提供 --config，且默认配置不存在：{cfg_path}\n"
                         f"请创建该文件，或通过 --config 指定配置路径。")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[CFG] 已加载配置：{cfg_path}")
    return cfg


# -----------------------
# 主训练入口（MPS）
# -----------------------
def main(cfg: dict):
    # 1) 随机种子
    set_seed(int(cfg.get("seed", 7)))

    # 2) 设备（仅 MPS；若不可用则退回 CPU）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 3) 数据路径
    data_root = Path(cfg.get("data_root", "data"))
    img_dir_cfg = Path(cfg.get("img_dir", "raw"))
    mask_dir_cfg = Path(cfg.get("mask_dir", "masks_tumor"))
    img_dir = img_dir_cfg if img_dir_cfg.is_absolute() else data_root / img_dir_cfg
    mask_dir = mask_dir_cfg if mask_dir_cfg.is_absolute() else data_root / mask_dir_cfg

    img_paths, mask_paths = make_pairs(img_dir, mask_dir)

    # 过拟合自检（可选）
    overfit_k = int(cfg.get("overfit_k", 0))
    if overfit_k > 0:
        img_paths = img_paths[: overfit_k]
        mask_paths = mask_paths[: overfit_k]

    # 4) 划分 train/val
    val_split = float(cfg.get("val_split", 0.2))
    idxs = list(range(len(img_paths)))
    random.shuffle(idxs)
    n_total = len(idxs)
    n_val = max(1, int(n_total * (0.25 if overfit_k > 0 else val_split)))
    val_ids = set(idxs[:n_val])
    train_ids = [i for i in idxs if i not in val_ids]

    if bool(cfg.get("overfit_val_on_train", False)):
        train_ids = idxs
        val_ids = set(idxs)

    train_img = [img_paths[i] for i in train_ids]
    train_msk = [mask_paths[i] for i in train_ids]
    val_img = [img_paths[i] for i in val_ids]
    val_msk = [mask_paths[i] for i in val_ids]

    print(f"数据统计：total={n_total} | train={len(train_img)} val={len(val_img)} | device={device.type} | "
          f"dirs=({img_dir.name}, {mask_dir.name})")

    # 5) 数据集 / DataLoader
    img_size = int(cfg.get("img_size", 384))
    ds_tr = SegDataset(
        train_img, train_msk,
        img_size=img_size,
        flip_p=float(cfg.get("flip_p", 0.5)),
        jitter=float(cfg.get("jitter", 0.2)),
        rotate90_p=float(cfg.get("rotate90_p", 0.5)),
        gamma_jitter=float(cfg.get("gamma_jitter", 0.25)),
        crop_mode=str(cfg.get("crop_mode", "mix")),
        crop_size=int(cfg.get("crop_size", max(128, img_size // 2))),
        lesion_p=float(cfg.get("lesion_p", 0.8)),
        cache_images=bool(cfg.get("cache_images", True)),
    )
    ds_va = SegDataset(
        val_img, val_msk,
        img_size=img_size,
        flip_p=0.0, jitter=0.0, rotate90_p=0.0, gamma_jitter=0.0,
        crop_mode="none", crop_size=0, lesion_p=0.0,
        cache_images=bool(cfg.get("cache_images", True)),
    )

    batch = int(cfg.get("batch", 2))
    num_workers = int(cfg.get("num_workers", 0))   # MPS 上建议 0 或 2；过大反而慢
    pin_mem = False
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_mem)
    dl_va = DataLoader(ds_va, batch_size=max(1, batch * 2), shuffle=False,
                       num_workers=num_workers, pin_memory=pin_mem)

    # 6) 估计 BCE pos_weight
    pos_cap = float(cfg.get("pos_weight_cap", 20.0))
    est_dl = DataLoader(ds_tr, batch_size=max(1, batch * 2), shuffle=False, num_workers=0)
    pos_w_val, pos_ratio = compute_pos_weight(est_dl, cap=pos_cap, max_batches=25, device=device)
    print(f"[Info] 估计像素正例率≈{pos_ratio:.6f} -> BCE pos_weight={pos_w_val:.2f}"
          f"{' (capped)' if pos_w_val == pos_cap else ''}")
    pos_weight_t = torch.tensor([pos_w_val], dtype=torch.float32, device=device)

    # 7) 模型
    net = UNet(
        in_ch=1,
        base=int(cfg.get("base", 16)),
        depth=int(cfg.get("depth", 4)),
        norm=str(cfg.get("norm", "gn")),
        drop=float(cfg.get("dropout", 0.05)),
    ).to(device)

    # 首次训练：输出层偏置=先验概率（若非续训、且未指定 init_ckpt）
    is_fresh = not (cfg.get("resume") or cfg.get("auto_resume")) and not cfg.get("init_ckpt")
    if is_fresh:
        prior = float(cfg.get("init_prior_prob", 0.1))
        with torch.no_grad():
            net.out_conv.bias.fill_(_logit(prior))
        print(f"[Init] out_conv.bias <- logit({prior:.4f}) = {_logit(prior):.3f}")

    # EMA Teacher（按需）
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema_model = copy_model(net) if ema_decay > 0 else None

    # Loss info
    lt = str(cfg.get("loss_type", "bce_dice")).lower()
    print(f"[Loss] using {lt}")

    opt = AdamW(
        net.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    sched = ReduceLROnPlateau(
        opt, mode="min", factor=0.5,
        patience=int(cfg.get("plateau_patience", 10)),
        min_lr=float(cfg.get("min_lr", 1e-6)),
    )

    grad_clip = float(cfg.get("grad_clip", 0.0))

    # 8) 保存目录 & 续训策略
    save_dir = Path(str(cfg.get("save_dir", "runs_mps")))
    os.makedirs(save_dir, exist_ok=True)
    best_pt = str(save_dir / "best.pt")
    best_ckpt = str(save_dir / "best.ckpt")
    last_ckpt = str(save_dir / "last.ckpt")

    start_epoch, best_dice, best_thr = 1, -1.0, 0.5

    resume = cfg.get("resume", "")
    resume_full = bool(cfg.get("resume_full", False))
    allow_partial = bool(cfg.get("allow_partial", False))
    strict_arch = bool(cfg.get("strict_arch", True))

    if (not resume) and bool(cfg.get("auto_resume", False)):
        if Path(last_ckpt).exists():
            resume, resume_full = last_ckpt, True
        elif Path(best_ckpt).exists():
            resume, resume_full = best_ckpt, True
        elif Path(best_pt).exists():
            resume, resume_full = best_pt, False

    loaded_obj = None
    if resume and Path(resume).exists():
        rtype, obj = load_any_weights(
            resume, net, opt if resume_full else None, sched if resume_full else None,
            map_location=device, allow_partial=allow_partial, strict_arch=strict_arch
        )
        print(f"==> Loaded {rtype} weights from {resume}. {'带优化器/调度器' if resume_full else '仅模型'}")
        loaded_obj = obj
        if rtype == "full" and isinstance(obj, dict):
            start_epoch = int(obj.get("epoch", 0)) + 1
            best_dice = float(obj.get("best_dice", -1.0))
            best_thr = float(obj.get("best_thr", 0.5))
    else:
        init_ckpt = cfg.get("init_ckpt", "")
        if init_ckpt and Path(init_ckpt).exists():
            _rtype, _ = load_any_weights(init_ckpt, net, optimizer=None, scheduler=None,
                                         map_location=device, allow_partial=allow_partial, strict_arch=strict_arch)
            print(f"==> Model initialized from {init_ckpt} ({_rtype})")

    # === 未标注数据 ===
    unlabeled_dir = str(cfg.get("unlabeled_dir", "")).strip()
    dl_ul, unl_iter = None, None
    if unlabeled_dir:
        u = Path(unlabeled_dir)
        if not u.is_absolute():
            # 容错：避免 data_root="data" + unlabeled_dir="data/unlabeled" -> data/data/unlabeled
            parts = u.parts
            if len(parts) >= 2 and parts[0] == str(data_root):
                u = Path(*parts[1:])
            udir = (data_root / u)
        else:
            udir = u
        if udir.exists():
            ul_imgs = [p for p in sorted(udir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
            if ul_imgs:
                ds_ul = UnlabeledDataset(
                    ul_imgs,
                    img_size=img_size,
                    flip_p=float(cfg.get("flip_p", 0.5)),
                    jitter=float(cfg.get("jitter", 0.2)),
                    rotate90_p=float(cfg.get("rotate90_p", 0.5)),
                    gamma_jitter=float(cfg.get("gamma_jitter", 0.25)),
                    crop_mode=str(cfg.get("crop_mode", "mix")),
                    crop_size=int(cfg.get("crop_size", max(128, img_size // 2))),
                    lesion_p=float(cfg.get("lesion_p", 0.8)),
                    cache_images=bool(cfg.get("cache_images", True)),
                )
                dl_ul = DataLoader(ds_ul, batch_size=int(cfg.get("unlabeled_batch", batch)),
                                   shuffle=True, num_workers=0, pin_memory=False)
                unl_iter = iter(dl_ul)
                print(f"[UL] 未标注样本数={len(ds_ul)} batch={int(cfg.get('unlabeled_batch', batch))}")
            else:
                print(f"[UL] 未在 {udir} 找到未标注图像")
        else:
            print(f"[UL] 未标注目录不存在：{udir}")

    # 9) 训练循环
    epochs = int(cfg.get("epochs", 200))
    patience = int(cfg.get("patience", 20))
    th_min, th_max, th_step = (
        float(cfg.get("th_min", 0.05)),
        float(cfg.get("th_max", 0.95)),
        float(cfg.get("th_step", 0.05)),
    )

    patience_counter = 0
    t0 = time.time()
    best_epoch = start_epoch

    for epoch in range(start_epoch, epochs + 1):
        tr_loss = run_epoch(
            net, dl_tr, opt, device, cfg, pos_weight_t, grad_clip=grad_clip,
            unl_iter=iter(dl_ul) if dl_ul is not None else None,  # 每个 epoch 刷新迭代器
            ema_model=ema_model, ema_decay=ema_decay,
            epoch=epoch, warmup_epochs=int(cfg.get("ema_warmup_epochs", 5)),
        )

        # eval: 用 EMA（若开启且 ema_eval=true），并可启用 TTA 指标
        use_ema_for_eval = bool(cfg.get("ema_eval", False)) and (ema_model is not None)
        model_eval = ema_model if use_ema_for_eval else net

        va_loss, va_dice, va_thr, metrics = evaluate_once(
            model_eval, dl_va, device, cfg, pos_weight_t, th_min, th_max, th_step
        )

        sched.step(va_loss)
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"val_dice={va_dice:.4f} | best_thr≈{va_thr:.2f} | lr={lr_now:.6f} | "
            f"pred_pos@0.5={metrics['pred_pos@0.5']:.6f} prob_mean={metrics['prob_mean']:.4f} "
            f"empty_gt={metrics['empty_gt_ratio']:.2f}"
        )

        # 保存最优
        if va_dice > best_dice:
            best_dice, best_thr, best_epoch = va_dice, va_thr, epoch
            torch.save(net.state_dict(), best_pt)
            save_ckpt(best_ckpt, net, opt, sched, epoch, best_dice, best_thr, cfg)
            if HAVE_SAFETENSORS:
                safe_path = str(Path(best_pt).with_suffix(".safetensors"))
                safe_save_file(net.state_dict(), safe_path)
            print(f"✅ New BEST saved: {best_pt}  dice={best_dice:.4f} (epoch {epoch})")
            patience_counter = 0
        else:
            patience_counter += 1

        # 每个 epoch 保存最近断点
        save_ckpt(last_ckpt, net, opt, sched, epoch, best_dice, best_thr, cfg)

        if patience_counter >= patience:
            print(f"Early stop at epoch {epoch}. Best dice={best_dice:.4f} @thr={best_thr:.2f} (epoch {best_epoch})")
            break

    print(f"Done. Best dice={best_dice:.4f} @thr={best_thr:.2f} (epoch {best_epoch}). Weights -> {best_pt}")
    print(f"Total time: {time.time()-t0:.1f}s")


# -----------------------
# 启动入口
# -----------------------
def build_parser():
    p = argparse.ArgumentParser(description="NF-1 tumor segmentation trainer (MPS + Focal-Tversky + EMA + TTA)")
    p.add_argument("--config", default="", help="YAML 配置路径。留空则使用 <repo_root>/configs/train.yaml")
    p.add_argument("--resume", default=None, help="优先覆盖 YAML 的 resume")
    p.add_argument("--resume_full", action="store_true", help="覆盖 YAML 的 resume_full=True")
    p.add_argument("--save_dir", default=None, help="覆盖 YAML 的 save_dir")
    return p

def merge_cli_over_yaml(yaml_cfg: dict, args):
    cfg = dict(yaml_cfg)
    if args.resume is not None:
        cfg["resume"] = args.resume
    if args.resume_full:
        cfg["resume_full"] = True
    if args.save_dir is not None:
        cfg["save_dir"] = args.save_dir
    return cfg

def train_cli(cli_args=None):
    parser = build_parser()
    args = parser.parse_args(cli_args)
    yaml_cfg = load_config(args.config)
    cfg = merge_cli_over_yaml(yaml_cfg, args)
    main(cfg)

if __name__ == "__main__":
    train_cli()
