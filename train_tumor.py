import os, math, time, copy, random, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torchvision.models import resnet50, ResNet50_Weights

# ------------------------
# Config
# ------------------------
CFG = dict(
    seed=7,
    data_root="data",
    img_dir="raw",
    mask_dir="masks_tumor",
    unlabeled_dir="unlabeled",
    save_dir="runs_nf1_plus",

    # 输入与增强
    img_size=512,
    flip_p=0.5,
    rotate90_p=0.5,
    jitter=0.2,
    gamma_jitter=0.25,
    crop_mode="mix",   # none|random|lesion|mix
    crop_size=320,
    lesion_p=0.9,
    cache_images=True,

    # 训练
    epochs=80,          # ★ 最多 80
    patience=40,
    batch=2,
    unlabeled_batch=2,
    num_workers=0,     # MPS 建议 0 或 2
    grad_clip=0.0,

    # 优化器 & 学习率
    optimizer="adamw",   # adamw|adam|sgd
    lr=1e-4,
    min_lr=1e-6,
    weight_decay=1e-4,
    momentum=0.9,        # adam/adamw 用作 beta1
    nbs=16,              # ref batch for lr scaling
    lr_warmup_epochs=10,
    lr_warmup_start=1e-5,

    # ★ Warmup → Cosine + Warm Restarts（首次重启在第 20 轮）
    lr_policy="warmcos_restart",
    restart_T0=10,      # ★ warmup(10)+T0(10)=20
    restart_Tmult=2,

    # Loss 相关
    loss_type="bce_tversky",  # bce_tversky|focal_tversky|bce_dice
    bce_w=0.4,
    dice_w=0.6,
    tversky_alpha=0.55,
    tversky_beta=0.45,
    # 动态切换到更均衡
    tversky_switch_epoch=12,
    tversky_alpha_late=0.50,
    tversky_beta_late=0.50,

    focal_gamma=0.75,
    pos_weight_cap=25.0,
    init_prior_prob=0.08,

    # 体素占比正则（温和）
    volume_reg_weight=0.10,
    volume_target=0.0,        # 0=自动用正像素率*1.5 限幅[0.02,0.20]

    # 半监督（EMA Teacher + Pseudo-label）
    ema_decay=0.995,
    ema_warmup_epochs=20,
    ema_eval=True,
    unlabeled_weight=0.8,
    unlabeled_ramp_epochs=40,

    # ★ 关键超参（本版新增/调整）
    lam_u_cap=0.6,              # ★ 无监督封顶
    lam_u_freeze_epoch=25,      # ★ 此后不再上升
    sup_only_period=5,          # ★ 每 5 轮监督-only
    teacher_tau=0.7,            # ★ 温度锐化
    conf_drop_low=0.03,         # ★ 整批丢弃阈（下）
    conf_drop_high=0.30,        # ★ 整批丢弃阈（上）
    pseudo_topk_ratio=0.10,     # ★ 每图前 10% 最确定像素
    pseudo_topk_cap=50000,      # ★ 每图最多 50k 像素
    pseudo_thr_high=0.80,
    pseudo_thr_low=0.20,
    pseudo_tta=True,
    pseudo_soft_weight=True,    # 仍做软权，但与 top-k 合并
    pseudo_gamma=1.5,

    # 验证 & 阈值搜索
    tta_val=True,
    th_min=0.05, th_max=0.95, th_step=0.01,

    # 归一化选项（decoder 的 BN 可换 GN；默认不换，仅冻结）
    decoder_norm="bn",         # "bn" | "gn"
    gn_groups=16,

    # 断点续训
    resume="",                 # 路径：best.pt 或 last.ckpt（可为空）
)

IMG_EXTS: Set[str] = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ------------------------
# Utils
# ------------------------
def set_seed(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def pick_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def resize_keep_ratio_pad(im: Image.Image, size: int, is_mask: bool):
    w, h = im.size
    if w == 0 or h == 0: raise ValueError("空图像")
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.NEAREST if is_mask else Image.BILINEAR)
    pad_w, pad_h = size - nw, size - nh
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    return ImageOps.expand(im, border=(left, top, right, bottom), fill=0)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def pil_to_tensor_rgb_norm(im: Image.Image) -> torch.Tensor:
    if im.mode != "RGB": im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC
    x = torch.from_numpy(np.transpose(arr, (2,0,1)))  # CHW
    return (x - IMAGENET_MEAN) / IMAGENET_STD

def pil_mask_to_tensor(im: Image.Image) -> torch.Tensor:
    if im.mode != "L": im = im.convert("L")
    m = np.asarray(im, dtype=np.float32)
    m = (m > 0).astype(np.float32)
    return torch.from_numpy(m)[None, ...]

def list_images(folder: Path) -> List[Path]:
    if not folder.exists(): return []
    return [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]

def make_pairs(img_dir: Path, mask_dir: Path) -> Tuple[List[Path], List[Path]]:
    imgs = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    mask_map = {p.stem: p for p in sorted(mask_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS}
    ipaths, mpaths = [], []
    for ip in imgs:
        mp = mask_map.get(ip.stem, None)
        if mp is not None:
            ipaths.append(ip); mpaths.append(mp)
    if not ipaths:
        raise FileNotFoundError(f"在 {img_dir} 和 {mask_dir} 未找到同名 (image, mask) 对")
    return ipaths, mpaths

def read_split_names(split_dir: Path, name: str) -> List[str]:
    f = split_dir / name
    if not f.exists(): return []
    return [x.strip() for x in open(f, "r", encoding="utf-8") if x.strip()]

def logit_np(p: np.ndarray, eps: float = 1e-6):
    p = np.clip(p, eps, 1 - eps); return np.log(p/(1-p))

def logit_torch(p: torch.Tensor, eps: float = 1e-6):
    p = p.clamp(eps, 1 - eps); return torch.log(p/(1 - p))

# ------------------------
# Dataset
# ------------------------
def _rand_crop_pair(img: torch.Tensor, msk: torch.Tensor, ch: int):
    _, H, W = img.shape
    if ch >= H or ch >= W: return img, msk
    top = random.randint(0, H - ch); left = random.randint(0, W - ch)
    return img[:, top:top+ch, left:left+ch], msk[:, top:top+ch, left:left+ch]

def _lesion_crop_pair(img: torch.Tensor, msk: torch.Tensor, ch: int):
    pos = (msk[0] > 0).nonzero(as_tuple=False)
    if pos.numel() == 0: return _rand_crop_pair(img, msk, ch)
    yx = pos[random.randint(0, pos.shape[0] - 1)]
    cy, cx = int(yx[0]), int(yx[1])
    _, H, W = img.shape
    top = max(0, min(cy - ch // 2, H - ch))
    left = max(0, min(cx - ch // 2, W - ch))
    return img[:, top:top+ch, left:left+ch], msk[:, top:top+ch, left:left+ch]

class LabeledSet(Dataset):
    def __init__(self, imgs, msks, cfg):
        self.imgs, self.msks, self.cfg = imgs, msks, cfg
        self.size = int(cfg["img_size"])
        self.flip_p=float(cfg["flip_p"]); self.rotate90_p=float(cfg["rotate90_p"])
        self.jitter=float(cfg["jitter"]); self.gamma_jitter=float(cfg["gamma_jitter"])
        self.crop_mode=str(cfg["crop_mode"]); self.crop_size=int(cfg["crop_size"]); self.lesion_p=float(cfg["lesion_p"])
        self.cache = []
        if bool(cfg["cache_images"]):
            for ip, mp in zip(imgs, msks):
                im = resize_keep_ratio_pad(Image.open(ip), self.size, False)
                mk = resize_keep_ratio_pad(Image.open(mp), self.size, True)
                self.cache.append((pil_to_tensor_rgb_norm(im), pil_mask_to_tensor(mk)))
        else:
            self.cache=None

    def __len__(self): return len(self.imgs)

    def _get(self, i):
        if self.cache is not None:
            x,y = self.cache[i]; return x.clone(), y.clone()
        im = resize_keep_ratio_pad(Image.open(self.imgs[i]), self.size, False)
        mk = resize_keep_ratio_pad(Image.open(self.msks[i]), self.size, True)
        return pil_to_tensor_rgb_norm(im), pil_mask_to_tensor(mk)

    def __getitem__(self, i):
        x, y = self._get(i)
        if self.flip_p>0 and random.random()<self.flip_p:
            if random.random()<0.5: x = torch.flip(x,[2]); y = torch.flip(y,[2])
            else: x = torch.flip(x,[1]); y = torch.flip(y,[1])
        if self.rotate90_p>0 and random.random()<self.rotate90_p:
            k = random.randint(0,3)
            if k: x = torch.rot90(x,k,[1,2]); y = torch.rot90(y,k,[1,2])
        if self.jitter>0:
            fac = 1.0 + (random.random()*2-1)*self.jitter
            x = x*fac
        if self.gamma_jitter>0:
            g = 1.0 + (random.random()*2-1)*self.gamma_jitter
            x = torch.clamp(x, -5, 5)
            x = torch.sign(x) * (torch.abs(x) ** g)

        if self.crop_mode!="none" and self.crop_size>0:
            ch = min(self.crop_size, self.size)
            use_les = (self.crop_mode=="lesion") or (self.crop_mode=="mix" and random.random()<self.lesion_p)
            xc,yc = (_lesion_crop_pair(x,y,ch) if use_les else _rand_crop_pair(x,y,ch))
            if ch!=self.size:
                x = F.interpolate(xc.unsqueeze(0), size=(self.size,self.size), mode="bilinear", align_corners=False).squeeze(0)
                y = F.interpolate(yc.unsqueeze(0), size=(self.size,self.size), mode="nearest").squeeze(0)
            else:
                x,y = xc,yc
        return x, y

class UnlabeledSet(Dataset):
    def __init__(self, imgs, cfg):
        self.imgs, self.cfg = imgs, cfg
        self.size=int(cfg["img_size"])
        self.flip_p=float(cfg["flip_p"]); self.rotate90_p=float(cfg["rotate90_p"])
        self.jitter=float(cfg["jitter"]); self.gamma_jitter=float(cfg["gamma_jitter"])
        self.crop_mode=str(cfg["crop_mode"]); self.crop_size=int(cfg["crop_size"])
        self.cache=[]
        if bool(cfg["cache_images"]):
            for ip in imgs:
                im = resize_keep_ratio_pad(Image.open(ip), self.size, False)
                self.cache.append(pil_to_tensor_rgb_norm(im))
        else: self.cache=None

    def __len__(self): return len(self.imgs)

    def _get(self, i):
        if self.cache is not None:
            return self.cache[i].clone()
        im = resize_keep_ratio_pad(Image.open(self.imgs[i]), self.size, False)
        return pil_to_tensor_rgb_norm(im)

    def __getitem__(self, i):
        x = self._get(i)
        if self.flip_p>0 and random.random()<self.flip_p:
            x = torch.flip(x, dims=[2] if random.random()<0.5 else [1])
        if self.rotate90_p>0 and random.random()<self.rotate90_p:
            k = random.randint(0,3)
            if k: x = torch.rot90(x,k,[1,2])
        if self.jitter>0:
            fac = 1.0 + (random.random()*2-1)*self.jitter
            x = x*fac
        if self.gamma_jitter>0:
            g = 1.0 + (random.random()*2-1)*self.gamma_jitter
            x = torch.clamp(x, -5, 5)
            x = torch.sign(x) * (torch.abs(x) ** g)
        if self.crop_mode!="none" and self.crop_size>0:
            ch = min(self.crop_size, self.size)
            xc,_ = _rand_crop_pair(x, x.new_zeros(1,*x.shape[-2:]), ch)
            if ch!=self.size:
                x = F.interpolate(xc.unsqueeze(0), size=(self.size,self.size), mode="bilinear", align_corners=False).squeeze(0)
            else: x = xc
        return x

# ------------------------
# Model: ResNet50 Encoder + Attention U-Net Decoder
# ------------------------
class AttentionGate(nn.Module):
    def __init__(self, in_ch_skip, in_ch_g, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(in_ch_skip, inter_ch, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv2d(in_ch_g,   inter_ch, kernel_size=1, bias=False)
        self.psi     = nn.Conv2d(inter_ch, 1, kernel_size=1)
        self.bn      = nn.BatchNorm2d(inter_ch)
        self.act     = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, g):
        theta = self.theta_x(x)
        phi   = self.phi_g(g)
        if theta.shape[-2:] != phi.shape[-2:]:
            phi = F.interpolate(phi, size=theta.shape[-2:], mode="bilinear", align_corners=False)
        f = self.act(self.bn(theta + phi))
        att = self.sigmoid(self.psi(f))
        return x * att

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_att=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.use_att = use_att and (skip_ch>0)
        if self.use_att:
            self.ag = AttentionGate(skip_ch, out_ch, inter_ch=max(out_ch//2, 32))
        conv_in = out_ch + (skip_ch if self.use_att else 0)
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.use_att:
                skip = self.ag(skip, x)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

def replace_bn_with_gn(module: nn.Module, num_groups=16):
    for name, m in module.named_children():
        if isinstance(m, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=m.num_features))
        else:
            replace_bn_with_gn(m, num_groups)

class ResNetAttentionUNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, freeze_bn=True, decoder_norm="bn", gn_groups=16):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet50(weights=weights)

        # Encoder outputs
        self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # C=64, /2
        self.pool  = encoder.maxpool
        self.layer1 = encoder.layer1  # C=256, /4
        self.layer2 = encoder.layer2  # C=512, /8
        self.layer3 = encoder.layer3  # C=1024, /16
        self.layer4 = encoder.layer4  # C=2048, /32

        # Center 1x1 降维
        self.center = nn.Conv2d(2048, 1024, kernel_size=1)

        # Decoder with attention
        self.dec4 = DecoderBlock(1024, 1024, 512, use_att=True)
        self.dec3 = DecoderBlock(512,  512,  256, use_att=True)
        self.dec2 = DecoderBlock(256,  256,  128, use_att=True)
        self.dec1 = DecoderBlock(128,   64,   64, use_att=True)

        if decoder_norm.lower() == "gn":
            replace_bn_with_gn(self.dec1, gn_groups)
            replace_bn_with_gn(self.dec2, gn_groups)
            replace_bn_with_gn(self.dec3, gn_groups)
            replace_bn_with_gn(self.dec4, gn_groups)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        if freeze_bn:
            self._freeze_bn(self)

    @staticmethod
    def _freeze_bn(module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters(): p.requires_grad_(False)

    def forward(self, x):
        x0 = self.conv1(x)               # 64,  /2
        x1 = self.layer1(self.pool(x0))  # 256, /4
        x2 = self.layer2(x1)             # 512, /8
        x3 = self.layer3(x2)             # 1024,/16
        x4 = self.layer4(x3)             # 2048,/32

        c  = self.center(x4)
        d4 = self.dec4(c,  x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        logit = self.out_conv(d1)
        logit = F.interpolate(logit, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logit

# ------------------------
# Loss & Metrics
# ------------------------
def soft_dice(prob: torch.Tensor, tgt: torch.Tensor, eps=1e-6):
    inter = (prob*tgt).sum(dim=(1,2,3))
    union = prob.sum(dim=(1,2,3)) + tgt.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean()

def tversky_loss(prob, tgt, alpha=0.55, beta=0.45, eps=1e-6):
    tp = (prob*tgt).sum((1,2,3))
    fp = (prob*(1-tgt)).sum((1,2,3))
    fn = ((1-prob)*tgt).sum((1,2,3))
    t  = (tp+eps)/(tp + alpha*fp + beta*fn + eps)
    return (1 - t).mean()

def compute_loss_with_ab(logits, tgt, cfg, pos_weight_t, alpha=None, beta=None):
    prob = torch.sigmoid(logits)
    lt = str(cfg["loss_type"]).lower()
    a = float(cfg["tversky_alpha"]) if alpha is None else float(alpha)
    b = float(cfg["tversky_beta"])  if beta  is None else float(beta)
    if lt == "focal_tversky":
        base = tversky_loss(prob, tgt, alpha=a, beta=b)
        return torch.pow(base, float(cfg["focal_gamma"]))
    if lt == "bce_tversky":
        bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight_t)
        tv  = tversky_loss(prob, tgt, alpha=a, beta=b)
        return float(cfg["bce_w"])*bce + float(cfg["dice_w"])*tv
    bce = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight_t)
    dice = soft_dice(prob, tgt)
    return float(cfg["bce_w"])*bce + float(cfg["dice_w"])*(1 - dice)

def volume_regularizer(prob: torch.Tensor, target_ratio: float, w: float):
    if w <= 0: return prob.new_zeros(())
    pred_ratio = prob.mean(dim=(1,2,3))
    return w * (pred_ratio - target_ratio).pow(2).mean()

@torch.no_grad()
def compute_pos_weight(loader, cap=25.0, max_batches=25, device="cpu"):
    pos, tot, n = 0.0, 0.0, 0
    for _,y in loader:
        y = y.to(device)
        pos += y.sum().item()
        tot += y.numel()
        n += 1
        if n>=max_batches: break
    p = max(pos/max(1.0, tot), 1e-6)
    raw = (1.0 - p)/p
    return min(cap, raw), p

# ------------------------
# EMA / TTA
# ------------------------
def copy_model(model: nn.Module):
    ema = copy.deepcopy(model); ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    return ema

@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    esd = ema.state_dict(); msd = model.state_dict()
    for k in esd.keys():
        if k in msd and esd[k].dtype == msd[k].dtype:
            esd[k].lerp_(msd[k], 1.0 - decay)
    ema.load_state_dict(esd, strict=False)

@torch.no_grad()
def predict_tta(net, x):
    probs = []
    p0 = torch.sigmoid(net(x)); probs.append(p0)
    x1 = torch.flip(x, [-1]);        probs.append(torch.sigmoid(net(x1)).flip([-1]))
    x2 = torch.rot90(x, 1, [2,3]);   probs.append(torch.rot90(torch.sigmoid(net(x2)), -1, [2,3]))
    x3 = torch.rot90(x, 2, [2,3]);   probs.append(torch.rot90(torch.sigmoid(net(x3)), -2, [2,3]))
    x4 = torch.rot90(x, 3, [2,3]);   probs.append(torch.rot90(torch.sigmoid(net(x4)), -3, [2,3]))
    return torch.stack(probs, 0).mean(0)

# ------------------------
# Optimizer & LR
# ------------------------
def build_opt_sched(net: nn.Module, cfg: Dict[str,Any], epochs: int):
    opt_type = str(cfg["optimizer"]).lower()
    base_lr = float(cfg["lr"]); min_lr=float(cfg["min_lr"]); wd=float(cfg["weight_decay"])
    momentum=float(cfg["momentum"]); warm=int(cfg["lr_warmup_epochs"])
    nbs=int(cfg["nbs"]); batch=int(cfg["batch"])
    if opt_type in ["adam", "adamw"]:
        lr_limit_max, lr_limit_min = 1e-4, 1e-4
    else:
        lr_limit_max, lr_limit_min = 1e-1, 5e-4
    base_lr_scaled = min(max(batch/nbs*base_lr, lr_limit_min), lr_limit_max)
    min_lr_scaled  = min(max(batch/nbs*min_lr,  lr_limit_min*1e-2), lr_limit_max*1e-2)

    if opt_type == "sgd":
        optimizer = SGD(net.parameters(), lr=base_lr_scaled, momentum=momentum, nesterov=True, weight_decay=wd)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=base_lr_scaled, betas=(momentum,0.999), weight_decay=wd)
    else:
        optimizer = AdamW(net.parameters(), lr=base_lr_scaled, weight_decay=wd)

    policy = str(cfg.get("lr_policy","warmcos_restart")).lower()
    if policy == "warmcos_restart":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg.get("restart_T0", 10)),   # ★ 10
            T_mult=int(cfg.get("restart_Tmult", 2)),
            eta_min=min_lr_scaled
        )
    else:
        def lr_lambda(ep):
            if ep < warm:
                start=float(cfg["lr_warmup_start"])
                alpha=(ep+1)/max(1,warm)
                return (start + (base_lr_scaled-start)*alpha)/base_lr_scaled
            prog=(ep-warm)/max(1,epochs-warm)
            cosine=0.5*(1+math.cos(math.pi*min(1.0,max(0.0,prog))))
            lr_now=min_lr_scaled + (base_lr_scaled-min_lr_scaled)*cosine
            return lr_now/base_lr_scaled
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler, base_lr_scaled, min_lr_scaled

# ------------------------
# Evaluate
# ------------------------
@torch.no_grad()
def evaluate(net, loader, device, cfg, pos_weight_t, alpha, beta):
    """
    保留原先的阈值扫描与 best_thr 挑选（按 Dice 最大），
    并在 best_thr 处计算 IoU / Precision / Recall / PixelAcc / SkinIoU。
    SkinIoU 定义为背景类 IoU = TN / (TN + FP + FN)。
    """
    net.eval()
    ths = np.arange(float(cfg["th_min"]), float(cfg["th_max"])+1e-9,
                    float(cfg["th_step"]), dtype=np.float32).tolist()

    inter_acc   = torch.zeros(len(ths), device=device)  # TP
    pred_sum_acc= torch.zeros(len(ths), device=device)  # 预测为正
    tgt_sum_acc = torch.zeros((), device=device)        # GT 正
    total_pixels= torch.zeros((), device=device)        # 总像素

    total_loss, nb = 0.0, 0
    dice_list=[]

    for x,y in loader:
        x=x.to(device); y=y.to(device)
        logits = net(x)  # 验证损失不用 TTA
        loss = compute_loss_with_ab(logits,y,cfg,pos_weight_t,alpha=alpha,beta=beta)
        total_loss += float(loss.item()); nb += 1

        prob = predict_tta(net,x) if bool(cfg["tta_val"]) else torch.sigmoid(logits)
        dice_list.append(soft_dice(prob,y).item())

        B, C, H, W = prob.shape
        tgt_sum_acc += y.sum()
        total_pixels += torch.tensor(float(B*H*W), device=device)

        for i, th in enumerate(ths):
            pred = (prob>th).to(torch.float32)
            inter_acc[i]    += (pred*y).sum()   # TP
            pred_sum_acc[i] += pred.sum()       # 预测正

    eps=1e-6
    dice_all = (2.0*inter_acc + eps)/(pred_sum_acc + tgt_sum_acc + eps)
    best_idx = int(torch.argmax(dice_all).item())
    best_dice = float(dice_all[best_idx].item())
    best_thr = float(ths[best_idx])
    val_loss = float(total_loss/max(1,nb))

    # 在 best_thr 处一次性求出全部指标
    TP = inter_acc[best_idx]
    pred_sum = pred_sum_acc[best_idx]
    tgt_sum = tgt_sum_acc
    FP = pred_sum - TP
    FN = tgt_sum - TP
    TN = total_pixels - TP - FP - FN

    iou        = (TP / (TP + FP + FN + eps)).item()
    precision  = (TP / (TP + FP + eps)).item()
    recall     = (TP / (TP + FN + eps)).item()
    pixel_acc  = ((TP + TN) / (total_pixels + eps)).item()
    skin_iou   = (TN / (TN + FP + FN + eps)).item()

    metrics = dict(
        avg_dice=float(np.mean(dice_list)) if dice_list else 0.0,
        IoU=iou,
        Precision=precision,
        Recall=recall,
        PixelAcc=pixel_acc,
        SkinIoU=skin_iou
    )
    return val_loss, best_dice, best_thr, metrics

# ------------------------
# Train
# ------------------------
def train(cfg):
    set_seed(int(cfg["seed"]))
    device = pick_device()
    dev_type = device.type

    root = Path(cfg["data_root"])
    img_dir = root / cfg["img_dir"]; mask_dir=root / cfg["mask_dir"]; unl_dir=root / cfg["unlabeled_dir"]
    split_dir = root / "splits"

    img_paths, mask_paths = make_pairs(img_dir, mask_dir)

    # splits 优先
    tr_list=set(read_split_names(split_dir, "train.txt")); va_list=set(read_split_names(split_dir, "val.txt"))
    if tr_list and va_list:
        tr_idx=[i for i,p in enumerate(img_paths) if p.stem in tr_list]
        va_idx=[i for i,p in enumerate(img_paths) if p.stem in va_list]
    else:
        idxs=list(range(len(img_paths))); random.shuffle(idxs)
        n_val=max(1,int(0.2*len(idxs)))
        va_idx=idxs[:n_val]; tr_idx=idxs[n_val:]

    tr_imgs=[img_paths[i] for i in tr_idx]; tr_msks=[mask_paths[i] for i in tr_idx]
    va_imgs=[img_paths[i] for i in va_idx]; va_msks=[mask_paths[i] for i in va_idx]
    ul_imgs=list_images(unl_dir)

    print(f"数据统计: total={len(img_paths)} | train={len(tr_imgs)} val={len(va_imgs)} | unlabeled={len(ul_imgs)} | device={device.type}")

    ds_tr = LabeledSet(tr_imgs, tr_msks, cfg)
    ds_va = LabeledSet(va_imgs, va_msks, {**cfg, "flip_p":0, "rotate90_p":0, "jitter":0, "gamma_jitter":0, "crop_mode":"none", "cache_images":cfg["cache_images"]})
    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["batch"]), shuffle=True, num_workers=int(cfg["num_workers"]), pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=max(1,int(cfg["batch"])*2), shuffle=False, num_workers=0, pin_memory=False)

    # 估计正像素率 -> BCE pos_weight
    est_dl = DataLoader(ds_tr, batch_size=max(1,int(cfg["batch"])*2), shuffle=False, num_workers=0)
    pos_w0, pos_ratio = compute_pos_weight(est_dl, cap=float(cfg["pos_weight_cap"]), device=device)
    cfg["pos_ratio_runtime"] = float(pos_ratio)
    pos_w_min = max(12.0, pos_w0 * 0.4)   # ★ 下限 12

    print(f"[Info] 正像素率≈{pos_ratio:.6f} -> BCE pos_weight0={pos_w0:.2f}"
          f"{' (capped)' if abs(pos_w0 - float(cfg['pos_weight_cap']))<1e-6 else ''}")

    # 模型 & EMA
    net = ResNetAttentionUNet(
        num_classes=1, pretrained=True, freeze_bn=True,
        decoder_norm=str(cfg.get("decoder_norm","bn")), gn_groups=int(cfg.get("gn_groups",16))
    ).to(device)
    with torch.no_grad():
        b = logit_np(np.array([float(cfg["init_prior_prob"])], dtype=np.float32))[0]
        net.out_conv.bias.fill_(float(b))
    ema = copy_model(net) if float(cfg["ema_decay"])>0 else None

    # 断点续训（可选）
    if str(cfg.get("resume","")).strip():
        path = str(cfg["resume"])
        sd = torch.load(path, map_location=device)
        state = sd.get("model", sd) if isinstance(sd, dict) else sd
        missing = net.load_state_dict(state, strict=False)
        if ema is not None:
            ema = copy_model(net)
        print(f"[Resume] loaded from {path}. missing_keys={len(missing.missing_keys)}")

    # 优化器 / 调度器
    optimizer, scheduler, base_lr_scaled, min_lr_scaled = build_opt_sched(net, cfg, int(cfg["epochs"]))
    print(f"[LR] base={base_lr_scaled:.6g}  min={min_lr_scaled:.6g}  warmup={int(cfg['lr_warmup_epochs'])}  policy={cfg.get('lr_policy')}")

    # 计算热重启发生的 epoch（用于 LR 拉回 & 无监督去噪）
    restarts = []
    if str(cfg.get("lr_policy","warmcos_restart")).lower() == "warmcos_restart":
        warm = int(cfg["lr_warmup_epochs"])
        T = int(cfg.get("restart_T0", 10))
        Tmult = int(cfg.get("restart_Tmult", 2))
        e = warm + T  # 第一次重启点
        while e <= int(cfg["epochs"]):
            restarts.append(e)
            T = T * Tmult
            e += T

    # 未标注 loader
    dl_ul=None
    if ul_imgs:
        ds_ul = UnlabeledSet(ul_imgs, cfg)
        dl_ul = DataLoader(ds_ul, batch_size=int(cfg["unlabeled_batch"]), shuffle=True, num_workers=0, pin_memory=False)
        print(f"[UL] 未标注样本数={len(ds_ul)} batch={int(cfg['unlabeled_batch'])}")

    # 训练循环
    save_dir = Path(cfg["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    best_pt = save_dir/"best.pt"; last_ckpt = save_dir/"last.ckpt"
    best_dice, best_thr, best_epoch = -1.0, 0.5, 0
    patience = 0
    t0 = time.time()

    # AMP（仅 CUDA，MPS 关闭）
    use_amp = (dev_type=="cuda")
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    for epoch in range(1, int(cfg["epochs"])+1):
        net.train()
        losses=[]

        # ---------- 半监督权重 ---------- #
        t = min(1.0, epoch/max(1.0, float(cfg["unlabeled_ramp_epochs"])))
        lam_u_raw = (0.5 - 0.5*math.cos(math.pi*t)) * float(cfg["unlabeled_weight"])
        lam_u = min(lam_u_raw, float(cfg["lam_u_cap"]))
        if epoch >= int(cfg["lam_u_freeze_epoch"]):
            lam_u = min(lam_u, float(cfg["lam_u_cap"]))  # 冻结上限
        if (epoch in restarts) or (epoch-1 in restarts) or (epoch+1 in restarts):
            lam_u *= 0.6
        # ★ 每 5 轮监督-only
        if int(cfg["sup_only_period"])>0 and (epoch % int(cfg["sup_only_period"]) == 0):
            lam_u_epoch = 0.0
        else:
            lam_u_epoch = lam_u

        # pos_weight 温和衰减
        r = min(1.0, epoch / max(1.0, float(cfg["unlabeled_ramp_epochs"])))
        pos_w_now = pos_w0*(1.0-r) + pos_w_min*r
        pos_weight_t = torch.tensor([pos_w_now], dtype=torch.float32, device=device)

        # Tversky 动态切换
        alpha = float(cfg["tversky_alpha"])
        beta  = float(cfg["tversky_beta"])
        if epoch >= int(cfg.get("tversky_switch_epoch", 12)):
            alpha = float(cfg.get("tversky_alpha_late", alpha))
            beta  = float(cfg.get("tversky_beta_late",  beta))

        v_w=float(cfg["volume_reg_weight"])
        v_tgt = float(cfg["volume_target"]) if float(cfg["volume_target"])>0 else float(np.clip(cfg["pos_ratio_runtime"]*1.5, 0.02, 0.20))
        th_h=float(cfg["pseudo_thr_high"]); th_l=float(cfg["pseudo_thr_low"])
        ul_iter = iter(dl_ul) if (dl_ul is not None and lam_u_epoch>0) else None

        for x,y in dl_tr:
            x=x.to(device); y=y.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = net(x)
                    sup = compute_loss_with_ab(logits, y, cfg, pos_weight_t, alpha=alpha, beta=beta)
                    prob = torch.sigmoid(logits)
                    reg  = volume_regularizer(prob, v_tgt, v_w)
            else:
                logits = net(x)
                sup = compute_loss_with_ab(logits, y, cfg, pos_weight_t, alpha=alpha, beta=beta)
                prob = torch.sigmoid(logits)
                reg  = volume_regularizer(prob, v_tgt, v_w)

            unsup = x.new_zeros(())
            if ul_iter is not None:
                try:
                    xu = next(ul_iter).to(device)
                except StopIteration:
                    ul_iter = iter(dl_ul)
                    xu = next(ul_iter).to(device)

                with torch.no_grad():
                    if ema is not None and bool(cfg["pseudo_tta"]):
                        p_teacher = predict_tta(ema, xu)
                    elif ema is not None:
                        p_teacher = torch.sigmoid(ema(xu))
                    else:
                        p_teacher = torch.sigmoid(net(xu))

                    # ★ 温度锐化
                    tau = float(cfg.get("teacher_tau", 0.7))
                    if tau != 1.0:
                        p_teacher = torch.sigmoid(logit_torch(p_teacher) / max(1e-3, tau))

                    # 高置信像素占比（用于整批丢弃）
                    hi_mask = ((p_teacher>=th_h) | (p_teacher<=th_l)).float()
                    hi_frac = hi_mask.mean().item()

                    # ★ 批级过滤：在 [3%,30%] 之外则整批丢弃无监督
                    if (hi_frac < float(cfg["conf_drop_low"])) or (hi_frac > float(cfg["conf_drop_high"])):
                        pass
                    else:
                        # ★ Top-k 选最确定像素
                        B, _, H, W = p_teacher.shape
                        conf = torch.abs(p_teacher - 0.5) * 2.0  # [0,1]
                        w = torch.zeros_like(conf)
                        k_ratio = float(cfg.get("pseudo_topk_ratio", 0.10))
                        k_cap = int(cfg.get("pseudo_topk_cap", 50000))
                        N = H*W
                        k = max(1, min(int(N * k_ratio), k_cap))
                        conf_flat = conf.view(B, -1)
                        topk_vals, topk_idx = torch.topk(conf_flat, k=k, dim=1, largest=True, sorted=False)
                        w_flat = torch.zeros_like(conf_flat)
                        w_flat.scatter_(1, topk_idx, 1.0)
                        w = w_flat.view_as(conf)

                        # 与高置信门合取
                        w = w * hi_mask

                        if bool(cfg.get("pseudo_soft_weight", True)):
                            gamma = float(cfg.get("pseudo_gamma", 1.5))
                            w = w * (conf ** gamma)

                        pseudo = (p_teacher>0.5).float()

                        logits_u = net(xu)
                        bce_pix = F.binary_cross_entropy_with_logits(logits_u, pseudo, reduction="none")
                        denom = w.sum().clamp_min(1.0)
                        unsup = (bce_pix * w).sum() / denom

            loss = sup + lam_u_epoch*unsup + reg

            if use_amp:
                scaler.scale(loss).backward()
                if float(cfg["grad_clip"])>0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(net.parameters(), float(cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(cfg["grad_clip"])>0:
                    nn.utils.clip_grad_norm_(net.parameters(), float(cfg["grad_clip"]))
                optimizer.step()

            # EMA 更新（慢热）
            if ema is not None and float(cfg["ema_decay"])>0:
                if epoch <= int(cfg["ema_warmup_epochs"]):
                    decay_now = float(cfg["ema_decay"]) * (epoch/max(1,int(cfg["ema_warmup_epochs"])))**0.5
                else:
                    decay_now = float(cfg["ema_decay"])
                ema_update(ema, net, decay_now)

            losses.append(float(loss.item()))

        # ---- 学习率步进（支持 warmup + 热重启）----
        warm = int(cfg["lr_warmup_epochs"])
        if str(cfg.get("lr_policy","warmcos_restart")).lower() == "warmcos_restart":
            if epoch <= warm:
                start=float(cfg["lr_warmup_start"])
                lr_now = start + (base_lr_scaled - start) * (epoch / max(1,warm))
                for g in optimizer.param_groups: g["lr"] = lr_now
            else:
                scheduler.step()
                if epoch in restarts:
                    for g in optimizer.param_groups:
                        g["lr"] = base_lr_scaled
        else:
            scheduler.step()

        # 验证（可选用 EMA 模型；alpha/beta 用当期设定）
        eval_model = ema if bool(cfg["ema_eval"]) and ema is not None else net
        va_loss, va_dice, va_thr, metrics = evaluate(eval_model, dl_va, device, cfg, pos_weight_t, alpha, beta)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{cfg['epochs']} | train_loss={np.mean(losses):.4f} | "
              f"val_loss={va_loss:.4f} | val_dice={va_dice:.4f} | best_thr≈{va_thr:.2f} | "
              f"lr={lr_now:.6g} | pos_w≈{pos_w_now:.2f} | lam_u={lam_u_epoch:.3f}")
        # 新增：与 Lim 一致的指标打印
        print(f"   IoU={metrics['IoU']:.4f}  Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  Acc={metrics['PixelAcc']:.4f}  "
              f"SkinIoU={metrics['SkinIoU']:.4f}")

        # 保存
        save_dir = Path(cfg["save_dir"])
        best_pt = save_dir/"best.pt"; last_ckpt = save_dir/"last.ckpt"
        torch.save({
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_dice": max(best_dice, va_dice),
            "best_thr": va_thr,
            "cfg": cfg
        }, last_ckpt)

        if va_dice > best_dice:
            best_dice, best_thr, best_epoch = va_dice, va_thr, epoch
            torch.save(net.state_dict(), best_pt)
            tag = "(EMA/eval)" if (eval_model is ema) else ""
            print(f"  ✅ New BEST {tag} saved: {best_pt}  dice={best_dice:.4f} (epoch {epoch})")
            patience = 0
        else:
            patience += 1

        if patience >= int(cfg["patience"]):
            print(f"Early stop at epoch {epoch}. Best dice={best_dice:.4f} @thr={best_thr:.2f} (epoch {best_epoch})")
            break

    print(f"Done. Best dice={best_dice:.4f} @thr={best_thr:.2f} (epoch {best_epoch}). Weights -> {best_pt}")
    print(f"Total time: {time.time()-t0:.1f}s")

# ------------------------
# CLI 覆盖
# ------------------------
def parse_args():
    p = argparse.ArgumentParser("NF1 ResNet-AttentionUNet Semi-supervised Trainer (robust)")
    # 常用覆盖项
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--unlabeled_batch", type=int, default=None)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--loss_type", type=str, default=None)
    p.add_argument("--unlabeled_weight", type=float, default=None)
    p.add_argument("--unlabeled_ramp_epochs", type=float, default=None)
    p.add_argument("--tta_val", type=int, default=None)
    # 学习率策略
    p.add_argument("--lr_policy", type=str, default=None)
    p.add_argument("--restart_T0", type=int, default=None)
    p.add_argument("--restart_Tmult", type=int, default=None)
    # Tversky 动态
    p.add_argument("--tversky_alpha", type=float, default=None)
    p.add_argument("--tversky_beta", type=float, default=None)
    p.add_argument("--tversky_switch_epoch", type=int, default=None)
    p.add_argument("--tversky_alpha_late", type=float, default=None)
    p.add_argument("--tversky_beta_late", type=float, default=None)
    p.add_argument("--pos_weight_cap", type=float, default=None)
    p.add_argument("--volume_reg_weight", type=float, default=None)
    # 归一化
    p.add_argument("--decoder_norm", type=str, default=None)   # bn|gn
    p.add_argument("--gn_groups", type=int, default=None)
    # 伪标签/教师设置
    p.add_argument("--teacher_tau", type=float, default=None)
    p.add_argument("--pseudo_soft_weight", type=int, default=None)
    p.add_argument("--pseudo_gamma", type=float, default=None)
    p.add_argument("--pseudo_thr_high", type=float, default=None)
    p.add_argument("--pseudo_thr_low", type=float, default=None)
    p.add_argument("--pseudo_topk_ratio", type=float, default=None)
    p.add_argument("--pseudo_topk_cap", type=int, default=None)
    p.add_argument("--lam_u_cap", type=float, default=None)
    p.add_argument("--lam_u_freeze_epoch", type=int, default=None)
    p.add_argument("--sup_only_period", type=int, default=None)
    p.add_argument("--conf_drop_low", type=float, default=None)
    p.add_argument("--conf_drop_high", type=float, default=None)
    # 断点续训
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = CFG.copy()
    for k, v in vars(args).items():
        if v is not None:
            if isinstance(CFG.get(k), bool):
                cfg[k] = bool(v)
            else:
                cfg[k] = v
    os.makedirs(cfg["save_dir"], exist_ok=True)
    train(cfg)
