"""
utils_model_v2.py — preprocessing and postprocessing for TwinLiteNetPlus.

All non-twinplus model paths removed. Key fixes:
  - Single definition of letterbox_single_twin
  - preprocess_image: scaleup=True, explicit padding tracking
  - postprocess_masks: logit-space bilinear interpolation before argmax
  - No hardcoded shapes[] global
  - Transform pipeline built once at module level
  - Connected-components filter isolated and optional
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn


def letterbox_single_twin(
    img: np.ndarray,
    new_shape: tuple = (640, 640),
    color: tuple = (114, 114, 114),
    auto: bool = True,
    scaleup: bool = True,
) -> tuple:
    """
    Resize + pad img to new_shape with grey borders (letterbox).

    Returns
    -------
    img_padded          : np.ndarray  padded image
    ratio               : (float, float)  (r_w, r_h) scale applied
    padding_tblr        : (int, int, int, int)  top/bottom/left/right pad pixels
    unpadded_shape_hw   : (int, int)  content size after scale, before pad
    dw                  : float  half-width  padding (for callers that need it)
    dh                  : float  half-height padding
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        r = min(r, 1.0)

    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))

    dw = (new_shape[1] - new_unpad_w) / 2
    dh = (new_shape[0] - new_unpad_h) / 2

    if auto:
        # Snap padding to nearest 32-pixel multiple (model stride alignment)
        dw = np.mod(new_shape[1] - new_unpad_w, 32) / 2
        dh = np.mod(new_shape[0] - new_unpad_h, 32) / 2

    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))

    if (h, w) != (new_unpad_h, new_unpad_w):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    img_padded = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )

    return (
        img_padded,
        (r, r),
        (top, bottom, left, right),
        (new_unpad_h, new_unpad_w),
        dw,
        dh,
    )



def preprocess_twinliteplus(
    image_bgr: np.ndarray,
    target_size: int = 640,
    device: torch.device = None,
    half: bool = False,
) -> dict:
    """
    Preprocess a BGR image for TwinLiteNetPlus inference.

    Returns a dict so callers can pass it straight to postprocess_twinliteplus
    without juggling positional arguments:

        info = preprocess_twinliteplus(frame)
        seg_road, seg_lane = model(info['tensor'])
        road_mask, lane_mask = postprocess_twinliteplus(seg_road, seg_lane, info)

    Dict keys
    ---------
    tensor              : torch.Tensor  [1, 3, H, W] float32/16 on device
    original_shape_hw   : (int, int)
    padded_shape_hw     : (int, int)
    padding_tblr        : (int, int, int, int)
    unpadded_shape_hw   : (int, int)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_shape_hw = image_bgr.shape[:2]

    # BGR → RGB (model trained on RGB)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    img_padded, ratio, padding_tblr, unpadded_shape_hw, _, _ = letterbox_single_twin(
        img_rgb,
        new_shape=(target_size, target_size),
        auto=True,
        scaleup=True,   # always upscale: fills the 640 grid, avoids wasted padding
    )

    # HWC → CHW, [0,1] float, batch dim
    tensor = (
        torch.from_numpy(img_padded.transpose(2, 0, 1))
        .contiguous()
        .float()
        .div_(255.0)
        .unsqueeze_(0)
        .to(device)
    )

    if half and device.type == "cuda":
        tensor = tensor.half()

    return {
        "tensor":             tensor,
        "original_shape_hw":  original_shape_hw,
        "padded_shape_hw":    tuple(tensor.shape[2:]),
        "padding_tblr":       padding_tblr,
        "unpadded_shape_hw":  unpadded_shape_hw,
    }



def postprocess_twinliteplus(
    da_logits: torch.Tensor,
    ll_logits: torch.Tensor,
    preprocess_info: dict,
    improve: bool = True,
) -> tuple:
    """
    Convert raw model logits to binary uint8 masks aligned with the original image.

    Correct order:
        1. Crop padding from logit tensor         (in network-output space)
        2. Bilinear interpolation on logits        (smoother than nearest on labels)
        3. argmax to class labels
        4. Nearest-neighbour resize to orig shape  (no new classes introduced)
        5. Optional: keep only largest road blob

    Parameters
    ----------
    da_logits        : [1, 2, H, W]  drivable-area output
    ll_logits        : [1, 2, H, W]  lane-line output
    preprocess_info  : dict returned by preprocess_twinliteplus
    improve          : if True, discard all but the largest connected component

    Returns
    -------
    road_mask, lane_mask : np.ndarray uint8, 0/255, shape == original_shape_hw
    """
    pad_t, pad_b, pad_l, pad_r = preprocess_info["padding_tblr"]
    out_h, out_w              = preprocess_info["padded_shape_hw"]
    unpadded_hw               = preprocess_info["unpadded_shape_hw"]
    orig_hw                   = preprocess_info["original_shape_hw"]

    road_mask = _decode_logits(da_logits, pad_t, pad_b, pad_l, pad_r,
                               out_h, out_w, unpadded_hw, orig_hw, improve)
    lane_mask = _decode_logits(ll_logits, pad_t, pad_b, pad_l, pad_r,
                               out_h, out_w, unpadded_hw, orig_hw, improve=False)

    return road_mask, lane_mask


def _decode_logits(
    logits: torch.Tensor,
    pad_t: int, pad_b: int, pad_l: int, pad_r: int,
    out_h: int, out_w: int,
    unpadded_hw: tuple,
    orig_hw: tuple,
    improve: bool,
) -> np.ndarray:
    """Internal: unpad → bilinear interp on logits → argmax → resize → binary."""
    with torch.no_grad():
        # 1. Crop padding from logit space
        cropped = logits[:, :, pad_t: out_h - pad_b, pad_l: out_w - pad_r]

        # 2. Bilinear interpolation on logits (better than nearest on hard labels)
        upsampled = F.interpolate(
            cropped,
            size=unpadded_hw,
            mode="bilinear",
            align_corners=False,
        )

        # 3. Argmax → class label map  [1, H, W]
        labels = torch.argmax(upsampled, dim=1)

        # 4. Nearest-neighbour resize to original image shape
        mask = F.interpolate(
            labels.float().unsqueeze(1),
            size=orig_hw,
            mode="nearest",
        ).squeeze().cpu().numpy().astype(np.uint8)

    # 5. Convert to binary 0/255
    mask = (mask > 0).astype(np.uint8) * 255

    # 6. Optional: discard all but largest connected blob
    if improve:
        mask = _keep_largest_blob(mask)

    return mask


def _keep_largest_blob(mask: np.ndarray) -> np.ndarray:
    """
    Return a binary mask containing only the largest connected component.
    Handles the edge case where the mask is completely empty.
    """
    n, labeled, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        # No foreground at all
        return np.zeros_like(mask)
    # stats row 0 is background; find largest foreground label
    foreground_sizes = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(foreground_sizes)) + 1
    return np.where(labeled == largest_label, np.uint8(255), np.uint8(0))



class PAM_Module(Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv   = Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = Conv2d(in_dim, in_dim, 1)
        self.gamma      = Parameter(torch.zeros(1))
        self.softmax    = Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, H * W)
        v = self.value_conv(x).view(B, -1, H * W)
        attn = self.softmax(torch.bmm(q, k))
        out  = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class CAM_Module(Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gamma   = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = x.view(B, C, -1)
        k = x.view(B, C, -1).permute(0, 2, 1)
        e = torch.bmm(q, k)
        e = torch.max(e, -1, keepdim=True)[0].expand_as(e) - e
        attn = self.softmax(e)
        out  = torch.bmm(attn, x.view(B, C, -1)).view(B, C, H, W)
        return self.gamma * out + x


class UPx2(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nIn, nOut, 2, stride=2, bias=False)
        self.bn  = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        p = (kSize - 1) // 2
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act  = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        p = (kSize - 1) // 2
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(nOut, eps=1e-3)

    def forward(self, x):
        return self.bn(self.conv(x))


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        p = (kSize - 1) // 2
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=p, bias=False)

    def forward(self, x):
        return self.conv(x)


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        p = ((kSize - 1) // 2) * d
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=p,
                              dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn  = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        return self.act(self.bn(x))


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n  = nOut // 5
        n1 = nOut - 4 * n
        self.c1  = C(nIn, n, 3, 2)
        self.d1  = CDilated(n, n1, 3, d=1)
        self.d2  = CDilated(n, n,  3, d=2)
        self.d4  = CDilated(n, n,  3, d=4)
        self.d8  = CDilated(n, n,  3, d=8)
        self.d16 = CDilated(n, n,  3, d=16)
        self.bn  = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, x):
        x = self.c1(x)
        d1 = self.d1(x)
        a1 = self.d2(x)
        a2 = a1 + self.d4(x)
        a3 = a2 + self.d8(x)
        a4 = a3 + self.d16(x)
        return self.act(self.bn(torch.cat([d1, a1, a2, a3, a4], 1)))


class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n  = max(nOut // 5, 1)
        n1 = max(nOut - 4 * n, 1)
        self.c1  = C(nIn, n, 1)
        self.d1  = CDilated(n, n1, 3, d=1)
        self.d2  = CDilated(n, n,  3, d=2)
        self.d4  = CDilated(n, n,  3, d=4)
        self.d8  = CDilated(n, n,  3, d=8)
        self.d16 = CDilated(n, n,  3, d=16)
        self.bn  = BR(nOut)
        self.add = add

    def forward(self, x):
        o = self.c1(x)
        d1 = self.d1(o)
        a1 = self.d2(o)
        a2 = a1 + self.d4(o)
        a3 = a2 + self.d8(o)
        a4 = a3 + self.d16(o)
        c  = torch.cat([d1, a1, a2, a3, a4], 1)
        return self.bn(x + c if self.add else c)


class InputProjectionA(nn.Module):
    def __init__(self, samplingTimes):
        super().__init__()
        self.pool = nn.ModuleList(
            [nn.AvgPool2d(3, stride=2, padding=1) for _ in range(samplingTimes)]
        )

    def forward(self, x):
        for p in self.pool:
            x = p(x)
        return x


class ESPNet_Encoder(nn.Module):
    def __init__(self, p=5, q=3):
        super().__init__()
        self.level1    = CBR(3, 16, 3, 2)
        self.sample1   = InputProjectionA(1)
        self.sample2   = InputProjectionA(2)
        self.b1        = CBR(19, 19, 3)
        self.level2_0  = DownSamplerB(19, 64)
        self.level2    = nn.ModuleList([DilatedParllelResidualBlockB(64, 64) for _ in range(p)])
        self.b2        = CBR(131, 131, 3)
        self.level3_0  = DownSamplerB(131, 128)
        self.level3    = nn.ModuleList([DilatedParllelResidualBlockB(128, 128) for _ in range(q)])
        self.b3        = CBR(256, 32, 3)
        self.sa        = PAM_Module(32)
        self.sc        = CAM_Module(32)
        self.conv_sa   = CBR(32, 32, 3)
        self.conv_sc   = CBR(32, 32, 3)
        self.classifier = CBR(32, 32, 1)

    def forward(self, x):
        o0  = self.level1(x)
        i1  = self.sample1(x)
        i2  = self.sample2(x)
        o0c = self.b1(torch.cat([o0, i1], 1))
        o1_0 = self.level2_0(o0c)
        o1 = o1_0
        for layer in self.level2:
            o1 = layer(o1)
        o1c  = self.b2(torch.cat([o1, o1_0, i2], 1))
        o2_0 = self.level3_0(o1c)
        o2 = o2_0
        for layer in self.level3:
            o2 = layer(o2)
        o2c  = self.b3(torch.cat([o2_0, o2], 1))
        s    = self.conv_sa(self.sa(o2c)) + self.conv_sc(self.sc(o2c))
        return self.classifier(s)


class TwinLiteNet(nn.Module):
    def __init__(self, p=2, q=3):
        super().__init__()
        self.encoder      = ESPNet_Encoder(p, q)
        self.up_1_1       = UPx2(32, 16)
        self.up_2_1       = UPx2(16, 8)
        self.up_1_2       = UPx2(32, 16)
        self.up_2_2       = UPx2(16, 8)
        self.classifier_1 = UPx2(8, 2)
        self.classifier_2 = UPx2(8, 2)

    def forward(self, x):
        enc = self.encoder(x)
        x1  = self.classifier_1(self.up_2_1(self.up_1_1(enc)))
        x2  = self.classifier_2(self.up_2_2(self.up_1_2(enc)))
        return x1, x2