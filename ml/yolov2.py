"""YOLOv2-VOC (416 px) em PyTorch: modelo, pesos Darknet, decodificacao e mAP VOC2007.

Serve ao artigo ISCAS (Winograd-FPGA): as 3x3 com N_IC <= 512 rodam no acelerador
(`yolov2_cfg.no_acelerador`), o resto no host. A leaky ReLU fica no host: o netlist roda essas
convs com `relu_en=0` (saida com sinal) e o maxpool comuta com a leaky, entao os ciclos nao mudam.
`act="leaky"` + pesos do pjreddie reproduz o original (sanidade: ~76,8 mAP@0,5 no VOC2007 test).

`--numerics`:
  fp32   BN fundida na conv (a mesma fusao que vira o bias do netlist), sem quantizacao
  int8   INT8 padrao em TODA conv: ativacao por tensor, peso por canal (o custo so' da conversao)
  f23/f43/f63  o int8 com as convs do acelerador trocadas pela numerica empacotada da ordem
         (qat_wino.make_wino_conv, pack=True, escalas FIXAS do netlist: hw_params)
A calibracao passa `--calib` imagens do trainval (sem gradiente, em train()) e depois avalia.

    python ml/yolov2.py --weights ~/.cache/yolov2/yolov2-voc.weights --limit 200   # rapido
    python ml/yolov2.py --weights ~/.cache/yolov2/yolov2-voc.weights --numerics f43 --out-json m.json
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.ops import batched_nms

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
ANCHORS = torch.tensor([[1.3221, 1.73145], [3.19275, 4.00944], [5.05587, 8.09892],
                        [9.47112, 4.84053], [11.2364, 10.0071]])   # yolov2-voc.cfg, em celulas
SIZE, GRID, NA, NC = 416, 13, 5, 20
# yolov2-voc.cfg ate' a camada 24: (C_out, kernel) ou "M" (maxpool 2x2/2)
BACKBONE = [(32, 3), "M", (64, 3), "M", (128, 3), (64, 1), (128, 3), "M", (256, 3), (128, 1), (256, 3), "M",
            (512, 3), (256, 1), (512, 3), (256, 1), (512, 3), "M", (1024, 3), (512, 1), (1024, 3), (512, 1),
            (1024, 3), (1024, 3), (1024, 3)]
PASSTHROUGH_AT = 16   # indice (em BACKBONE) da conv 26x26x512 que o [route -9] le


def _act(kind: str) -> nn.Module:
    return nn.LeakyReLU(0.1) if kind == "leaky" else nn.ReLU()


def conv_bn(cin: int, cout: int, k: int, act: str) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, 1, k // 2, bias=False), nn.BatchNorm2d(cout, eps=1e-6), _act(act))


def reorg(x: torch.Tensor) -> torch.Tensor:
    """O [reorg] stride 2 do Darknet (reorg_cpu com forward=0), que NAO e' um space-to-depth comum:
    le a entrada (B,64,26,26) como (B,16,52,52) e escreve saida[(dy*2+dx)*16+c, j, i] =
    entrada[c, 2j+dy, 2i+dx]; o buffer de (B,64,26,26) vira (B,256,13,13). Sem isto os pesos do
    pjreddie nao casam."""
    b, c, h, w = x.shape
    v = x.reshape(b, c // 4, h, 2, w, 2)                  # [c2, j, dy, i, dx] de (16,52,52)
    return v.permute(0, 3, 5, 1, 2, 4).reshape(b, c, h, w).reshape(b, c * 4, h // 2, w // 2)


class YOLOv2(nn.Module):
    def __init__(self, act: str = "leaky"):
        super().__init__()
        layers, cin = [], 3
        for spec in BACKBONE:
            if spec == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(conv_bn(cin, spec[0], spec[1], act))
                cin = spec[0]
        self.backbone = nn.ModuleList(layers)
        self.pass_conv = conv_bn(512, 64, 1, act)
        self.head = conv_bn(256 + 1024, 1024, 3, act)
        self.out = nn.Conv2d(1024, NA * (5 + NC), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = None
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i == PASSTHROUGH_AT:
                route = x
        x = torch.cat([reorg(self.pass_conv(route)), x], 1)   # [route -1,-4]: reorg primeiro
        return self.out(self.head(x))

    def convs(self):
        """As convs na ORDEM do arquivo de pesos do Darknet."""
        return [m for m in self.backbone if isinstance(m, nn.Sequential)] + [self.pass_conv, self.head, self.out]


def load_darknet(model: YOLOv2, path: str) -> None:
    buf = np.fromfile(path, dtype=np.uint8)
    major, minor, _ = np.frombuffer(buf[:12], dtype=np.int32)
    off = 12 + (8 if major * 10 + minor >= 2 else 4)                  # `seen` int64 ou int32
    w = np.frombuffer(buf[off:].tobytes(), dtype=np.float32)
    p = 0

    def take(t: torch.Tensor):
        nonlocal p
        n = t.numel()
        t.data.copy_(torch.from_numpy(w[p:p + n].copy()).view_as(t))
        p += n
    for s in model.convs():
        if isinstance(s, nn.Sequential):   # Darknet: beta, gamma, media, variancia, pesos
            c, bn = s[0], s[1]
            for t in (bn.bias, bn.weight, bn.running_mean, bn.running_var, c.weight):
                take(t)
        else:
            take(s.bias)
            take(s.weight)
    assert p == w.size, f"sobraram {w.size - p} floats: arquitetura nao casa com {path}"


def decode(out: torch.Tensor, conf: float = 0.005, nms: float = 0.45, top: int = 200):
    """Saida (B,125,13,13) -> por imagem (caixas xyxy em [0,1], score, classe 0..19)."""
    b = out.shape[0]
    o = out.view(b, NA, 5 + NC, GRID, GRID).permute(0, 1, 3, 4, 2)            # B,A,gy,gx,25
    gy, gx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
    cx = (o[..., 0].sigmoid() + gx) / GRID
    cy = (o[..., 1].sigmoid() + gy) / GRID
    wh = o[..., 2:4].exp() * ANCHORS.view(1, NA, 1, 1, 2) / GRID
    box = torch.stack([cx - wh[..., 0] / 2, cy - wh[..., 1] / 2, cx + wh[..., 0] / 2, cy + wh[..., 1] / 2], -1)
    prob = o[..., 4:5].sigmoid() * o[..., 5:].softmax(-1)                     # objectness x classe
    res = []
    for i in range(b):
        bx, pr = box[i].reshape(-1, 4), prob[i].reshape(-1, NC)
        idx, cls = (pr > conf).nonzero(as_tuple=True)
        bb, sc = bx[idx], pr[idx, cls]
        k = batched_nms(bb, sc, cls, nms)[:top]
        res.append((bb[k].clamp(0, 1), sc[k], cls[k]))
    return res


def voc_gt(root: Path, iid: str):
    r = ET.parse(root / "Annotations" / f"{iid}.xml").getroot()
    out = []
    for o in r.iter("object"):
        bb = o.find("bndbox")
        out.append((VOC_CLASSES.index(o.find("name").text), int(o.find("difficult").text),
                    [float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]))
    return out


def voc07_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """AP de 11 pontos do VOC2007 (o metodo do 76,8 publicado)."""
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in np.arange(0, 1.1, 0.1)]))


def voc07_map(dets: dict, gts: dict) -> tuple[float, list[float]]:
    """dets[classe] = [(iid, score, caixa)]; gts[iid] = voc_gt(). IoU 0,5; difficult ignorado."""
    aps = []
    for c in range(NC):
        g = {i: [(np.array(b), d) for cc, d, b in v if cc == c] for i, v in gts.items()}
        npos = sum(1 for v in g.values() for _, d in v if not d)
        used = {i: [False] * len(v) for i, v in g.items()}
        ds = sorted(dets.get(c, []), key=lambda t: -t[1])
        tp, fp = np.zeros(len(ds)), np.zeros(len(ds))
        for k, (iid, _, bb) in enumerate(ds):
            best, j = 0.0, -1
            for jj, (gb, _) in enumerate(g.get(iid, [])):
                ix = max(0.0, min(bb[2], gb[2]) - max(bb[0], gb[0]) + 1)
                iy = max(0.0, min(bb[3], gb[3]) - max(bb[1], gb[1]) + 1)
                inter = ix * iy
                uni = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (gb[2] - gb[0] + 1) * (gb[3] - gb[1] + 1) - inter
                if inter / uni > best:
                    best, j = inter / uni, jj
            if best > 0.5:
                if g[iid][j][1]:
                    continue                      # difficult: nem TP nem FP
                if not used[iid][j]:
                    tp[k], used[iid][j] = 1, True
                else:
                    fp[k] = 1
            else:
                fp[k] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        aps.append(voc07_ap(tp / max(npos, 1), tp / np.maximum(tp + fp, 1e-9)))
    return float(np.mean(aps)), aps


def _batch(voc2007: Path, chunk: list[str]):
    ims = [Image.open(voc2007 / "JPEGImages" / f"{i}.jpg").convert("RGB") for i in chunk]
    x = torch.stack([torch.from_numpy(np.array(im.resize((SIZE, SIZE), Image.BILINEAR))).permute(2, 0, 1)
                     for im in ims]).float() / 255
    return ims, x


class Int8Conv(nn.Module):
    """INT8 padrao (o do fbgemm): ativacao simetrica por tensor (max da calibracao), peso simetrico
    por canal de saida, acumulacao exata, bias em float. E' a numerica do host e a da linha `int8`."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.register_buffer("act_absmax", torch.zeros(()))

    def forward(self, x):
        if self.training:
            self.act_absmax.copy_(torch.maximum(self.act_absmax, x.detach().abs().amax()))
        s = self.act_absmax.clamp(min=1e-8) / 127
        xq = torch.clamp(torch.round(x / s), -128, 127) * s
        w = self.conv.weight
        sw = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127
        wq = torch.clamp(torch.round(w / sw), -127, 127) * sw
        return F.conv2d(xq, wq, self.conv.bias, self.conv.stride, self.conv.padding)


HW_PARAMS = {"f23": "f23", "f43": "f43", "f63": "f63ab18"}   # os mesmos do `_hw` do AlexNet
# ponytail: a mesma raiz do ml.winograd_bridge, sem importar o pacote `ml` (puxa torchmetrics)
BRIDGE = Path(os.environ.get("WINOGRAD_FPGA_ROOT",
                             Path.home() / "Documents/Winograd-FPGA/scripts/avaliacao_redes")).expanduser()


def _bridge(name: str):
    if str(BRIDGE) not in sys.path:
        sys.path.insert(0, str(BRIDGE))
    return importlib.import_module(name)


class F64(nn.Module):
    """Roda a conv do acelerador em float64: la' a aritmetica inteira e' EXATA (acumuladores de ate'
    52 b cabem em 53). Em float32 o acumulador passa de 2^24 e 0,004% das saidas da c13 do F(4,3)
    sairam 1 LSB fora (medido contra float64 com o bias zerado)."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m.double()

    def forward(self, x):
        return self.m(x.double()).float()


def quantize(model: YOLOv2, numerics: str, pack: str = "hw") -> list[str]:
    """Funde BN e troca as convs pela numerica pedida. Devolve as convs que foram para o acelerador.
    `pack`: hw = packing com as escalas FIXAS do netlist (o que roda); adaptive = packing com `sv`
    do V real de cada camada (limite superior, exige mudar o RTL); off = Winograd INT8 sem packing."""
    for s in model.convs():
        if isinstance(s, nn.Sequential) and isinstance(s[1], nn.BatchNorm2d):
            s[0] = fuse_conv_bn_eval(s[0].eval(), s[1].eval())
            s[1] = nn.Identity()
    if numerics == "fp32":
        return []
    cfg = _bridge("yolov2_cfg")
    W = None
    if numerics != "int8":
        W = _bridge("qat_wino").make_wino_conv(numerics, pack=pack != "off",
                                               hw_params=HW_PARAMS[numerics] if pack == "hw" else None)
    no_accel = []
    for (name, _, cin, cout, k), s in zip(cfg.LAYERS, model.convs(), strict=True):
        conv = s[0] if isinstance(s, nn.Sequential) else s
        assert (conv.in_channels, conv.out_channels, conv.kernel_size[0]) == (cin, cout, k), name
        if W is not None and cfg.no_acelerador((name, None, cin, cout, k)):
            assert conv.stride[0] == 1 and conv.groups == 1 and conv.padding[0] == 1, name
            w = W(cin, cout, bias=True)             # relu=False: piso -128, a leaky vem depois
            w.weight.data.copy_(conv.weight.data)
            w.bias.data.copy_(conv.bias.data)
            novo = F64(w)
            no_accel.append(name)
        else:
            novo = Int8Conv(conv)
        if isinstance(s, nn.Sequential):
            s[0] = novo
        else:
            model.out = novo
    return no_accel


@torch.no_grad()
def calibrate(model: nn.Module, voc2007: Path, n: int, batch: int) -> None:
    """Enche act_absmax (host e acelerador) e o post_shift do acelerador. Semente fixa: o
    percentil do post_shift amostra quando o acumulador passa de 1M valores."""
    torch.manual_seed(0)
    ids = (voc2007 / "ImageSets/Main/trainval.txt").read_text().split()[:n]
    model.train()
    for s in range(0, len(ids), batch):
        model(_batch(voc2007, ids[s:s + batch])[1])
    model.eval()
    bias_inteiro(model)


def bias_inteiro(model: nn.Module) -> None:
    """Escalas do netlist: o bias tem de ser o INTEIRO do bias.mem (escala do acumulador, BIAS_W=26).
    O qat_wino soma `bias/(s_x*s_w)*2^(PS-ab)` em float; com o bias cru isso cai entre inteiros e
    0,2% das saidas saem 1 LSB fora do RTL (medido na c09 real). Nem todo inteiro e' atingivel por
    essa conta em float32, mas nao precisa: o requant e' floor((Y + meio + bias) / 2^s) com Y
    inteiro, e somar um delta em [0, 1) a um inteiro nunca muda esse floor. Entao o bias float e'
    empurrado, ulp a ulp, ate' a MESMA conta do forward dar inteiro + delta, com 0 <= delta < 0,5."""
    for m in model.modules():
        if getattr(m, "ab_hw", None) is None or m.bias is None:
            continue
        k = 2.0 ** (_bridge("qat_wino").variant_consts(m.variant)["POST_SHIFT"] - m.ab_hw)
        P = m._act_scale(None) * m._w_scale()
        alvo = torch.clamp(torch.round(m.bias.data / P * k), -(2 ** 25), 2 ** 25 - 1)
        b = alvo / k * P
        inf = torch.full_like(b, float("inf"))
        for _ in range(64):
            baixo = b / P * k < alvo
            if not baixo.any():
                break
            b = torch.where(baixo, torch.nextafter(b, inf), b)
        d = b / P * k - alvo
        assert bool(((d >= 0) & (d < 0.5)).all()), "bias nao ficou em [inteiro, inteiro+0,5)"
        m.bias.data.copy_(b)


@torch.no_grad()
def evaluate(model: nn.Module, voc2007: Path, limit: int | None = None,
             batch: int = 16) -> tuple[float, list[float]]:
    ids = (voc2007 / "ImageSets/Main/test.txt").read_text().split()[:limit]
    model.eval()
    dets, gts, t0 = {}, {}, time.time()
    for s in range(0, len(ids), batch):
        chunk = ids[s:s + batch]
        ims, x = _batch(voc2007, chunk)
        for iid, im, (bb, sc, cl) in zip(chunk, ims, decode(model(x))):
            gts[iid] = voc_gt(voc2007, iid)
            scale = torch.tensor([im.width, im.height, im.width, im.height])
            for b, p, c in zip((bb * scale).tolist(), sc.tolist(), cl.tolist()):
                dets.setdefault(c, []).append((iid, p, b))
        print(f"\r[{s + len(chunk)}/{len(ids)}] {time.time() - t0:.0f}s", end="", flush=True)
    print()
    return voc07_map(dets, gts)


def _selftest() -> None:
    x = torch.arange(64 * 26 * 26, dtype=torch.float32).view(1, 64, 26, 26)
    ref = torch.empty(64 * 26 * 26)                        # reorg_cpu do Darknet, literal
    flat = x.flatten()
    for k in range(64):
        for j in range(26):
            for i in range(26):
                c2, off = k % 16, k // 16
                ref[i + 26 * (j + 26 * k)] = flat[(i * 2 + off % 2) + 52 * ((j * 2 + off // 2) + 52 * c2)]
    assert torch.equal(reorg(x).flatten(), ref)
    assert YOLOv2()(torch.zeros(1, 3, SIZE, SIZE)).shape == (1, 125, 13, 13)
    assert abs(voc07_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5])) - (6 * 1.0 + 5 * 0.5) / 11) < 1e-9
    torch.manual_seed(0)                                   # fundir a BN nao muda a saida
    m, x = YOLOv2(), torch.rand(1, 3, SIZE, SIZE)
    for s in m.convs():
        if isinstance(s, nn.Sequential):
            s[1].running_mean.uniform_(-0.1, 0.1), s[1].running_var.uniform_(0.5, 2), s[1].weight.data.uniform_(0.5, 2)
    y0 = m.eval()(x)
    quantize(m, "fp32")
    assert torch.allclose(m(x), y0, rtol=1e-3, atol=1e-3), (m(x) - y0).abs().max()
    print("yolov2: selftest OK")


def _git(path: Path) -> str | None:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights")
    ap.add_argument("--act", default="leaky", choices=("leaky", "relu"))
    ap.add_argument("--numerics", choices=("orig", "fp32", "int8", "f23", "f43", "f63"), default="orig",
                    help="orig = o modelo cru, sem fundir a BN (o comportamento antigo)")
    ap.add_argument("--pack", choices=("hw", "adaptive", "off"), default="hw",
                    help="so' f23/f43/f63: escalas do netlist (o que roda), adaptativas, ou sem packing")
    ap.add_argument("--calib", type=int, default=200, help="imagens do trainval para calibrar")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--voc", default=str(Path.home() / ".cache/torchvision/datasets/VOCdevkit/VOC2007"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out-json", type=Path)
    a = ap.parse_args()
    _selftest()
    if a.weights:
        t0 = time.time()
        m = YOLOv2(a.act)
        load_darknet(m, a.weights)
        no_accel = [] if a.numerics == "orig" else quantize(m, a.numerics, a.pack)
        if a.numerics not in ("orig", "fp32"):
            calibrate(m, Path(a.voc), a.calib, a.batch)
        mp, aps = evaluate(m, Path(a.voc), a.limit, a.batch)
        print(f"mAP@0.5 (VOC07, 11 pontos, act={a.act}, numerics={a.numerics}, pack={a.pack}): {100 * mp:.2f}")
        if a.out_json:
            wino = {n: dict(act_absmax=float(c.act_absmax), post_shift=int(round(float(c.post_shift))),
                            sat_frac=float(c.sat_frac))
                    for n, c in m.named_modules() if hasattr(c, "post_shift")}
            rec = dict(net="yolov2_voc_416", weights=a.weights, act=a.act, numerics=a.numerics,
                       pack=a.pack if no_accel else None,
                       hw_params=HW_PARAMS.get(a.numerics) if a.pack == "hw" else None, calib=a.calib, batch=a.batch,
                       limit=a.limit, n_test=a.limit or 4952, dataset="VOC2007 test", metric="mAP@0.5 VOC07 11-pt",
                       map=round(100 * mp, 2), ap_per_class=dict(zip(VOC_CLASSES, (round(100 * v, 2) for v in aps))),
                       no_acelerador=no_accel, dnn_study=_git(Path(__file__).parent),
                       bridge=_git(BRIDGE), wino_calib=wino or None,
                       seconds=round(time.time() - t0), date=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            a.out_json.parent.mkdir(parents=True, exist_ok=True)
            a.out_json.write_text(json.dumps(rec, indent=1, default=str))
