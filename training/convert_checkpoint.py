"""
Конвертирует checkpoint TRAIN_1024 -> формат Ctrl-System.

  Различия:
    1) В моём UNet: Conv2d(bias=False), имена 'net' и 'head'
    2) В Ctrl-System UNet: Conv2d(bias=True), имена 'double_conv' и 'final_conv'
    3) Их формат ожидает checkpoint['model_state_dict']

  Конвертация:
    - rename:  '.net.' -> '.double_conv.',  'head.' -> 'final_conv.'
    - add nulled biases for Conv2d positions inside DoubleConv (.0 и .3)
"""

import re
import sys
from collections import OrderedDict
from pathlib import Path

import torch

SRC = Path('D:/AI/defect_model_red_1024/best_model_red_1024_ema.pth')
DST = Path('D:/Ctrl-System/best_model.pth')

print(f"src: {SRC}")
print(f"dst: {DST}")

ckpt = torch.load(str(SRC), map_location='cpu')
sd = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
print(f"src keys: {len(sd)}")
print(f"src epoch:    {ckpt.get('epoch')}")
print(f"src val_iou:  {ckpt.get('val_iou')}")
print(f"src img_size: {ckpt.get('image_size')}")

# 1) Rename keys
new_sd = OrderedDict()
for k, v in sd.items():
    nk = k.replace('.net.', '.double_conv.')
    if nk.startswith('head.'):
        nk = 'final_conv.' + nk[len('head.'):]
    new_sd[nk] = v

# 2) Add zero biases for Conv2d in DoubleConv
add = {}
for k, v in list(new_sd.items()):
    m = re.match(r'^(.*\.double_conv\.[03])\.weight$', k)
    if m:
        bk = m.group(1) + '.bias'
        if bk not in new_sd:
            add[bk] = torch.zeros(v.shape[0], dtype=v.dtype)

new_sd.update(add)
print(f"renamed + added {len(add)} zero biases  ->  total keys: {len(new_sd)}")

# 3) Validate by loading into Ctrl-System UNet (strict)
sys.path.insert(0, 'D:/Ctrl-System')
from models.unet import UNet  # noqa

m = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
res = m.load_state_dict(new_sd, strict=True)
if res.missing_keys or res.unexpected_keys:
    print(f"[FAIL] load failed:\n  missing:    {res.missing_keys}\n  unexpected: {res.unexpected_keys}")
    sys.exit(1)
print("[OK] strict load OK in Ctrl-System UNet")

# 4) Save in expected format
out = {
    'model_state_dict': new_sd,
    'epoch':            ckpt.get('epoch'),
    'val_iou':          ckpt.get('val_iou'),
    'image_size':       ckpt.get('image_size'),
    'features':         ckpt.get('features'),
    '_source':          str(SRC.name),
    '_note':            'converted from TRAIN_1024 (bias=False net/head) to Ctrl-System format',
}
torch.save(out, str(DST))
print(f"[OK] saved -> {DST}")
