#!/usr/bin/env python3
import argparse, os, time, statistics, csv
import numpy as np
from typing import List, Tuple, Dict
from PIL import Image

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="agdn_nano.onnx")
    p.add_argument("--mxq",  default="agdn_nano.mxq")
    p.add_argument("--images", default="LR_val")
    p.add_argument("--hr-dir", default="HR_val")
    p.add_argument("--mapping", default=None)  # CSV with columns: lr,hr
    p.add_argument("--psnr-mode", choices=["rgb","y"], default="rgb")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)  # 0 = all files
    p.add_argument("--no-cpu", dest="no_cpu", action="store_true")
    p.add_argument("--save-outputs", default="sr_outputs")
    p.add_argument("--per-image-psnr", action="store_true")
    return p.parse_args()

def load_lrs(dir_path, size, limit) -> Tuple[List[np.ndarray], List[str]]:
    exts=(".png",".jpg",".jpeg",".bmp")
    files=[f for f in sorted(os.listdir(dir_path)) if f.lower().endswith(exts)]
    if limit and limit>0: files=files[:limit]
    imgs=[]; names=[]
    for f in files:
        im=Image.open(os.path.join(dir_path,f)).convert("RGB").resize((size,size), Image.BILINEAR)
        imgs.append(np.asarray(im,dtype=np.float32)/255.0)
        names.append(f)
    if not imgs: raise RuntimeError("no input images found")
    return imgs, names

def cpu_infer(onnx_path, hwc_imgs):
    import onnxruntime as ort
    s=ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name=s.get_inputs()[0].name
    def nhwc(x): return x[None,...].astype(np.float32)
    def nchw(x): return x.transpose(2,0,1)[None,...].astype(np.float32)
    def run(conv):
        t=[]; outs=[]
        for a in hwc_imgs:
            x=conv(a); t0=time.perf_counter(); y=s.run(None,{name:x}); t.append((time.perf_counter()-t0)*1000.0)
            o=np.array(y[0])
            if o.ndim==4:
                o=o[0] if o.shape[-1] in (1,3) else o[0].transpose(1,2,0)
            elif o.ndim==3 and o.shape[0] in (1,3):
                o=o.transpose(1,2,0)
            outs.append(np.clip(o.astype(np.float32),0,1))
        return statistics.mean(t), outs
    try: return run(nhwc)
    except Exception: return run(nchw)

def npu_infer(mxq_path, hwc_imgs):
    import maccel
    acc=maccel.Accelerator(0); m=maccel.Model(mxq_path); m.launch(acc)
    def run_hwc():
        t=[]; outs=[]
        for a in hwc_imgs:
            t0=time.perf_counter(); y=m.infer([a.astype(np.float32)]); t.append((time.perf_counter()-t0)*1000.0)
            o=np.array(y[0])
            if o.ndim==4:
                o=o[0] if o.shape[-1] in (1,3) else o[0].transpose(1,2,0)
            elif o.ndim==3 and o.shape[0] in (1,3):
                o=o.transpose(1,2,0)
            outs.append(np.clip(o.astype(np.float32),0,1))
        return statistics.mean(t), outs
    def run_chw():
        t=[]; outs=[]
        for a in hwc_imgs:
            x=a.transpose(2,0,1).astype(np.float32); t0=time.perf_counter(); y=m.infer_chw([x]); t.append((time.perf_counter()-t0)*1000.0)
            o=np.array(y[0])
            if o.ndim==4:
                o=o[0] if o.shape[-1] in (1,3) else o[0].transpose(1,2,0)
            elif o.ndim==3 and o.shape[0] in (1,3):
                o=o.transpose(1,2,0)
            outs.append(np.clip(o.astype(np.float32),0,1))
        return statistics.mean(t), outs
    try: mean, outs=run_hwc()
    except Exception: mean, outs=run_chw()
    m.dispose(); return mean, outs

def save_outputs(folder, names, imgs):
    os.makedirs(folder, exist_ok=True)
    for n, im in zip(names, imgs):
        x=(np.clip(im,0,1)*255.0).round().astype(np.uint8)
        Image.fromarray(x).save(os.path.join(folder, n))

def to_y(img: np.ndarray) -> np.ndarray:
    r,g,b=img[...,0],img[...,1],img[...,2]
    return 0.299*r + 0.587*g + 0.114*b

def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    mse=float(np.mean((a-b)**2))
    if mse<=1e-12: return 99.0
    return 20.0*np.log10(1.0/np.sqrt(mse))

def psnr_y(a: np.ndarray, b: np.ndarray) -> float:
    ya,yb=to_y(a),to_y(b)
    mse=float(np.mean((ya-yb)**2))
    if mse<=1e-12: return 99.0
    return 20.0*np.log10(1.0/np.sqrt(mse))

def load_mapping(csv_path: str) -> Dict[str,str]:
    m={}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            lr=row.get('lr') or row.get('LR') or row.get('lr_name')
            hr=row.get('hr') or row.get('HR') or row.get('hr_name')
            if lr and hr: m[lr.strip()]=hr.strip()
    return m

def main():
    a=parse()
    imgs,names=load_lrs(a.images,a.size,a.limit)

    if (not a.no_cpu) and os.path.isfile(a.onnx):
        try:
            cpu_mean,_=cpu_infer(a.onnx,imgs)
            print(f"CPU {cpu_mean:.2f} ms/img ({1000.0/cpu_mean:.1f} img/s)")
        except Exception as e:
            print(f"CPU failed: {e}")

    if not os.path.isfile(a.mxq): raise FileNotFoundError(a.mxq)
    npu_mean, outs=npu_infer(a.mxq, imgs)
    print(f"NPU {npu_mean:.2f} ms/img ({1000.0/npu_mean:.1f} img/s)")
    if a.save_outputs: save_outputs(a.save_outputs, names, outs)

    if a.hr_dir:
        mapping=None
        if a.mapping and os.path.isfile(a.mapping):
            mapping=load_mapping(a.mapping)
        vals=[]; miss=0
        for n, y in zip(names, outs):
            hr_name=mapping.get(n, n) if mapping else n
            hp=os.path.join(a.hr_dir, hr_name)
            if not os.path.isfile(hp):
                miss+=1; continue
            hr=Image.open(hp).convert("RGB").resize(y.shape[1::-1], Image.BILINEAR)
            hr=np.asarray(hr, dtype=np.float32)/255.0
            s=psnr_y(y,hr) if a.psnr_mode=="y" else psnr_rgb(y,hr)
            vals.append(s)
            if a.per_image_psnr: print(f"{n} {s:.2f} dB")
        if vals: print(f"PSNR({a.psnr_mode}) mean: {float(np.mean(vals)):.2f} dB over {len(vals)} images")
        if miss>0: print(f"Missing HR files: {miss}")

if __name__=="__main__":
    main()
