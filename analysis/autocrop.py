import numpy as np
from PIL import Image
def autocrop(path, out, left=0.24, bottom=0.86, triad=(0.84,0.70), top=0.0, pad=8, thr=242, extra_masks=()):
    im=Image.open(path).convert("RGB"); a=np.asarray(im).astype(int); H,W,_=a.shape
    fg=(a.min(axis=2)<thr)
    m=np.zeros_like(fg)
    m[int(top*H):int(bottom*H), int(left*W):W]=True
    m[int(triad[1]*H):, int(triad[0]*W):]=False
    for (x0,y0,x1,y1) in extra_masks:
        m[int(y0*H):int(y1*H), int(x0*W):int(x1*W)]=False
    fg=fg&m
    ys,xs=np.where(fg)
    if len(xs)==0: raise Exception("empty "+path)
    x0,x1,y0,y1=max(xs.min()-pad,0),min(xs.max()+pad,W),max(ys.min()-pad,0),min(ys.max()+pad,H)
    c=im.crop((x0,y0,x1,y1))
    # whiten anything outside mask inside crop (legend remnants) 
    c.save(out); return c.size
