from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference'))
from conv_reference import conv
r=np.random.default_rng(42)
x=r.uniform(-1,1,(28,28)).astype(np.float32)
w=r.uniform(-1,1,(3,3)).astype(np.float32)
sx=sw=1/127
q=lambda a,s: np.clip(np.rint(a/s),-128,127).astype(np.int8)
qx,qw=q(x,sx),q(w,sw)
ref=conv(x,w)
out=ROOT/'results'; out.mkdir(exist_ok=True)
for name,a in [('input',x),('weights',w),('fp32_reference',ref),('input_int8',qx),('weights_int8',qw)]: np.save(out/f'{name}.npy',a)
with (out/'vectors.txt').open('w') as f:
    f.write(' '.join(map(str,qx.ravel()))+'\n'+' '.join(map(str,qw.ravel()))+'\n')
(out/'quantization.json').write_text(json.dumps({'seed':42,'input_scale':sx,'weight_scale':sw,'output_scale':sx*sw,'accumulator_bits':32,'rounding':'numpy.rint ties to even','clamp':[-128,127],'output':'INT32 accumulation; dequantize using output_scale'},indent=2)+'\n')
