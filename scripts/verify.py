from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference'))
from conv_reference import conv
p=ROOT/'results'; cfg=json.loads((p/'quantization.json').read_text())
x=np.load(p/'input.npy');w=np.load(p/'weights.npy');ref=np.load(p/'fp32_reference.npy').ravel()
qx=np.load(p/'input_int8.npy').astype(np.int64);qw=np.load(p/'weights_int8.npy').astype(np.int64)
expected=np.array([np.sum(qx[i:i+3,j:j+3]*qw,dtype=np.int64) for i in range(26) for j in range(26)])
actual=np.loadtxt(sys.argv[1],dtype=np.int64).reshape(-1)
if actual.size!=676:raise ValueError('expected 676 outputs')
err=actual*cfg['output_scale']-ref
res={'shape':[26,26],'mae':float(np.mean(np.abs(err))),'rmse':float(np.sqrt(np.mean(err**2))),'max_error':float(np.max(np.abs(err))),'exact_mismatch_count':int(np.count_nonzero(actual!=expected)),'saturation_count':sum(int(np.count_nonzero((np.rint(a / scale) < -128) | (np.rint(a / scale) > 127))) for a, scale in [(x, cfg['input_scale']), (w, cfg['weight_scale'])]),'accumulator_overflow_count':int(np.count_nonzero((expected<-(2**31))|(expected>2**31-1))),'fp32_reference_recheck_max_error':float(np.max(np.abs(conv(x,w).ravel()-ref))),'provenance':'native C++ testbench, not HLS co-simulation'}
(p/'numerical_results.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))
if res['exact_mismatch_count']:sys.exit(1)
