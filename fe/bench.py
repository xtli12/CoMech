import sys, time, numpy as np
sys.path.insert(0,'.')
from fe.simulate import *
for m in ['cg']:
    t=time.time()
    r = simulate(Foot(), Action(), method=m, tol=1e-7)
    print(m, 'total', round(time.time()-t,2), 'timing', [round(x,2) for x in r['timing']], 'iters', r['iters'], 'ndof', r['ndof'])
    for k,v in r['eff'].items(): print(f'  {k}: {v:.4f}')
