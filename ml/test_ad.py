import sys, numpy as np, scipy.sparse as sp
sys.path.insert(0,'.')
import ml.ad as ad
ad.DT = np.float64
from ml.ad import *
rng=np.random.default_rng(0)
def check(f, shapes, name):
    xs=[ad.T(rng.standard_normal(s), req=True) for s in shapes]
    y=f(*xs); L=(y*ad.T(np.cos(np.arange(y.data.size).reshape(y.shape)))).sum(); L.backward()
    worst=0
    for x in xs:
        g=x.grad; num=np.zeros_like(x.data); h=1e-6
        it=np.nditer(x.data, flags=['multi_index'])
        cnt=0
        for _ in it:
            idx=it.multi_index; cnt+=1
            if cnt>40: break
            old=x.data[idx]; x.data[idx]=old+h; lp=(f(*xs)*np.cos(np.arange(y.data.size).reshape(y.shape))).sum().data
            x.data[idx]=old-h; lm=(f(*xs)*np.cos(np.arange(y.data.size).reshape(y.shape))).sum().data; x.data[idx]=old
            num[idx]=(lp-lm)/(2*h)
            worst=max(worst, abs(num[idx]-g[idx])/(abs(num[idx])+abs(g[idx])+1e-8))
    print(f'{name:20s} max rel err {worst:.2e}')
S=sp.random(7,5,density=0.5,random_state=1,format='csr')
check(lambda a,b: a@b, [(2,3,4),(4,5)], 'matmul bcast')
check(lambda a,b: a@b, [(2,3,4),(2,4,5)], 'bmm')
check(lambda a,b,c: layernorm(a,b,c), [(3,6),(6,),(6,)], 'layernorm')
check(lambda a: spmm(S,a), [(5,3)], 'spmm')
check(lambda a: take(a,np.array([0,2,2,1]),0), [(4,3)], 'take')
check(lambda a,b: concat([a,b],1), [(2,3),(2,2)], 'concat')
check(lambda a: gelu(a), [(3,4)], 'gelu')
check(lambda a: softmax(a,1), [(3,4)], 'softmax')
check(lambda a,b: (a*b).sum(1)+a.mean(1), [(3,4),(1,4)], 'mul/sum/mean')
check(lambda a: sqrt(a*a+1.0), [(3,4)], 'sqrt')
check(lambda a: relu(a-0.1), [(3,4)], 'relu')
check(lambda a: a.reshape(4,3).transpose(1,0), [(3,4)], 'reshape/T')
check(lambda a: exp(a)*sigmoid(a)+tanh(a), [(3,4)], 'exp/sig/tanh')
modes=np.array([0,1,5,7,20,33,47])
check(lambda x,wr,wi: spectral_conv3d(x,wr,wi,modes), [(2,3,4,3,4),(3,2,7),(3,2,7)], 'spectral_conv3d')
