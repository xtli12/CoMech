import numpy as np, time, sys
sys.path.insert(0,'.')
from fe.solver import *
g = Grid(np.r_[0,1,3,4,7,10], np.r_[0,2,5,10], np.r_[0,1,2,4,6,10])
E=1000.; nu=0.3
lam,mu = lame(np.full(g.ncell,E), np.full(g.ncell,nu))
active = np.ones(g.ncell,bool)
bot = np.nonzero(g.cK==0)[0]; top=np.nonzero(g.cK==g.nz-1)[0]
nb,ab = face_node_areas(g,bot,'z-'); nt,at = face_node_areas(g,top,'z+')
U,info = solve(g,lam,mu,active,[(nb,ab,np.array([1e-6,1e-6,1e9]))],[(nt,at,np.array([0,0,-1.0]))],method='direct')
sig,vm,eps = cell_stress(g,lam,mu,U,np.arange(g.ncell))
print('uz top', np.nanmin(U[:,2]), np.nanmax(U[:,2]), 'expected', -10/E)
print('szz', sig[:,2].min(), sig[:,2].max(), 'vm', vm.min(), vm.max())
print('ux range', np.nanmin(U[:,0]), np.nanmax(U[:,0]), 'expected lateral', nu*10/E)
