import numpy as np, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'.')
from fe.geometry import *
from fe.solver import Grid
from matplotlib.colors import ListedColormap
CMAP = ListedColormap(['white','#c9b38a','#a88c5c','#8a9c6a','#6a8c9c','#e6550d','#fd8d3c','#fdae6b','#3182bd','#6baed6','#cb181d','#9e9ac8','#fff7bc','#636363','#31a354'])
def dominant(frac):
    return frac.argmax(1)
def show(foot, act, fname, grid=None):
    grid = grid or Grid(*make_grid_edges())
    g = FootGeom(foot, act)
    fr = voxelise(g, grid)
    lab = dominant(fr).reshape(grid.shape)
    fig, axs = plt.subplots(2,3, figsize=(15,8))
    # top views at joint band levels
    for ax, zz in zip(axs[0], [32.5, 34.5, 36.5]):
        k = np.searchsorted(grid.ze, zz)-1
        ax.pcolormesh(grid.xe, grid.ye, lab[:,:,k].T, cmap=CMAP, vmin=-0.5, vmax=14.5)
        ax.set_title(f'top view z={grid.zc[k]:.1f}'); ax.set_aspect('equal')
    for ax, yy in zip(axs[1][:2], [-2.0, 11.0]):
        j = np.searchsorted(grid.ye, yy)-1
        ax.pcolormesh(grid.xe, grid.ze, lab[:,j,:].T, cmap=CMAP, vmin=-0.5, vmax=14.5)
        ax.set_title(f'sagittal y={grid.yc[j]:.1f}'); ax.set_aspect('equal')
    i = np.searchsorted(grid.xe, 35.0)-1
    axs[1][2].pcolormesh(grid.ye, grid.ze, lab[i,:,:].T, cmap=CMAP, vmin=-0.5, vmax=14.5)
    axs[1][2].set_title(f'coronal x={grid.xc[i]:.1f}'); axs[1][2].set_aspect('equal')
    plt.tight_layout(); plt.savefig(fname, dpi=80); plt.close()
    return fr, grid
if __name__ == '__main__':
    import time
    t=time.time()
    fr, grid = show(Foot(), Action(), 'figs/geom_ref.png')
    print('vox time', time.time()-t, 'cells', grid.ncell, 'occupied', (fr[:,1:].sum(1)>0.01).sum())
    for l,n in enumerate(LABEL_NAMES): print(n, round(float(fr[:,l].sum()),1), end='; ')
