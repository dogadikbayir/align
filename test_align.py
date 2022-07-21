from align import plotly_scatter
from align import read_obj

import numpy as np

import glob

fnames_root = glob.glob('./shapes_first_5/*')
fnames_al = glob.glob('./aligned/*')

fnames_root.sort()
fnames_al.sort()

pcs_root = []
pcs_aligned = []

for fr, fa in zip(fnames_root, fnames_al):

    pcs_r, faces_r = read_obj(fr)
    pcs_al, faces_al = read_obj(fa)

    pcs_root.append(pcs_r)
    pcs_aligned.append(pcs_al)

plotly_scatter(pcs_root)

plotly_scatter(pcs_aligned)

print(list(zip(fnames_root, fnames_al)))
