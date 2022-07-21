import numpy as np
import glob
import tqdm
import pqdm

from pqdm.processes import pqdm

import argparse

from pathlib import Path

from align import *


parser = argparse.ArgumentParser()

parser.add_argument(
        '--dataset_root', type=str, default='./dataset/'
)
parser.add_argument(
        '--ax1', type=int, default=0
)
parser.add_argument(
        '--ax2', type=int, default=1
)
parser.add_argument(
        '--out_root', type=str, default='./aligned'
)

opt = parser.parse_args()

# Get file names in opt.dataset_root
fnames = glob.glob(opt.dataset_root + '/*')

# Create dir for results
Path(opt.out_root).mkdir(parents=True, exist_ok=True)

# Align each object and save it to opt.out_root
for fname in tqdm.tqdm(fnames):
    
    fn = fname.split('/')[-1]
    #print(fn)
    
    # Load object
    pts, faces = read_obj(fname)
    
    # Align
    pts = align(pts)

    # Save object
    save_obj(opt.out_root + '/' + fn, pts, faces)


