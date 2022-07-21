import numpy as np
import glob
import tqdm
import pqdm

from pqdm.processes import pqdm

import argparse

from pathlib import Path

from align import *

def process(inp):

    fn = inp[0]
    dataset_root = inp[1]
    out_root = inp[2]

    ps, faces = read_obj(dataset_root + '/' + fn)

    ps = align(ps)

    save_obj(out_root + '/' + fn, ps, faces)


if __name__ == "__main__":

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
    parser.add_argument(
            '--num_proc', type=int, default=40
    )
    opt = parser.parse_args()

    # Get file names in opt.dataset_root
    fnames = glob.glob(opt.dataset_root + '/*')

    # Create dir for results
    Path(opt.out_root).mkdir(parents=True, exist_ok=True)

    # Align each object and save it to opt.out_root
    inps = [[fn.split('/')[-1], opt.dataset_root, opt.out_root] for fn in fnames]
    #print(inps)
    #exit(0)
    #    Process dataset in parallel
    res = pqdm(inps, process, n_jobs=opt.num_proc)
