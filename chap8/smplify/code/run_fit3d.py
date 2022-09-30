from os.path import join, exists, abspath, dirname
from os import makedirs
import cPickle as pickle
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from smpl.serialization import load_model
from smplify.code.fit3d_utils import run_single_fit

MODEL_DIR = join(abspath(dirname(__file__)), 'models')
MODEL_NEUTRAL_PATH = join(
    MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')


def main(base_dir, out_dir):
    viz = True
    n_betas = 10
    flength = 5000.0
    pix_thsh = 25.0
    img_dir = join(abspath(base_dir), 'images/lsp')
    data_dir = join(abspath(base_dir), 'results/lsp')

    if not exists(out_dir):
        makedirs(out_dir)

    # Camera azimuth angles for visulization
    do_degrees = [0.]
    model = load_model(MODEL_NEUTRAL_PATH)
    est = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    img_paths = sorted(glob(join(img_dir, '*[0-9].jpg')))
    for ind, img_path in enumerate(img_paths):
        out_path = '%s/%04d.pkl' % (out_dir, ind)
        img = cv2.imread(img_path)

        joints = est[:2, :, ind].T
        conf = est[2, :, ind]

        params, vis = run_single_fit(img, joints, conf, model, regs=None,
                                     n_betas=n_betas, flength=flength,
                                     pix_thsh=pix_thsh, scale_factor=2,
                                     viz=viz, do_degrees=do_degrees)
        if viz:
            plt.ion()
            plt.show()
            plt.subplot(121)
            plt.imshow(img[:, :, ::-1])
            if do_degrees is not None:
                for di, deg in enumerate(do_degrees):
                    plt.subplot(122)
                    plt.cla()
                    plt.imshow(vis[di])
                    plt.draw()
                    plt.title('%d deg' % deg)
                    plt.pause(1)
            raw_input('Press any key to continue...')

        with open(out_path, 'w') as outf:
            pickle.dump(params, outf)

        # This only saves the first rendering.
        if do_degrees is not None:
            cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run SMPLify on LSP dataset')
    parser.add_argument(
        '--base_dir',
        default='/scratch1/projects/smplify_public/',
        nargs='?',
        help="Directory that contains images/lsp and results/lps , i.e."
        "the directory you untared smplify_code.tar.gz")
    parser.add_argument(
        '--out_dir',
        default='/tmp/smplify_lsp/',
        type=str,
        help='Where results will be saved, default is /tmp/smplify_lsp')
    args = parser.parse_args()

    main(args.base_dir, args.out_dir)
