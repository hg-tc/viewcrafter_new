from viewcrafter_cam0 import ViewCrafterCam0
import os
from configs.infer_config import get_parser
from utils.pvd_utils import *
from datetime import datetime


if __name__=="__main__":
    parser = get_parser() # infer config.py
    opts = parser.parse_args()
    if opts.exp_name == None:
        prefix = datetime.now().strftime("%Y%m%d_%H%M")
        opts.exp_name = f'{prefix}_{os.path.splitext(os.path.basename(opts.image_dir))[0]}'
    opts.save_dir = os.path.join(opts.out_dir,opts.exp_name)
    os.makedirs(opts.save_dir,exist_ok=True)
    pvd = ViewCrafterCam0(opts)

    if opts.mode == 'sparse_view_interp':
        pvd.nvs_sparse_view_interp_cam0()
    elif opts.mode == 'external_pose':
        pvd.nvs_external_pose_cam0()
    elif opts.mode == 'single_view_target':
        pvd.nvs_single_view_cam0()
    elif opts.mode == 'single_view_txt':
        pvd.nvs_single_view_cam0()
    elif opts.mode == 'single_view_eval':
        pvd.nvs_single_view_eval_cam0()
    else:
        raise KeyError(f"Invalid Mode: {opts.mode}")

