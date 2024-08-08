import os
import sys
import glob
from tqdm import tqdm
import torch
import torchvision
from pytorch_fid import fid_score
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips

# ps: pixel value should be in [0, 1]!

def calculate_fid(hr_path, sr_path):
    return fid_score.calculate_fid_given_paths(
        [hr_path, sr_path], batch_size=1, device="cuda", dims=2048, num_workers=4
    )

def read_dir(dir_path, exts=[".png", ".jpg", ".jpeg"], sort=True):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dir_path, "*" + ext)))
    if sort:
        files = sorted(files)
    return files

if __name__ == "__main__":
    ori_dir = sys.argv[1]
    gen_dir = sys.argv[2]
    device = torch.device("cuda")

    ori_files = read_dir(ori_dir)
    gen_files = read_dir(gen_dir)

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0

    assert len(ori_files) == len(gen_files)
    idx = 1

    ori_img_list = []
    gen_img_list = []
    for ori_file, gen_file in tqdm(zip(ori_files, gen_files)):
        ori_img = torchvision.io.read_image(ori_file) / 255.
        gen_img = torchvision.io.read_image(gen_file) / 255.
        ori_img_list.append(ori_img)
        gen_img_list.append(gen_img)

    ori_img = torch.stack(ori_img_list).unsqueeze(1)
    gen_img = torch.stack(gen_img_list).unsqueeze(1)

    avg_psnr = calculate_psnr(ori_img, gen_img)
    print("PSNR: ", avg_psnr)
    avg_ssim = calculate_ssim(ori_img, gen_img)
    print("SSIM: ", avg_ssim)
    avg_lpips = calculate_lpips(ori_img, gen_img, device)
    print("LPIPS: ", avg_lpips)
    avg_fid = calculate_fid(ori_dir, gen_dir).item()
    print("FID: ", avg_fid)
