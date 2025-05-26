import os
import logging
# import string
# import random
import torch
import yaml
import wandb
import cv2 as cv
import torch.nn as nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

from data.dataloader import ImageDataset
from network.net import SpectFormer
from deeplens.geolens import GeoLens
from deeplens.srf import SRF_BGR_31_CHANNEL_400_700NM, h2rgb
from deeplens.utils import (
    batch_PSNR,
    batch_SSIM,
    denormalize_ImageNet,
    normalize_ImageNet,
)

def config():
    with open("configs/RGB2hyperspectral.yml") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    task_prefix = "RGB2hyperspectral"
    # characters = string.ascii_letters + string.digits
    # random_string = "".join(random.choice(characters) for i in range(4))
    # exp_name = current_time + "-End2End-5-lines-" + random_string
    # result_dir = f"./results/{exp_name}"
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_prefix = f"{task_prefix}-{current_time}"
    result_dir = f"./results/{exp_prefix}"

    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    return args

def train(lens, net, args):
    result_dir = args["result_dir"]

    # ==> Dataset
    train_set = ImageDataset(args["train"]["train_dir"], lens.sensor_res)
    train_loader = DataLoader(train_set, batch_size=args["train"]["bs"])

    # ==> Network optimizer
    batchs = len(train_loader)
    epochs = args["train"]["epochs"]
    net_optim = torch.optim.AdamW(
        net.parameters(), lr=args["network"]["lr"], betas=(0.9, 0.98), eps=1e-08
    )
    net_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
        net_optim, T_max=epochs * batchs, eta_min=0, last_epoch=-1
    )

    # ==> Lens optimizer
    # ========================================
    # Line 2: get lens optimizers
    # ========================================
    lens_lrs = [float(i) for i in args["lens"]["lr"]]
    lens_optim = lens.get_optimizer(lr=lens_lrs)
    lens_sche = torch.optim.lr_scheduler.CosineAnnealingLR(
        lens_optim, T_max=epochs * batchs, eta_min=0, last_epoch=-1
    )

    # ==> Criterion
    cri_l1 = nn.L1Loss()

    # ==> Log
    logging.info("Start End2End optical design.")
    lens.write_lens_json(f"{result_dir}/epoch0.json")
    lens.analysis(f"{result_dir}/epoch0", render=False, zmx_format=True)

    # srf
    SRF = np.array(SRF_BGR_31_CHANNEL_400_700NM, dtype=np.float32)
    SRF= SRF / np.sum(SRF, axis=1, keepdims=True)

    # ==> Training
    for epoch in range(args["train"]["epochs"] + 1):
        # ==> Train 1 epoch
        for img_org in tqdm(train_loader):

            # => Render image
            # ========================================
            # Line 3: plug-and-play diff-rendering
            # ========================================
            img_render = lens.render(img_org)

            # Seneor Spectral response function
            img = h2rgb(img_render, SRF)

            # => Image restoration
            img_rec = net(img)

            # => Loss
            L_rec = cri_l1(img_rec, img_org)

            # => Back-propagation
            net_optim.zero_grad()
            # ========================================
            # Line 4: zero-grad
            # ========================================
            lens_optim.zero_grad()

            L_rec.backward()

            net_optim.step()
            # ========================================
            # Line 5: step
            # ========================================
            lens_optim.step()

            if not args["DEBUG"]:
                wandb.log({"loss_class": L_rec.detach().item()})

        net_sche.step()
        lens_sche.step()

        logging.info(f"Epoch{epoch + 1} finishs.")

        # ==> Evaluate
        if epoch % 3 == 0:
            net.eval()
            with torch.no_grad():
                # => Save data and simple evaluation
                lens.write_lens_json(f"{result_dir}/epoch{epoch + 1}.json")
                lens.analysis(
                    f"{result_dir}/epoch{epoch + 1}", render=False, zmx_format=True
                )

                torch.save(net.state_dict(), f"{result_dir}/net_epoch{epoch + 1}.pth")

                # => Qualitative evaluation
                img1 = cv.cvtColor(cv.imread("./datasets/bird.png"), cv.COLOR_BGR2RGB)
                img1 = cv.resize(img1, args["train"]["img_res"]).astype(np.float32)
                img1 = (
                    torch.from_numpy(img1 / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                img1 = normalize_ImageNet(img1)

                img1_render = lens.render(img1)
                psnr_render = batch_PSNR(img1, img1_render)
                ssim_render = batch_SSIM(img1, img1_render)
                save_image(
                    denormalize_ImageNet(img1_render),
                    f"{result_dir}/img1_render_epoch{epoch + 1}.png",
                )
                img1_rec = net(img1_render)
                psnr_rec = batch_PSNR(img1, img1_rec)
                ssim_rec = batch_SSIM(img1, img1_rec)
                save_image(
                    denormalize_ImageNet(img1_rec),
                    f"{result_dir}/img1_rec_epoch{epoch + 1}.png",
                )

                logging.info(
                    f'Epoch [{epoch + 1}/{args["train"]["epochs"]}], PSNR_render: {psnr_render:.4f}, SSIM_render: {ssim_render:.4f}, PSNR_rec: {psnr_rec:.4f}, SSIM_rec: {ssim_rec:.4f}'
                )

                # => Quantitative evaluation
                # validate(net, lens, epoch, args, val_loader)

            net.train()



if __name__ == "__main__":
    device_count = torch.cuda.device_count()
    if device_count > 0:
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print("\n \n GPU(s) to be used: \n")
        print(f"Device {current_device}: {device_name}")
        device = torch.device(f"cuda:{current_device}")
    else:
        print("\n \n No GPU found, using CPU instead. \n")
        device = torch.device("cpu")

    with torch.device(device):
        # args
        args = config()
        # lens
        lense = GeoLens(filename=args["lens"]["path"])
        lense.change_sensor_res(args["train"]["img_res"])
        # network
        network = SpectFormer(img_size=256, patch_size=16, in_chans=3, num_classes=31, embed_dim=768, depth=12, mlp_ratio=4, drop_rate=0.1, drop_path_rate=0.1)

        if args["network"]["pretrained"]:
            state_dict = torch.load(args["network"]["pretrained"], map_location=device)
            network.load_state_dict(state_dict)
        # train
        train(lense, network, args)

