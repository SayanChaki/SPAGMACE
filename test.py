import argparse
import os
import numpy as np
import cv2
import datetime
import random
from ipdb import set_trace
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import torch
from torch import autograd
from torch import nn, optim
from tensorboardX import SummaryWriter

from gmair.models.model import gmair
from gmair.config import config as cfg
from gmair.dataset.fruit2d import FruitDataset
#from gmair.dataset.multi_mnist import SimpleScatteredMNISTDataset
from gmair.utils import debug_tools
#from gmair.test import metric, cluster_metric

# args = parser.parse_args()

random.seed(3)
torch.manual_seed(3)
np.random.seed(3)

log_dir = os.path.join(
        cfg.log_dir,
        datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
log_dir = os.path.abspath(log_dir)
writer = SummaryWriter(log_dir)
print('log path : {}'.format(log_dir))

device = 'cuda'

def train():
    load_model_path = '/home/sayanchaki/Downloads/GMAIR-pytorch-main2/logs/von_mises_smaller/2024_11_06__00_48_13/checkpoints/step_23000.pkl'
    #load_model_path = None
    start_iter = 0
    image_shape = cfg.input_image_shape
    
    if cfg.dataset == 'multi_mnist':
        data = SimpleScatteredMNISTDataset('/home/sayanchaki/Downloads/GMAIR-pytorch-main2/data/multi_mnist/scattered_mnist_128x128_obj14x14.hdf5')
    elif cfg.dataset == 'fruit2d':
        data = FruitDataset('/home/sayanchaki/Downloads/GMAIR-pytorch-main2/generate_test.py/output_smaller/images','/home/sayanchaki/Downloads/GMAIR-pytorch-main2/generate_test.py/output_smaller/bb')
    else:
        print('No implementation for {}'.format(cfg.dataset))
        exit(0)

    print(device)

    gmair_net = gmair(image_shape, writer, device).to(device)
    print('ABOUT TO LOAD MODEL')    
    if load_model_path is not None:
        print('MODEL LOADED')
        gmair_net.load_state_dict(torch.load(load_model_path), strict=False)
        print('Model loaded')

    encoder_params = list(map(id, gmair_net.object_encoder_what.parameters()))
    decoder_params = list(map(id, gmair_net.object_decoder.parameters()))
    encoder_cat_params = list(map(id, gmair_net.object_encoder_cat.parameters()))

    pre_params = encoder_params + decoder_params + encoder_cat_params
    other_params = filter(lambda p: id(p) not in pre_params, gmair_net.parameters())
    
    params = [
      {"params": gmair_net.object_encoder_what.parameters(), "lr": cfg.encoder_what_lr},
      {"params": gmair_net.object_decoder.parameters(), "lr": cfg.decoder_lr},
      {"params": gmair_net.object_encoder_cat.parameters(), "lr": cfg.encoder_cat_lr},
      {"params": other_params, "lr": cfg.lr},
    ]
    
    gmair_optim = optim.Adam(params, lr=cfg.lr)

    start_cluster_measure = False
    
    dataloader = torch.utils.data.DataLoader(data,
                                           batch_size = cfg.batch_size,
                                           pin_memory = True,
                                           num_workers = cfg.num_workers,
                                           drop_last = True,
                                           shuffle = True
                                           )
    for batch_idx, batch in enumerate(dataloader):
            iteration = 0* len(dataloader) + batch_idx + start_iter
            
            x_image, y_bbox, y_obj_count = batch
            
            x_image = x_image.to(device)
            y_bbox = y_bbox.to(device)
            y_obj_count = y_obj_count.to(device)

            print('Iteration', iteration)

            gmair_net.train()
            gmair_optim.zero_grad()
            print("NEW IMAGE SHAPE")
            print(x_image.shape)
            loss, out_img = gmair_net(x_image, iteration)
            loss.backward()
            gmair_optim.step()

            # Display input and output images
            image_out = out_img[0].cpu().detach()
            image_in = x_image[0].cpu().detach()

            # Print shapes for debugging
            print(f"Input image shape: {image_in.shape}")
            print(f"Output image shape: {image_out.shape}")

            # Handle grayscale images (1 channel) and color images (3 channels)
            if image_out.shape[0] == 1:  # Grayscale (C=1)
                print("ENTERED TO DISPLAy")
                #image_in = image_in.squeeze(0)  # Remove channel dimension for grayscale images
                image_out = image_out.squeeze(0)

                # Display the grayscale images using matplotlib
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title('Input Image (Grayscale)')
                plt.imshow(image_in, cmap='gray')

                plt.subplot(1, 2, 2)
                plt.title('Output Image (Grayscale)')
                plt.imshow(image_out, cmap='gray')
                plt.show()

            elif image_in.shape[0] == 3:  # Color images (C=3)
                print("HELLOW")
                # Transpose from (C, H, W) to (H, W, C) for displaying with matplotlib
                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.title('Input Image (Color)')
                plt.imshow(np.transpose(image_in.numpy(), (1, 2, 0)))  # From (C, H, W) to (H, W, C)

                plt.subplot(1, 2, 2)
                plt.title('Output Image (Color)')
                plt.imshow(np.transpose(image_out.numpy(), (1, 2, 0)))  # From (C, H, W) to (H, W, C)

                plt.show()

            # Option 2: TensorBoard logging
            writer.add_image('Input Image', vutils.make_grid(image_in, normalize=True), iteration)
            writer.add_image('Output Image', vutils.make_grid(image_out, normalize=True), iteration)

            if iteration % 1 == 0:
                with torch.no_grad():
                    gmair_net.eval()
                    out_img, z_cls, z_what, z_where, obj_prob = gmair_net(x_image, mode='infer')
                    
                    print("Z6WHERE SHAPE BEFORE DEBUG TOOLS IS: "+str(z_where.shape))
                debug_tools.plot_infer_render_components(x_image, y_bbox, obj_prob, z_cls, z_where, out_img, writer, iteration)

            # Save model
            '''if iteration > 0 and iteration % 10 == 0:
                check_point_name = 'step_%d.pkl' % iteration
                cp_dir = os.path.join(log_dir, 'checkpoints')
                os.makedirs(cp_dir, exist_ok=True)
                save_path = os.path.join(log_dir, 'checkpoints', check_point_name)
                torch.save(gmair_net.state_dict(), save_path)'''

            print('=================\n\n')

if __name__ == '__main__':
    train()
