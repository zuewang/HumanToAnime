"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils.utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

# debug reference: https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/human2anime.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

if __name__ == '__main__':
    # Setup model and data loader
    trainer = MUNIT_Trainer(config)
    trainer.cuda()
    # a: anime; b:human
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i]['image'] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i]['image'] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i]['image'] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i]['image'] for i in range(display_size)]).cuda()

    # train_display_labels_b = torch.stack([train_loader_b.dataset[i]['label'] for i in range(display_size)]).cuda()
    # test_display_labels_b = torch.stack([test_loader_b.dataset[i]['label'] for i in range(display_size)]).cuda()
    print('test_display_images_b', test_display_images_b.size())
    # print('test_display_labels_b', test_display_labels_b.size())
    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    print('='*10, 'start training', '='*10)
    # Start training
    # sample with input 256x256: c_a: torch.Size([batch, 256, 64, 64]), s_a_fake: torch.Size([batch, 8, 1, 1])
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    while True:
        for it, (samples_a, samples_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            # detach
            images_a, labels_a = samples_a['image'].cuda().detach(), samples_a['label'].cuda()
            images_b, labels_b = samples_b['image'].cuda().detach(), samples_b['label'].cuda()
            # print(images_a.shape, labels_a, images_b.shape, labels_b.shape)
            # print(labels_b)
            # with torch.no_grad():
            #     trainer(train_display_images_a, train_display_images_b)
            # exit(0)

            with Timer("Elapsed time in update: %f"):
                # Main training code
                trainer.dis_update(images_a, images_b, config)
                trainer.gen_update(images_a, images_b, config, labels_b)
                torch.cuda.synchronize()
            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    # input: (x_a, x_b), output: x_a, x_a_recon, x_ab1, x_ab2 (from noise),  x_b, x_b_recon, x_ba1, x_ba2
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b) #, test_display_labels_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b) #, train_display_labels_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 0) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b) #, train_display_labels_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')
                # print('========================= face_alignment_net weight ===============================')
                # torch.save(trainer.face_alignment_net.state_dict(), './facemark_%08d.pt' % (iterations + 0))
                
            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

