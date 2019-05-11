from utils.utils import get_config #get_all_data_loaders, prepare_sub_folder, write_html, write_loss, write_2images, Timer
from trainer import MUNIT_Trainer
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
import os
import numpy as np
# from array2gif import write_gif
from moviepy.editor import ImageSequenceClip # need specific version: e.g. conda install -c conda-forge imageio=2.4.1

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/human2anime.yaml', help='Path to the config file.')
parser.add_argument('--image_path', type=str, help="path to image to be transformed")
parser.add_argument('--checkpoint_dir', type=str, help="directory containing checkpoints")
parser.add_argument("--a2b", action="store_true", help='transform type-a image to type-b, omitting this flag will transform type-b image to type-a')
parser.add_argument('--fps', type=int, default=10, help="fps of generated GIF")
opts = parser.parse_args()


# Load experiment setting
config = get_config(opts.config)

if __name__ == '__main__':
    # number of images needed to generate GIF 
    TOTAL_ROUNDS = 10 # generate new random style vector
    TOTAL_STEPS = opts.fps*5 # number of steps for each round
    NUM_IMG_GIF = TOTAL_STEPS # number of steps taken into gif

    trainer = MUNIT_Trainer(config)
    trainer.cuda()
    trainer.resume(opts.checkpoint_dir, hyperparameters=config)

    img = Image.open(opts.image_path)
    print('input image shape:', img.size)
    transform_list = [transforms.Resize(config['new_size']),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    transform_list = transforms.Compose(transform_list)
    
    img_tensor = transform_list(img).cuda().unsqueeze(0)
    print('image tensor shape:', img_tensor.size())
    
    trainer.eval()    
    output_img = []

    gen1 = trainer.gen_a if opts.a2b else trainer.gen_b
    gen2 = trainer.gen_b if opts.a2b else trainer.gen_a

    content, style_fake = gen1.encode(img_tensor)

    style_old = Variable(torch.randn(img_tensor.size(0), config['gen']['style_dim'], 1, 1).cuda()).unsqueeze(0)
    for r in range(TOTAL_ROUNDS):
        style_start = style_old
        style_end = Variable(torch.randn(img_tensor.size(0), config['gen']['style_dim'], 1, 1).cuda()).unsqueeze(0)
        style_old = style_end
        for i in range(NUM_IMG_GIF):
            # use detach() to save VRAM
            style = style_start + float(i) / float(TOTAL_STEPS) * (style_end - style_start)
            style.detach()
            out = gen2.decode(content, style).detach().squeeze()
            out = (out + 1.0) / 2.0 * 255.0
            # C, H, W => H, W, C
            out = out.cpu().numpy().astype(int).transpose((1, 2, 0))
            output_img.append(out)

    print(output_img)
    print('output images shape: {} * {}, max: {}, min: {}'.format(len(output_img), output_img[0].shape, np.max(output_img[0]), np.min(output_img[0])))

    # generate GIF
    path_noext = '.'.join(opts.image_path.split('.')[:-1])
    out_gif_path = path_noext + '.gif'
    
    clip = ImageSequenceClip(output_img, fps=opts.fps)
    clip.write_gif(out_gif_path, fps=opts.fps)

