import os
import argparse
import math
import time

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def test(args):
    device = args.device
    print(f"devide: {device}")
    
    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {args.checkpoint}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue
            
        image = load_image(os.path.join(args.input_dir, image_name), args.x32)

        print(f"image size: w: {image.width}, h: {image.height}")

        with torch.no_grad():
            img = to_tensor(image).unsqueeze(0) * 2 - 1
            out1 = net(img.to(device), args.upsample_align).cpu()
            out2 = out1.squeeze(0).clip(-1, 1) * 0.5 + 0.5
            out = to_pil_image(out2)

        out.save(os.path.join(args.output_dir, image_name))
        print(f"image saved: {image_name}")
    # print(net)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./pytorch_model/paprika.pt',
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    args = parser.parse_args()

    start = time.time()
    math.factorial(100000)
    test(args)
    end = time.time()
    print(f"{end - start:.5f} sec")
