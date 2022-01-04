import os
import argparse
import math
import time

from PIL import Image
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        # if pad_mode not in pad_layer:
        #     raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        # if expansion_ratio != 1:
        #     layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out

class Generator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        # scripted_IRB1 = torch.jit.script(InvertedResBlock(128, 256, 2))
        # scripted_IRB2 = torch.jit.script(InvertedResBlock(256, 256, 2))
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            # scripted_IRB1,
            # scripted_IRB2,
            # scripted_IRB2,
            # scripted_IRB2,
            ConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    # def forward(self, input, align_corners=True):
    @torch.jit.script_method
    def forward(self, input):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        # if align_corners:
        #     out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        # else:
        #     out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        # if align_corners:
        #     out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        # else:
        #     out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out

def animae_ganv2_loaded(args):
    device = args.device

    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device)
    print(f"model loaded: {args.checkpoint} with {device}")
    return net

class MyScriptModule(torch.jit.ScriptModule):
    def __init__(self, args):
        super(MyScriptModule, self).__init__()

        net = animae_ganv2_loaded(args)
        net.eval()
        self.net = net

    @torch.jit.script_method
    def helper(self, val):
        with torch.no_grad():
            # NHWC (RGB) tensor to NCHW tensor
            # And then again to NHWC tensor
            float_input = val.float() / 255
            float_input = float_input * 2 - 1
            val = float_input.transpose_(2, 3).transpose_(1, 2)
            out = self.net(val)
            out = out.clip(-1, 1) * 0.5 + 0.5
            out = (out * 255).byte()
            out = out.transpose_(1, 2).transpose_(2, 3)
        return out

    @torch.jit.script_method
    def forward(self, val):
        return self.helper(val)

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

    animae_net = MyScriptModule(args)
    # HWC
    traced_script_module = torch.jit.trace(animae_net, torch.rand(1, 480, 640, 3).to(device))
    traced_script_module.save("pytorch_animae_ganv2_paprika.pt")
    # print(traced_script_module)
    print("traced_script_module saved.")

    # This is testing code to verify that the generated model file is working correctly
    image_file = './samples/inputs/opencv_frame_0.png'
    img = load_image(image_file, args.x32)
    img_np = np.array(img)
    img_t = torch.Tensor(img_np).unsqueeze(0)
    try:
        while(1):
            out = traced_script_module(img_t.to(device))
        # out = traced_script_module(img_t.to(device)).cpu()
        out = out.transpose_(2, 3).transpose_(1, 2)
        out = out.squeeze(0).float() / 255
        out = to_pil_image(out)
        out.save('./samples/results/opencv_frame_0_jit_result.png')
    except:
        print("Error from image workload")

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
        # default='cpu',
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
