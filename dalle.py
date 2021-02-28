import io
import os
import sys
import requests
import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import clip
import PIL
from dall_e import map_pixels, unmap_pixels, load_model
from IPython.display import display, display_markdown

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_path',
    type=str,
    default='./generations',
    help='',
)
parser.add_argument(
    '--ref_img_path',
    type=str,
    default="./muskete.jpeg",
    help='',
)
parser.add_argument(
    '--prompt',
    type=str,
    default='A delicious avocado',
    help='',
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-1,
    help='',
)
parser.add_argument(
    '--img_save_freq',
    type=int,
    default=5,
    help='',
)

args = parser.parse_args()

output_path = args.output_path
ref_img_path = args.ref_img_path
prompt = args.prompt
lr = args.lr
img_save_freq = args.img_save_freq

output_dir = os.path.join(output_path, f'"{prompt}"')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

target_image_size = 256


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def preprocess(img):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return map_pixels(img)


def compute_clip_loss(img, text):
    img = clip_transform(img)
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    img_logits = clip_model.encode_image(img)

    tokenized_text = clip.tokenize([text]).to(device).detach().clone()
    text_logits = clip_model.encode_text(tokenized_text)

    loss = 10 * -torch.cosine_similarity(text_logits, img_logits).mean()

    return loss


def get_stacked_random_crops(img, num_random_crops=64):
    img_size = [img.shape[2], img.shape[3]]
    crop_list = []
    for _ in range(num_random_crops):
        crop_size = int(img_size[0] * torch.zeros(1, ).uniform_(.5, .99))

        x_offset = torch.randint(0, img_size[1] - crop_size, ())
        y_offset = torch.randint(0, img_size[0] - crop_size, ())

        crop = img[:, :, x_offset:x_offset + crop_size,
                   y_offset:y_offset + crop_size]
        crop = torch.nn.functional.interpolate(
            crop,
            (224, 224),
            mode='bilinear',
        )

        crop_list.append(crop)

    img = torch.cat(crop_list, axis=0)

    return img


clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
clip_transform = torchvision.transforms.Compose([
    # clip_preprocess.transforms[2],
    clip_preprocess.transforms[4],
])

dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", device)
dec.eval()

z_logits = torch.rand((1, 8192, 64, 64)).cuda()
z_logits = torch.argmax(z_logits, axis=1)
z_logits = F.one_hot(z_logits, num_classes=8192).permute(0, 3, 1, 2).float()

z_logits = torch.nn.Parameter(z_logits, requires_grad=True)

optimizer = torch.optim.Adam(
    params=[z_logits],
    lr=lr,
    betas=(0.9, 0.999),
)

counter = 0
while True:
    z = torch.nn.functional.gumbel_softmax(
        z_logits.permute(0, 2, 3, 1).view(1, 64**2, 8192),
        hard=False,
        dim=1,
    ).view(1, 8192, 64, 64)

    x_stats = dec(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

    x_rec = get_stacked_random_crops(x_rec, num_random_crops=16)

    loss = compute_clip_loss(x_rec, prompt)

    print(loss)
    print(z_logits[0, 0, 0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    counter += 1

    if counter % img_save_freq == 0:
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
        x_rec.save(f"{output_dir}/{counter}.png")
