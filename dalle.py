
import torch.nn.functional as F
import io
import os
import sys
import requests
import PIL
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import clip

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
batch_size = 1

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()
clip_transform = torchvision.transforms.Compose([
    # clip_preprocess.transforms[2],
    clip_preprocess.transforms[4],
])

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
    tokenized_text = clip.tokenize([text]).to(device).detach().clone()
    text_logits = clip_model.encode_text(tokenized_text)
    img_logits = clip_model.encode_image(img)

    loss = 10 * -torch.cosine_similarity(text_logits, img_logits).mean()

    return loss

    # img_logits, _text_logits = clip_model(img, tokenized_text)

    # return 1/img_logits * 100



class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()

        hots = torch.nn.functional.one_hot((torch.arange(0, 8192).to(torch.int64)), num_classes=8192)
        rng = torch.zeros(batch_size, 64*64, 8192).uniform_()**torch.zeros(batch_size, 64*64, 8192).uniform_(.1,1)
        for b in range(batch_size):
          for i in range(64**2):
            rng[b,i] = hots[[np.random.randint(8191)]]

        rng = rng.permute(0, 2, 1)

        self.normu = torch.nn.Parameter(rng.cuda().view(batch_size, 8192, 64, 64))
        
    def forward(self):
      normu = torch.softmax(hadies*self.normu.reshape(batch_size, 8192//2, -1), dim=1).view(batch_size, 8192, 64, 64)
      return normu

_ = Pars()

# x = PIL.Image.open(ref_img_path)

# For faster load times, download these files locally and use the local paths instead.
# enc = load_model("https://cdn.openai.com/dall-e/encoder.pkl", device)
dec = load_model("https://cdn.openai.com/dall-e/decoder.pkl", device)

# enc.eval()
dec.eval()

# x = preprocess(x).to(device)

# z_logits = enc(x).detach().clone()

# batch_size = 1
# hots = torch.nn.functional.one_hot((torch.arange(0, 8192).to(torch.int64)), num_classes=8192)
# rng = torch.zeros(batch_size, 64*64, 8192).uniform_()**torch.zeros(batch_size, 64*64, 8192).uniform_(.1,1)
# for b in range(batch_size):
#     for i in range(64**2):
#         rng[b,i] = hots[[np.random.randint(8191)]]

# rng = rng.permute(0, 2, 1)

# z_logits = torch.nn.Parameter(rng.cuda().view(batch_size, 8192, 64, 64))
hots = torch.nn.functional.one_hot((torch.arange(0, 8192).to(torch.int64)), num_classes=8192)
z_logits = torch.zeros(batch_size, 64**2, 8192)
for b in range(batch_size):
    for i in range(64**2):
        z_logits[b,i] = hots[[np.random.randint(8191)]]
        
z_logits = torch.nn.Parameter(z_logits.cuda().view(batch_size, 8192, 64, 64))
# z_logits = torch.nn.Parameter(torch.rand((1,8192,64,64)).cuda(), requires_grad=True)

optimizer = torch.optim.Adam(
    params=[z_logits],
    lr=lr,
    betas=(0.9, 0.999),
)

# NOTE shape [1, 8192, 32, 32]
hadies = 1
will_it = False
counter = 0
while True:
    z = torch.nn.functional.gumbel_softmax(hadies*z_logits.reshape(batch_size, 8192//2, -1), hard=will_it, dim=1).view(batch_size, 8192, 64, 64)
    # z_logits = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1)
    # z = torch.argmax(z_logits*hadies, axis=1)
    # z = F.one_hot(z, num_classes=8192).permute(0, 3, 1, 2).float()
    # z.requires_grad = True

    x_stats = dec(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
  
    img_size = 512
    p_s = []
    cutn = 64 # improves quality
    for _ch in range(cutn):
        size = int(img_size*torch.zeros(1,).uniform_(.5, .99))#.normal_(mean=.8, std=.3).clip(.5, .98))
        offsetx = torch.randint(0, img_size - size, ())
        offsety = torch.randint(0, img_size - size, ())
        apper = x_rec[:, :, offsetx:offsetx + size, offsety:offsety + size]
        apper = torch.nn.functional.interpolate(apper, (224,224), mode='bilinear')
        p_s.append(apper)
    x_rec = torch.cat(p_s, axis=0)

    loss = compute_clip_loss(x_rec, prompt)

    print(loss)
    print(z_logits[0,0,0])
    
    optimizer.zero_grad()
    loss.backward()
    
    # z_logits.grad = z.grad

    optimizer.step()

    counter += 1
    
    if counter % img_save_freq == 0:
        x_rec = T.ToPILImage(mode='RGB')(x_rec[0])
        x_rec.save(f"{output_dir}/{counter}.png")
