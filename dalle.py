import io
import os
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

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_path',
    type=str,
    default='./generations',
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
    default=3e-1,
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
prompt = args.prompt
lr = args.lr
img_save_freq = args.img_save_freq

output_dir = os.path.join(output_path, f'"{prompt}"')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

target_img_size = 256
final_img_size = 512


def preprocess(img):
    min_img_dim = min(img.size)

    if min_img_dim < target_img_size:
        raise ValueError(f'min dim for img {min_img_dim} < {target_img_size}')

    img_ratio = target_img_size / min_img_dim
    min_img_dim = (round(img_ratio * img.size[1]),
                   round(img_ratio * img.size[0]))
    img = TF.resize(img, min_img_dim, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_img_size])
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
        crop_size_y = int(img_size[0] * torch.zeros(1, ).uniform_(.75, .95))
        crop_size_x = int(img_size[1] * torch.zeros(1, ).uniform_(.75, .95))

        y_offset = torch.randint(0, img_size[0] - crop_size_y, ())
        x_offset = torch.randint(0, img_size[1] - crop_size_x, ())

        crop = img[:, :, y_offset:y_offset + crop_size_y,
                   x_offset:x_offset + crop_size_x]

        crop = torch.nn.functional.upsample_bilinear(crop, (224, 224))

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

scale_x = 2
scale_y = 1

z_logits = torch.rand((1, 8192, 64 * scale_y, 64 * scale_x)).cuda()
# z_logits = torch.argmax(z_logits, axis=1)
# z_logits = F.one_hot(z_logits, num_classes=8192).permute(0, 3, 1, 2).float()

z_logits = torch.nn.Parameter(z_logits, requires_grad=False)

optimizer = torch.optim.Adam(
    params=[z_logits],
    lr=lr,
    betas=(0.9, 0.999),
)

final_x_rec = torch.zeros(
    [3, final_img_size * scale_y, final_img_size * scale_x])

counter = 0
rec_steps = 4
x_rec_merged = None
while True:
    for s_y in range(scale_y):
        for s_x in np.linspace(0, scale_x - 1, rec_steps):
            z_logits_part = z_logits[:, :, int(64 * s_y):int(64 * (s_y + 1)),
                         int(64 * s_x):int(64 * (s_x + 1))]

            # z = torch.nn.functional.gumbel_softmax(
            #     z_logits[:, :, int(64 * s_y):int(64 * (s_y + 1)),
            #              int(64 * s_x):int(64 * (s_x + 1))].permute(0, 2, 3, 1).reshape(
            #                  1, 64**2, 8192),
            #     hard=False,
            #     dim=1,
            # ).view(1, 8192, 64, 64)

            z = torch.nn.functional.gumbel_softmax(
                z_logits_part.permute(0, 2, 3, 1).reshape(1, 64**2, 8192),
                hard=False,
                dim=1,
            ).view(1, 8192, 64, 64)

            x_stats = dec(z).float()
            x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))

            x_rec_stacked = get_stacked_random_crops(
                x_rec,
                num_random_crops=16,
            )

            if x_rec_merged is None:
                x_rec_merged = x_rec
            else:
                step_img_size = int(final_img_size / rec_steps)

                y_init_img_part = step_img_size * s_y
                x_init_img_part = step_img_size * s_x
                y_final_img_part = y_init_img_part + step_img_size * (s_y + 1)
                x_final_img_part = x_init_img_part + step_img_size * (s_x + 1)

                x_rec_merged[:, :, y_init_img_part:y_final_img_part,
                             x_init_img_part:
                             x_final_img_part] = x_rec[:, :, 0:(step_img_size *
                                                                (s_y + 1)),
                                                       0:(step_img_size *
                                                          (s_x + 1))]

            final_x_rec[:,
                        int(512 * s_y):int(512 * (s_y + 1)),
                        int(512 * s_x):int(512 * (s_x + 1))] = x_rec_merged

            final_x_rec_stacked = get_stacked_random_crops(
                x_rec_merged,
                num_random_crops=64,
            )

            part_loss = compute_clip_loss(x_rec_stacked, prompt)
            final_loss = compute_clip_loss(final_x_rec_stacked, prompt)
            loss = (part_loss + final_loss)/2

            print(loss)
            # print(z_logits[0, 0, 0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    counter += 1
    if counter % img_save_freq == 0:
        x_rec = T.ToPILImage(mode='RGB')(final_x_rec)
        x_rec.save(f"{output_dir}/{counter}.png")
