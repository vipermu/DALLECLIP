import os
import argparse
import glob

from PIL import Image

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_folder',
    type=str,
    help='',
)
parser.add_argument(
    '--video_name',
    type=str,
    help='',
)
parser.add_argument(
    '--out_dir',
    type=str,
    default='./gifs',
)
parser.add_argument(
    '--rate',
    type=int,
    default=16,
    help='',
)
args = parser.parse_args()
print(args)

image_folder = args.image_folder
video_name = args.video_name

out_dir = args.out_dir
fp_out = os.path.join(out_dir, f"{video_name}.gif")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
images = [int(x[:-4]) for x in images]
images.sort()
images = [f"{x}.png" for x in images]
# images = [f"{x}.jpg" for x in images]

print(image_folder)
print(images)

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(os.path.join(image_folder, img)) for img in images]
img = img.resize((200,200))
imgs = [im.resize((200,200)) for im in imgs]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=32, loop=0)