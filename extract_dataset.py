import numpy as np
import argparse, math, shutil, os, json, time
from PIL import Image

parser = argparse.ArgumentParser(description='Edge Impulse => YOLOv5')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)
parser.add_argument('--model-size', type=str, help='e.g. "n" for nano', required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory, 'X_split_test.npy'), mmap_mode='r')

with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
    Y_train = json.loads(f.read())
with open(os.path.join(args.data_directory, 'Y_split_test.npy'), 'r') as f:
    Y_test = json.loads(f.read())

image_width, image_height, image_channels = list(X_train.shape[1:])
if image_width != image_height:
    print('Error: only square images are supported (width != height)')
    exit(1)

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with YOLOv5')

def current_ms():
    return round(time.time() * 1000)

total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = current_ms()
converted_images = 0

def convert(X, Y, category):
    global class_count, total_images, zf, last_printed, converted_images

    for ix in range(0, len(X)):
        img_shape = ()
        if image_channels == 1:
            img_shape = (image_width, image_height)
        else:
            img_shape = (image_width, image_height, image_channels)

        raw_img_data = (np.reshape(X[ix], img_shape) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        images_dir = os.path.join(out_dir, category, 'images')
        labels_dir = os.path.join(out_dir, category, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        im = Image.fromarray(raw_img_data)
        im.save(os.path.join(images_dir, 'image' + str(ix).zfill(5) + '.jpg'))

        labels_text = []
        for l in labels:
            if (l['label'] > class_count):
                class_count = l['label']

            x = l['x']
            y = l['y']
            w = l['w']
            h = l['h']

            # class x_center y_center width height
            x_center = (x + (w / 2)) / image_width
            y_center = (y + (h / 2)) / image_height
            width = w / image_width
            height = h / image_height

            labels_text.append(str(l['label'] - 1) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height))

        with open(os.path.join(labels_dir, 'image' + str(ix).zfill(5) + '.txt'), 'w') as f:
            f.write('\n'.join(labels_text))

        converted_images = converted_images + 1
        if (converted_images == 1 or current_ms() - last_printed > 3000):
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = current_ms()

convert(X=X_train, Y=Y_train, category='train')
convert(X=X_test, Y=Y_test, category='valid')

print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')

print('Transforming Edge Impulse data format into something compatible with YOLOv5 OK')
print('')

class_names = []
for c in range(0, class_count):
    class_names.append("'class" + str(c) + "'")
class_names = ', '.join(class_names)

data_yaml = """
train: """ + os.path.join(os.path.abspath(out_dir), 'train', 'images') + """
val: """ + os.path.join(os.path.abspath(out_dir), 'valid', 'images') + """

nc: """ + str(class_count) + """
names: [""" + class_names + """]
"""

with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
    f.write(data_yaml)
