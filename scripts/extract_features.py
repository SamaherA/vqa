import argparse, os, json
import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import h5py
from scipy.misc import imread, imresize, imshow
import imageio
import matplotlib.pyplot
from IPython.display import Image , display


def build_model(cur_batch):
    model = ResNet50(weights='imagenet')

    img_path = cur_batch
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feats = model.predict(x)
    return feats


input_paths = []
idx_set = set()
max_images= None

for fn in os.listdir(input_image_dir):
    if not fn.endswith('.png'): continue
    idx = int(os.path.splitext(fn)[0].split('_')[-1])
    input_paths.append((os.path.join(input_image_dir, fn), idx))
    idx_set.add(idx)


input_paths.sort(key=lambda x: x[1])
assert len(idx_set) == len(input_paths)
assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1

if max_images is not None:
    input_paths = input_paths[:max_images]

# print(input_paths[0])
# print(input_paths[-1])

image_height = 224
image_width = 224
output_h5_file = "/train_features.h5"
batch_size = 128


img_size = (image_height, image_width)
with h5py.File(output_h5_file, 'w') as f:
    feat_dset = None
    i0 = 0
    cur_batch = []
    for i, (path, idx) in enumerate(input_paths):
        img = imread(path, mode='RGB')
        imgg = imresize(img, img_size, interp='bicubic')
        imggg = imgg.transpose(2, 0, 1)[None]

        cur_batch.append(imggg)
        if len(cur_batch) == batch_size:
#             feats = run_batch(cur_batch)
             if feat_dset is None:
                N = len(input_paths)
                 _, C, H, W = feats.shape
                feat_dset = f.create_dataset('features', (N, C, H, W),
                                       dtype=np.float32)
            i1 = i0 + len(cur_batch)
            feat_dset[i0:i1] = feats
            i0 = i1
            print('Processed %d / %d images' % (i1, len(input_paths)))
            cur_batch = []
    if len(cur_batch) > 0:
#         feats = run_batch(cur_batch, model)
        i1 = i0 + len(cur_batch)
        feat_dset[i0:i1] = feats
        print('Processed %d / %d images' % (i1, len(input_paths)))