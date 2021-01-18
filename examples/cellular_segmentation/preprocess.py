import os
import cv2
import numpy as np
import tqdm
import glob
import zipfile
import urllib.request
from shutil import copyfile

def my_hook(t): # https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to

def download(url, out):

    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not os.path.exists(out):
        with tqdm.tqdm(unit='B', unit_scale=True, miniters=1, ncols=80) as t:
            urllib.request.urlretrieve (url, out, reporthook=my_hook(t))

def unzip(zip_file, out):

    os.makedirs(os.path.dirname(out), exist_ok=True)

    with zipfile.ZipFile(zip_file) as existing_zip:
        existing_zip.extractall(out)

def preprocess_images(files, out_dir):

    commonpath = os.path.commonpath(files)

    for f in tqdm.tqdm(files):
        out = os.path.join(out_dir, os.path.relpath(f, commonpath))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        copyfile(f, out)

def preprocess_labels(files, out_dir, binary=True):

    commonpath = os.path.commonpath(files)

    for f in tqdm.tqdm(files):
        out = os.path.join(out_dir, os.path.relpath(f, commonpath))
        os.makedirs(os.path.dirname(out), exist_ok=True)

        src = cv2.imread(f)
        src = src[:,:,0]

        dst = np.zeros(src.shape, src.dtype)

        dst[src==255] = 1
        dst[src!=255] = 0


        cv2.imwrite(out, dst)

def preprocess_train(out_dir, temp_dir=None):

    train_dir = '/content/pytorch_bayesian_unet/examples/cellular_segmentation/train'

    train_image_files = glob.glob(os.path.join(train_dir, '**', '*_raw.png'), recursive=True)
    print('# train images:', len(train_image_files))
    preprocess_images(train_image_files, os.path.join(out_dir, 'train'))

    train_label_files = glob.glob(os.path.join(train_dir, '**', '*_class.png'), recursive=True)
    print('# train labels:', len(train_label_files))
    preprocess_labels(train_label_files, os.path.join(out_dir, 'train'))

def preprocess_test(out_dir, temp_dir=None):
    test_dir = '/content/pytorch_bayesian_unet/examples/cellular_segmentation/test'


    test_image_files = glob.glob(os.path.join(test_dir, '**', '*_raw.png'), recursive=True)
    print('# test images:', len(test_image_files))
    preprocess_images(test_image_files, os.path.join(out_dir, 'test'))

    test_label_files = glob.glob(os.path.join(test_dir, '**', '*_class.png'), recursive=True)
    print('# test labels:', len(test_label_files))
    preprocess_labels(test_label_files, os.path.join(out_dir, 'test'))


if __name__ == '__main__':

    out_dir = './preprocessed'

    preprocess_train(out_dir)
    preprocess_test(out_dir)
