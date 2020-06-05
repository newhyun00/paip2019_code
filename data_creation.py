import numpy as np
import random
import os, csv, glob, shutil
from shutil import copy
from PIL import Image
from skimage import io, color
import scipy.misc
from datetime import datetime
from skimage import transform
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import data, exposure, img_as_float
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
 
import cv2
from skimage import io
import warnings

import subprocess
import collections
from multiprocess import Pool, TimeoutError, cpu_count

import gc
 
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

IMAGE_SIZE = 512

from albumentations import Resize

def create_folders():
    coverage_check_folder = './Data/Coverage_1500_Balanced_20x/'

    train_image_folder = './Data/Train_1500_Balanced_20x/Image/Images/'
    train_label_folder = './Data/Train_1500_Balanced_20x/Label/Labels/'

    valid_image_folder = './Data/Valid_20x/Image/Images/'
    valid_label_folder = './Data/Valid_20x/Label/Labels/'

    folders=[]
    folders.append(coverage_check_folder)
    folders.append(train_image_folder)
    folders.append(train_label_folder)
    folders.append(valid_image_folder)
    folders.append(valid_label_folder)

    for i in range(len(folders)):
        if os.path.exists(folders[i]):
            shutil.rmtree(folders[i])
        os.makedirs(folders[i])

def create_5_folds_valid_wsis():
    total_train_file_ids = []
    wsi_list = sorted(glob.glob('/PAIP2019_Keras_Org/Data/Total_Train_20x/*.tiff'))

    for i in range(len(wsi_list)):
        base_name = os.path.splitext(os.path.basename(wsi_list[i]))[0]
        if not (base_name.endswith('merge')):
            total_train_file_ids.append(base_name)

    random.shuffle(total_train_file_ids)

    for i in range(5):
        fold_string = '%01d'%(i+1)
        valid_file_ids = total_train_file_ids[10*i: 10*(i+1)]
        np.save('./Data/Valid_20x_'+fold_string+'.npy', valid_file_ids)


def create_train_valid_wsis(k_fold_valid_file_path):
    train_file_ids = []
    valid_file_ids = np.load(k_fold_valid_file_path)
    wsi_list = sorted(glob.glob('/PAIP2019_Keras_Org/Data/Total_Train_20x/*.tiff'))

    for i in range(len(wsi_list)):
        base_name = os.path.splitext(os.path.basename(wsi_list[i]))[0]
        if not (base_name in valid_file_ids):
            if not (base_name.endswith('merge')):
                train_file_ids.append(base_name)

    return valid_file_ids, train_file_ids

def generate_valid_patches(item):
    imgFile_id = item.file_id
    imgOutput = item.output

    print(imgFile_id, imgOutput)

    image_path = '/PAIP2019_Keras_Org/Data/Total_Train_20x/' + imgFile_id + '.tiff'
    label_path = '/PAIP2019_Keras_Org/Data/Total_Train_20x/' + imgFile_id + '_merge.tiff'

    image = imread(image_path)
    label = imread(label_path)

    image_height = image.shape[0]
    image_width = image.shape[1]

    print('File ID: ', imgFile_id, ' Org Height: ', image_height, ' Org Width: ', image_width)

    filenumber = 0
    sliding_window_size = IMAGE_SIZE + IMAGE_SIZE
    heights = (image_height-IMAGE_SIZE)//sliding_window_size + 1
    widths = (image_width-IMAGE_SIZE)//sliding_window_size + 1

    for j in range(heights):
        for k in range(widths):
            image_patch = image[j*sliding_window_size:j*sliding_window_size+IMAGE_SIZE, k*sliding_window_size:k*sliding_window_size+IMAGE_SIZE, 0:3]
            label_patch = label[j*sliding_window_size:j*sliding_window_size+IMAGE_SIZE, k*sliding_window_size:k*sliding_window_size+IMAGE_SIZE]

            if(np.average(image_patch)<227.0):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    filenumber_str = '%07d' % (filenumber+1)
                    imsave(imgOutput + 'Image/Images/' + imgFile_id + filenumber_str+'.png', image_patch)
                    imsave(imgOutput + 'Label/Labels/' + imgFile_id + filenumber_str+'.png', label_patch)
                #print('Patch saved: ' + imgFile_id + filenumber_str)
                filenumber+=1

    gc.collect()

def generate_train_patches(item):
    imgFile_id = item.file_id
    imgOutput = item.output

    print(imgFile_id, imgOutput)

    image_path = '/PAIP2019_Keras_Org/Data/Total_Train_20x/' + imgFile_id + '.tiff'
    label_path = '/PAIP2019_Keras_Org/Data/Total_Train_20x/' + imgFile_id + '_merge.tiff'

    image = imread(image_path)
    label = imread(label_path)

    image_height = image.shape[0]
    image_width = image.shape[1]

    print('File ID: ', imgFile_id, ' Height: ', image_height, ' Width: ', image_width)

    image_width_start = 0
    image_width_end = image_width - IMAGE_SIZE -1

    image_height_start = 0
    image_height_end = image_height - IMAGE_SIZE -1

    x_coord = 0
    y_coord = 0

    coverage = np.zeros((image_height, image_width), np.uint8)

    filenumber = 0
    label_valid_sum = float(np.sum(np.asarray(label==1, np.float32)))
    print('File ID: ', imgFile_id, ' Valid Sum: ', label_valid_sum)
    if(label_valid_sum>0):
        for j in range(500):
            picked = False
            while(picked == False):
                x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                image_patch = image[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE, 0:3]
                label_patch = label[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE]
                image_average = np.average(image_patch)
                if(image_average<227.0):
                    back_sum = float(np.sum(np.asarray(label_patch==0, np.uint8)))
                    valid_sum = float(np.sum(np.asarray(label_patch==1, np.uint8)))
                    whole_sum = float(np.sum(np.asarray(label_patch==2, np.uint8)))
                    if((valid_sum > whole_sum) and (valid_sum > back_sum)):
                        picked = True
                else:
                    picked = False

            filenumber_str = '%07d' % (filenumber+1)
            image_patch = image[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE, 0:3]
            label_patch = label[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE]
            coverage[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE] = 170
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(imgOutput + 'Image/Images/' + imgFile_id + filenumber_str+'.png', image_patch)
                imsave(imgOutput + 'Label/Labels/' + imgFile_id + filenumber_str+'.png', label_patch)
            #print('Patch saved: ' + imgFile_id + filenumber_str, ' Image Patch Average: ', image_average)
            filenumber+=1

    label_whole_sum = float(np.sum(np.asarray(label==2, np.float32)))
    print('File ID: ', imgFile_id, ' Whole Sum: ', label_whole_sum)
    if(label_whole_sum>0):
        for j in range(500):
            picked = False
            while(picked == False):
                x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                image_patch = image[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE, 0:3]
                label_patch = label[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE]
                image_average = np.average(image_patch)
                if(image_average<227.0):
                    back_sum = float(np.sum(np.asarray(label_patch==0, np.uint8)))
                    valid_sum = float(np.sum(np.asarray(label_patch==1, np.uint8)))
                    whole_sum = float(np.sum(np.asarray(label_patch==2, np.uint8)))
                    if((whole_sum > valid_sum) and (whole_sum > back_sum)):
                        picked = True
                else:
                    picked = False

            filenumber_str = '%07d' % (filenumber+1)
            image_patch = image[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE, 0:3]
            label_patch = label[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE]
            coverage[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE] = 255
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(imgOutput + 'Image/Images/' + imgFile_id + filenumber_str+'.png', image_patch)
                imsave(imgOutput + 'Label/Labels/' + imgFile_id + filenumber_str+'.png', label_patch)
            #print('Patch saved: ' + imgFile_id + filenumber_str, ' Image Patch Average: ', image_average)
            filenumber+=1

    label_back_sum = float(np.sum(np.asarray(label==0, np.float32)))
    print('File ID: ', imgFile_id, ' Back Sum: ', label_back_sum)
    if(label_back_sum>0):
        for j in range(500):
            picked = False
            while(picked == False):
                x_coord = random.sample(range(image_width_start, image_width_end), 1)[0]
                y_coord = random.sample(range(image_height_start, image_height_end), 1)[0]
                image_patch = image[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE, 0:3]
                label_patch = label[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE]
                image_average = np.average(image_patch)
                if(image_average<227.0):
                    back_sum = float(np.sum(np.asarray(label_patch==0, np.uint8)))
                    valid_sum = float(np.sum(np.asarray(label_patch==1, np.uint8)))
                    whole_sum = float(np.sum(np.asarray(label_patch==2, np.uint8)))
                    if((back_sum > valid_sum) and (back_sum > whole_sum)):
                        picked = True
                else:
                    picked = False

            filenumber_str = '%07d' % (filenumber+1)
            image_patch = image[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE, 0:3]
            label_patch = label[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE]
            coverage[y_coord:y_coord+IMAGE_SIZE, x_coord:x_coord+IMAGE_SIZE] = 85
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(imgOutput + 'Image/Images/' + imgFile_id + filenumber_str+'.png', image_patch)
                imsave(imgOutput + 'Label/Labels/' + imgFile_id + filenumber_str+'.png', label_patch)
            #print('Patch saved: ' + imgFile_id + filenumber_str, ' Image Patch Average: ', image_average)
            filenumber+=1

    aug = Resize(p=1.0, height = image_height//4, width = image_width//4)
    augmented = aug(image = coverage, mask = coverage)
    coverage_resized = augmented['mask']
    #coverage_small = np.round(resize(coverage, (image_height//4, image_width//4), preserve_range=True)).astype("uint8")
    imsave('./Data/Coverage_1500_Balanced_20x/' + imgFile_id + '_coverage.png', coverage_resized)

    gc.collect()

def multi_generate_train(file_ids):
    patch_path = './Data/Train_1500_Balanced_20x/'

    FilePairs = collections.namedtuple('FilePairs', ['file_id', 'output'])
    pairs = []

    for i in range(len(file_ids)):
        pairs.append(FilePairs(file_id = file_ids[i], output = patch_path))

    pairsTuple = tuple(pairs)

    pool = Pool(processes=5)
    pool.map(generate_train_patches, pairsTuple)

def multi_generate_valid(file_ids):
    patch_path = './Data/Valid_20x/'

    FilePairs = collections.namedtuple('FilePairs', ['file_id', 'output'])
    pairs = []

    for i in range(len(file_ids)):
        pairs.append(FilePairs(file_id = file_ids[i], output = patch_path))

    pairsTuple = tuple(pairs)

    pool = Pool(processes=5)
    pool.map(generate_train_patches, pairsTuple)


create_folders()
create_5_folds_valid_wsis()
valid_file_ids, train_file_ids = create_train_valid_wsis('./Data/Valid_20x_1.npy')
print('Valid File IDS: ', valid_file_ids)
print('Length of Valid File IDS: ', len(valid_file_ids))
print('Train File IDS: ', train_file_ids)
print('Length of Train File IDS: ', len(train_file_ids))
multi_generate_train(train_file_ids)
multi_generate_valid(valid_file_ids)





