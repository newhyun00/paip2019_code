## UNET++ model architecture code is mostly from Unet Plus Plus with EfficientNet Encoder from Kaggle
## https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder

## Snapshot ensemble code for Keras is from the link below
## https://github.com/titu1994/Snapshot-Ensembles    

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import gc
import scipy.signal

from keras.models import *
from keras import models
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.utils.layer_utils import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.activations import relu
from keras.layers import LeakyReLU, MaxPooling2D, Conv2DTranspose, concatenate, ReLU
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.callbacks import Callback
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.losses import binary_crossentropy
from efficientnet import EfficientNetB4

from Create_Data_1500_Balanced_20x import *

from keras_radam import RAdam

import shutil
import sys
from datetime import datetime
import threading
from tifffile import imsave
from skimage.io import imread
from skimage.transform import resize
from threading import Thread
import cv2
from tqdm import tqdm

from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, CLAHE, GaussianBlur, Transpose, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf, Resize,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomBrightnessContrast, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,OpticalDistortion,RandomSizedCrop, PadIfNeeded
)

tf.compat.v1.disable_eager_execution()
clahe = cv2.createCLAHE(clipLimit=2.0)

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

# Normalizing only frame images, since masks contain label info
data_gen_args = dict()
mask_gen_args = dict()

train_frames_datagen = ImageDataGenerator(**data_gen_args)
train_masks_datagen = ImageDataGenerator(**mask_gen_args)
valid_frames_datagen = ImageDataGenerator(**data_gen_args)
valid_masks_datagen = ImageDataGenerator(**mask_gen_args)

# Seed defined for aligning images and their masks


def TrainAugmentGenerator(seed = 1, batch_size = 1):
    '''Train Image data generator
        Inputs:
            seed - seed provided to the flow_from_directory function to ensure aligned data flow
            batch_size - number of images to import at a time
        Output: Decoded RGB image (height x width x 3)
    '''
    train_image_generator = train_frames_datagen.flow_from_directory(
    './Data/Train_1500_Balanced_20x/Image/',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    color_mode="rgb",
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

    train_mask_generator = train_masks_datagen.flow_from_directory(
    './Data/Train_1500_Balanced_20x/Label/',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    color_mode="grayscale",
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

    while True:
        X1i = train_image_generator.next()
        X2i = train_mask_generator.next()

        mask_encoded = tf.keras.utils.to_categorical(X2i[0], num_classes=NUM_CLASSES)

        yield X1i[0]/255.0, mask_encoded

def ValAugmentGenerator(seed = 1, batch_size = 1):
    valid_image_generator = valid_frames_datagen.flow_from_directory(
    './Data/Valid_20x/Image/',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    color_mode="rgb",
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

    valid_mask_generator = valid_masks_datagen.flow_from_directory(
    './Data/Valid_20x/Label/',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    color_mode="grayscale",
    batch_size=batch_size,
    shuffle=True,
    seed=seed)

    while True:
        X1i = valid_image_generator.next()
        X2i = valid_mask_generator.next()

        mask_encoded = tf.keras.utils.to_categorical(X2i[0], num_classes=NUM_CLASSES)

        yield X1i[0]/255.0, mask_encoded

def dice_coef(y_true, y_pred, smooth = 1):
    sparse_y_true = K.argmax(y_true, axis=-1)
    sparse_y_pred = K.argmax(y_pred, axis=-1)
    sum_dice_coef = 0.0

    sparse_y_true_flat = K.flatten(sparse_y_true)
    sparse_y_pred_flat = K.flatten(sparse_y_pred)

    for i in range(NUM_CLASSES):
        sparse_y_true_flat_class = K.cast(K.equal(sparse_y_true_flat, i), 'float32')
        sparse_y_pred_flat_class = K.cast(K.equal(sparse_y_pred_flat, i), 'float32')

        intersection = K.sum(K.abs(sparse_y_true_flat_class * sparse_y_pred_flat_class))
        sum_ = K.sum(K.abs(sparse_y_true_flat_class)) + K.sum(K.abs(sparse_y_pred_flat_class))

        sum_dice_coef += ((2. * intersection+smooth)/(sum_+smooth))/float(NUM_CLASSES)

    return sum_dice_coef

def soft_jaccard_loss(y_true, y_pred, smooth=1):
    sum_jaccard_index = 0.0

    for i in range(NUM_CLASSES):
        intersection = K.sum(K.abs(y_true[:, :, :, i] * y_pred[:, :, :, i]))
        union = K.sum(K.abs(y_true[:, :, :, i]*y_true[:, :, :, i])) + \
                K.sum(K.abs(y_pred[:, :, :, i]*y_pred[:, :, :, i])) - intersection

        sum_jaccard_index += ((intersection+smooth)/(union+smooth))/float(NUM_CLASSES)

    return 1.0 - sum_jaccard_index

def jaccard_index(y_true, y_pred, smooth=1):
    sparse_y_true = K.argmax(y_true, axis=-1)
    sparse_y_pred = K.argmax(y_pred, axis=-1)
    sum_jaccard_index = 0.0

    sparse_y_true_flat = K.flatten(sparse_y_true)
    sparse_y_pred_flat = K.flatten(sparse_y_pred)

    for i in range(NUM_CLASSES):
        sparse_y_true_flat_class = K.cast(K.equal(sparse_y_true_flat, i), 'float32')
        sparse_y_pred_flat_class = K.cast(K.equal(sparse_y_pred_flat, i), 'float32')

        intersection = K.sum(K.abs(sparse_y_true_flat_class * sparse_y_pred_flat_class))
        union = K.sum(K.abs(sparse_y_true_flat_class)) + K.sum(K.abs(sparse_y_pred_flat_class)) - intersection

        sum_jaccard_index += ((intersection+smooth)/(union+smooth))/float(NUM_CLASSES)

    return sum_jaccard_index

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_index(y_true, y_pred)

def categorical_crossentropy(y_true, y_pred):
    return tf.clip_by_value(K.mean(K.categorical_crossentropy(y_true, y_pred)), -1.0e6, 1.0e6)

def combined_entropy_soft_jaccard_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + soft_jaccard_loss(y_true, y_pred)

def combined_entropy_jaccard_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)

def combined_entropy_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def UEfficientNet(input_shape=(None, None, 3), start_neurons=8, dropout_rate=0.1):

    backbone = EfficientNetB4(weights=INITIAL_WEIGHTS,
                            include_top=False,
                            input_shape=input_shape)

    input = backbone.input
    start_neurons = start_neurons

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

     # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 32)
    convm = residual_block(convm,start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4) 

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3,deconv4_up1, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2,deconv3_up1,deconv4_up2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1,deconv2_up1,deconv3_up2,deconv4_up3, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate/2)(uconv0)
    output_layer = Conv2D(NUM_CLASSES, (1,1), padding="same", activation="softmax")(uconv0)

    model = Model(input, output_layer)
    model.name = 'Upp_EffNet'

    return model

def run_gpu_inference(model, test_file_list, prediction_path):
    valid_tumor_ratio_list = []
    for m in range(len(test_file_list)):
        image = imread(test_file_list[m])

        height = image.shape[0]
        width = image.shape[1]

        basename_string = os.path.splitext(os.path.basename(test_file_list[m]))[0]
        filename_string = '%03d'%int(basename_string.split('_')[-1])

        IFBS = BATCH_SIZE*2
        VALID = IMAGE_SIZE//2
        PATCH_OFFSET = IMAGE_SIZE//4
        SLIDE_OFFSET = IMAGE_SIZE//2

        heights = height//(VALID*IFBS) + 1
        widths = width//(VALID*IFBS) + 1

        height_ext = VALID*IFBS*heights + PATCH_OFFSET*2
        width_ext = VALID*IFBS*widths + PATCH_OFFSET*2

        org_slide_ext = np.zeros((height_ext, width_ext, 3), np.uint8)
        prob_map_seg = np.zeros((height_ext, width_ext), dtype=np.float32)

        org_slide_ext[PATCH_OFFSET: PATCH_OFFSET+height, PATCH_OFFSET:PATCH_OFFSET+width, 0:3] = image[:, :, 0:3]

        test_patch = np.zeros((IFBS, IMAGE_SIZE, IMAGE_SIZE, 3), dtype = np.uint8)

        progress_count = 0

        for i in range(heights):
            for j in range(widths):
                for k in range(IFBS):
                    for l in range(IFBS):
                        test_patch[l, :, :, :] = org_slide_ext[(i*IFBS+k)*SLIDE_OFFSET:(i*IFBS+k)*SLIDE_OFFSET+IMAGE_SIZE, (j*IFBS+l)*SLIDE_OFFSET:(j*IFBS+l)*SLIDE_OFFSET+IMAGE_SIZE, 0:3]
                        progress_count+=1

                    prob_classes = model.predict(test_patch/255.0)
                    prob_argmax = np.argmax(prob_classes, axis = -1)
                    prob_patch_seg = np.reshape(prob_argmax, (IFBS, IMAGE_SIZE, IMAGE_SIZE))

                    for l in range(IFBS):
                        prob_map_seg[PATCH_OFFSET+(i*IFBS+k)*VALID: PATCH_OFFSET+(i*IFBS+k+1)*VALID, PATCH_OFFSET+(j*IFBS+l)*VALID: PATCH_OFFSET+(j*IFBS+l+1)*VALID] = \
                            prob_patch_seg[l, PATCH_OFFSET:PATCH_OFFSET+VALID, PATCH_OFFSET:PATCH_OFFSET+VALID]
                    print('Progress ' + filename_string + ' :', float(progress_count)/float(heights*widths*IFBS*IFBS)*100)


        data = {}
        prob_map_valid = prob_map_seg[PATCH_OFFSET:PATCH_OFFSET+height, PATCH_OFFSET:PATCH_OFFSET+width]

        aug_inf = Resize(p=1.0, height = height//4, width = width//4)
        augmented_inf = aug_inf(image=prob_map_valid, mask=prob_map_valid)
        prob_map_valid_resize = augmented_inf['mask']

        imsave(prediction_path + filename_string + '_Resized.tif', prob_map_valid_resize.astype('uint8')*127, compress=9)

        num_valid_tumors = np.sum((prob_map_valid == 1).astype('float32'))
        num_whole_tumors = np.sum((prob_map_valid == 2).astype('float32'))
        tumor_ratio = num_valid_tumors/(num_valid_tumors + num_whole_tumors)*100.

        imsave(prediction_path + filename_string + '_Whole.tif', prob_map_valid.astype('uint8')*127, compress=9)

        prob_map_valid[prob_map_valid==0] = 0 # Background
        prob_map_valid[prob_map_valid==1] = 1 # Valid Tumor
        prob_map_valid[prob_map_valid==2] = 0 # Whole Tumor - Valid Tumor

        tumor_mask =prob_map_valid.astype('uint8')

        data['wsi_id'] = filename_string
        data['ratio'] = tumor_ratio

        valid_tumor_ratio_list.append(data)

        imsave(prediction_path + filename_string + '.tif', tumor_mask, compress=9)
        print('Processsed Testing Slide Number:' + filename_string, ' Tumor ratio: ', tumor_ratio)

    task2_df = pd.DataFrame(valid_tumor_ratio_list, columns = ['wsi_id', 'ratio'])
    task2_df.to_csv(prediction_path + 'prediction.csv', index = False)

LEARNING_RATE = 0.000068
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_SIZE = 512
BATCH_SIZE = 12
NUM_CLASSES = 3
NUM_EPOCHS = 24
EPOCHS_PER_CYCLE = 8
NUM_SNAPSHOT = NUM_EPOCHS/EPOCHS_PER_CYCLE
SWA_EPOCHS = NUM_EPOCHS - EPOCHS_PER_CYCLE * (NUM_SNAPSHOT - 1)
NUM_GPUS = 4
SEED = 1
INITIAL_WEIGHTS = 'imagenet'
TB_DIR = './Weights/'

aug = Compose([
  OneOf([
    Blur(p=1.0, blur_limit=7),
    ShiftScaleRotate(p=1.0, shift_limit=0.0625, scale_limit=0.0, interpolation=2, rotate_limit=45, border_mode=0, value=(255,255,255), mask_value=0),
    ElasticTransform(p=1.0, alpha=IMAGE_SIZE, sigma=IMAGE_SIZE * 0.085, alpha_affine=IMAGE_SIZE * 0.02, border_mode=0, value=(255,255,255), mask_value=0),
    ], p=0.1)
  ])

class SWA(Callback):

    def __init__(self, filepath, swa_epoch, epochs_per_cycle):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch
        self.epochs_per_cycle = epochs_per_cycle

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) == self.swa_epoch:
            print('SWA Saving fist weight at currnt epoch: ', (epoch+1))
            self.swa_weights = self.model.get_weights()

        elif ((epoch+1) > self.swa_epoch) and ((epoch+1)%self.epochs_per_cycle == 0):
            print('SWA Weight averaging at currnt epoch: ', (epoch+1))
            num_models = (epoch+1 - self.swa_epoch)/self.epochs_per_cycle
            print('Number of Models: ', num_models)
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * num_models + self.model.get_weights()[i])/\
                                      (num_models + 1)
        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')

class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, folder_name, init_lr=LEARNING_RATE):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.folder_name = folder_name

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            ModelCheckpoint(mode='max', filepath=TB_DIR + self.folder_name + 'Training_{epoch:02d}-{val_jaccard_index:.4f}.h5',
                            monitor='val_jaccard_index', save_best_only='True', save_weights_only='True', verbose=1),
            swa,
            LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        print('Current epoch: ', (t+1), ' Learning rate: ', float(self.alpha_zero / 2 * cos_out))
        return float(self.alpha_zero / 2 * cos_out)

for i in range(5):
    K.clear_session()

    folder_name = 'Fold_' + '%01d'%(i+1) + '/'
    valid_file_path = './Data/Valid_20x_' + '%01d'%(i+1) + '.npy'
    prediction_path = './Predictions_Final_Test/' + folder_name

    if os.path.exists(TB_DIR + folder_name):
        shutil.rmtree(TB_DIR + folder_name)
    os.makedirs(TB_DIR + folder_name)

    if os.path.exists(prediction_path):
        shutil.rmtree(prediction_path)
    os.makedirs(prediction_path)

    if(i>0):
        create_folders()
        valid_file_ids, train_file_ids = create_train_valid_wsis(valid_file_path)
        print('Valid File IDS: ', valid_file_ids)
        print('Length of Valid File IDS: ', len(valid_file_ids))
        print('Train File IDS: ', train_file_ids)
        print('Length of Train File IDS: ', len(train_file_ids))
        multi_generate_train(train_file_ids)
        multi_generate_valid(valid_file_ids)

    snapshot = SnapshotCallbackBuilder(nb_epochs=NUM_EPOCHS,nb_snapshots=NUM_SNAPSHOT, folder_name = folder_name, init_lr=LEARNING_RATE)
    swa = SWA(TB_DIR + folder_name + 'keras_swa.h5', SWA_EPOCHS, EPOCHS_PER_CYCLE)
    model =UEfficientNet(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), start_neurons=8, dropout_rate=0.1)
    model.summary()

    optimizer = RAdam(lr=LEARNING_RATE)
    parallel_model = multi_gpu_model(model, gpus = NUM_GPUS)
    parallel_model.compile(optimizer=optimizer, loss=combined_entropy_jaccard_loss, metrics=[jaccard_index])
    parallel_model.summary()

    single_model = parallel_model.layers[-2]
    single_model.save(TB_DIR + folder_name + 'Single_Model.h5')

    x_train_filenames = sorted(glob.glob('./Data/Train_1500_Balanced_20x/Image/Images/*.png'))
    x_val_filenames = sorted(glob.glob('./Data/Valid_20x/Image/Images/*.png'))

    steps_per_epoch = np.ceil(len(x_train_filenames) / BATCH_SIZE).astype("int32")
    print('Steps for Training Epoch: ', steps_per_epoch)

    validation_steps = np.ceil(len(x_val_filenames) / BATCH_SIZE).astype("int32")
    print('Steps for Validation Epoch: ', validation_steps)

    history = parallel_model.fit_generator(TrainAugmentGenerator(SEED, BATCH_SIZE),
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data = ValAugmentGenerator(SEED, BATCH_SIZE),
                                  validation_steps = validation_steps,
                                  epochs=NUM_EPOCHS,
                                  callbacks=snapshot.get_callbacks()) 

    test_file_list = sorted(glob.glob('./Data/Test2_20x/*.tiff'))
    single_model = load_model(TB_DIR + folder_name + 'Single_Model.h5')
    final_model_parallel = multi_gpu_model(single_model, gpus = NUM_GPUS)
    final_model_parallel.summary()
    saved_weights_path = TB_DIR + folder_name + 'keras_swa.h5'
    print('Saved Weight: ', saved_weights_path)
    final_model_parallel.load_weights(saved_weights_path)
    run_gpu_inference(final_model_parallel, test_file_list, prediction_path)
