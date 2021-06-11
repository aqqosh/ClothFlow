import os
import cv2
import requests
import random
import numpy as np
import albumentations as A
from random import randrange

#TODO code review
#TODO dir path in functions
#TODO change functions names
#TODO убрать папки в первых функциях кода
source_dirs = ['raw_data/raw/sources', 'raw_data/raw/redressed']
cropped_dirs = ['raw_data/raw/sources_cropped', 'raw_data/raw/redressed_cropped']
bg_dir = 'raw_data/raw/backgrounds'
bg_size = (512, 512)
bg_samples_from_image = 5
images_amount = 1
input_dir1 = 'raw_data/raw/sources_cropped'
output_dir1 = 'raw_data/raw/sources_figures'
input_dir2 = 'raw_data/raw/redressed_cropped'
output_dir2 = 'raw_data/raw/redressed_figures'
images_dir1 = 'raw_data/raw/sources_cropped'
figures_dir1 = 'raw_data/raw/sources_figures'
humans_dir1 = 'raw_data/raw/sources_humans'

#request fgure mask
url_to_figure = 'http://195.201.163.22:5005/lip_figure'


HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#source_image
#redressed_image


def data_load(source_dir, redressed_dir):
    #TODO add exp len = len
    source_dir_list = os.listdir(source_dir)
    redressed_dir_list = os.listdir(redressed_dir)

    for index in range(0, len(source_dir_list)):
        source_name = source_dir_list[index]
        redressed_name = redressed_dir_list[index]
        source = cv2.imread('source_dir/' + source_name, 1)
        redressed = cv2.imread('redressed_dir' + redressed_name, 1)

        source, redressed = detect(source, redressed)
        cv2.imwrite('reDress/crop' + str(index) + '.jpg', source)
        #TODO СРОЧНО УБРАТЬ ЭТУ ХНЮ
        crop = open('reDress/crop' + str(index) + '.jpg', 'rb')
        headers = {
            'cache-control': "no-cache",
        }
        data = {
            'some_input_name': 'some input value',
            'another_input_name': 'another input value',
        }
        files = {
            'file': crop
        }
        r = requests.post(url_to_figure, headers=headers, data=data, files=files)
        mask = r.content
        file = open('reDress/masks/' + str(index) + '.jpg', "wb")
        file.write(mask)
        file.close()
        #mask = define_figure_mask(url_to_figure, source)
        #cv2.imwrite('mask.jpg', mask)


def detect(image1, image2):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(image1, winStride = (4, 4), padding = (8, 8), scale = 1.03)
    for x, y, w, h in bounding_box_cordinates: #TODO add check: bbox exists
        cropped_image1 = image1[y:y + h, x:x + w]
        cropped_image2 = image2[y:y + h, x:x + w]

    return cropped_image1, cropped_image2


def define_figure_mask(url_to_figure, source):
    # IT DOESNT WORK
    #TODO bring out of function request body
    #TODO normal variables names
    buffer = source.data
    dtype = source.dtype
    shape = source.shape
    strides = source.strides
    
    source = source.tobytes()
    print(source)
    headers = {
    'cache-control': "no-cache",
    }
    data = {
        'some_input_name': 'some input value',
        'another_input_name': 'another input value',
    }

    files = {'file': source}
    #print(files['file'])
    r = requests.post(url_to_figure, headers=headers, data=data, files=files)
    data = r.content
    mask = np.frombuffer(data, dtype=dtype)
    mask = np.lib.stride_tricks.as_strided(mask, shape, strides)
    
    return mask


def crop_by_figure(source, mask):
    images_names = os.listdir(images_dir)
    for idx, image_name in enumerate(images_names):
        src1 = cv2.imread(images_dir + '/' + image_name, 1)
        figure = cv2.imread(figures_dir + '/' + image_name, 1)
        #figure = cv2.bitwise_not(figure)

        mask_out = cv2.subtract(mask, source)
        mask_out = cv2.subtract(mask, mask_out)

        cv2.imwrite('mask_out.jpg', mask_out)
        print("wrt")
    return

def prepare_human_samples(human_dir, human_size, human_samples_from_image):
    #TODO it better works with face recognition
    human_samples_from_image = 25
    human_names = os.listdir(human_dir)
    #height, width = (400, 200)
    for idx, human_name in enumerate(human_names):
        human_image = cv2.imread(human_dir + '/' + human_name, 1)
        img_h, img_w, _ = human_image.shape
        for i in range(0, human_samples_from_image):
            height = randrange(300, 600)
            width = randrange(250, img_w)
            r = randrange(0, img_w - width)
            c = randrange(0, img_h - height - 100)
            sample = human_image[r:r+height,c:c+width]
            cv2.imwrite('raw_data/raw/human_cropped/' + str(idx) + str(i) + '.jpg', sample)
    return

def crop_zoom_pad(image_source, image_redressed):
    color = [0, 0, 0]
    height_bias = randrange(300, 700)
    width_bias = randrange(0, 100)
    subimage_source = image_source[0:height_bias, 0:image_source.shape[1] - width_bias]
    subimage_redressed = image_source[0:height_bias, 0:image_source.shape[1] - width_bias]

    if height_bias < 450:
        coef = random.randint(20, 30)/10
    else:
        coef = random.randint(10, 20)/10

    width = round(subimage_source.shape[0] * coef)
    height = round(subimage_source.shape[1] * coef)
    
    dim = (height, width)
    resized_source = cv2.resize(subimage_source, dim, interpolation = cv2.INTER_AREA)
    resized_redressed = cv2.resize(subimage_redressed, dim, interpolation = cv2.INTER_AREA)

    pad_bias = randrange(0, 300)
    resized_source = cv2.copyMakeBorder(resized_source.copy(),0,0,pad_bias,0,cv2.BORDER_CONSTANT,value=color)
    resized_redressed = cv2.copyMakeBorder(resized_redressed.copy(),0,0,pad_bias,0,cv2.BORDER_CONSTANT,value=color)

    return resized_source, resized_redressed

def apply_augmentation(image_source, image_redressed):
    transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.CLAHE(p=1),
        A.GaussNoise(p=0.2),
        A.ElasticTransform(p=0.2),
        A.RGBShift(p=0.2),
        A.CoarseDropout(p=0.6)
    ],
    additional_targets={'redressed': 'image'}
    )
    #random.seed(42)
    transformed = transform(image=source, redressed=redressed)

    return transformed['image'], transformed['redressed']

def resize(image_source, image_redressed):
    dim = (256, 256)
    source = cv2.resize(image_source, dim, interpolation = cv2.INTER_AREA)
    redressed = cv2.resize(image_redressed, dim, interpolation = cv2.INTER_AREA)

    return source, redressed










"""
def data_crop(source_dirs, images_amount):
    max_index = images_amount
    files = []
    for directory in source_dirs:
        files.append(os.listdir(directory)) #TODO check jpg
    print(files)
    for index in range(0, max_index):
        source_image = cv2.imread('raw_data/raw/sources/' + files[0][index], 1)
        redressed_image = cv2.imread('raw_data/raw/redressed/' + files[1][index], 1)
        detect(source_image, redressed_image, index)

    return

def prepare_bg_from_center(bg_dir, bg_size):
    #TODO change bg.shapes to x, y, w, h
    #TODO defined size
    print("Deprecated: use prepare_bg_samples instead")
    bgs_names = os.listdir(bg_dir)
    for idx, human_name in enumerate(bgs_names):
        human_image = cv2.imread(bg_dir + '/' + human_name, 1)
        print(human_image.shape)
        center = (human_image.shape[0]/2, human_image.shape[1]/2)
        x = center[1] - human_image.shape[1]/2
        y = center[0] - human_image.shape[0]/2
        crop_human_image = human_image[int(y):int(y+human_image.shape[0]), int(x):int(x+human_image.shape[1]/2)]

        cv2.imwrite('raw_data/raw/backgrounds_cropped/' + str(idx) + '.jpg', crop_human_image)
    return
"""

def prepare_bg_samples(bg_dir, bg_size, bg_samples_from_image):
    bgs_names = os.listdir(bg_dir)
    height, width = bg_size
    for idx, human_name in enumerate(bgs_names):
        human_image = cv2.imread(bg_dir + '/' + human_name, 1)
        img_h, img_w, _ = human_image.shape
        for i in range(0, bg_samples_from_image):
            r = randrange(0, img_w - width)
            c = randrange(0, img_h - height)
            sample = human_image[r:r+height,c:c+width]
            cv2.imwrite('raw_data/raw/backgrounds_cropped/' + str(idx) + str(i) + '.jpg', sample)
    return





#### PIPELINE NOTES
data_load('raw_data/raw/sources', 'raw_data/raw/redressed')

prepare_bg_samples(bg_dir, bg_size, bg_samples_from_image)
define_figure_mask(url_to_figure, input_dir1, output_dir1)
define_figure_mask(url_to_figure, input_dir2, output_dir2)
#change color mask for figure
crop_by_figure(images_dir1, figures_dir1, humans_dir1)
#crop_by_figure(images_dir2, figures_dir2, humans_dir2)
prepare_human_samples(humans_dir1, (300,300), 25)
# for human samples
source = cv2.imread('/home/arcsinx/reDress/raw_data/raw/sources/0088.jpg')
redressed = cv2.imread('/home/arcsinx/reDress/raw_data/raw/sources/0088.jpg')
source, redressed = crop_zoom_pad(source, redressed)
source, redressed = apply_augmentation(source, redressed)
source, redressed = resize(source, redressed)
# save result
cv2.imwrite('test_s' + str(0) + '.jpg', source)
cv2.imwrite('test_r' + str(0) + '.jpg', redressed)