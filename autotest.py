from matplotlib import pyplot
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
from keras.models import Input
from keras.models import Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from numpy.random import randint
from numpy import ones
from numpy import zeros
from numpy import load
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import cv2
import time
from keras.models import load_model

if __name__ == '__main__':
    while True:
        model = load_model('model_v1')
        source = cv2.imread('/home/arcsinx/ClothFlow/test1.jpg', 1)
        source = (source - 127.5) / 127.5

        source = source.reshape(1, 256, 256, 3)
        gen_image = model.predict(source)
        gen_image = gen_image.reshape(256, 256, 3)
        gen_image = (gen_image + 1) / 2.0

        #base = cv2.imread('/home/arcsinx/ClothFlow/real_test6.jpg', 1)
        #shape = base.shape
        #original = cv2.resize(gen_image, shape, interpolation = cv2.INTER_AREA)

        #cv2.imwrite('model_output.jpg', gen_image)

        fig, ax = pyplot.subplots()
        pyplot.axis('off')
        pyplot.imshow(gen_image)
        pyplot.savefig("pyplot.png")
        pyplot.close()
        time.sleep(120)