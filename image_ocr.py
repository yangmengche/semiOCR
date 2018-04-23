# -*- coding: utf-8 -*-
'''This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images.
'''

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--target", required=True, help="target type")
ap.add_argument("-p", "--predict", required=True, help="path to file of to predict")
# ap.add_argument("-w", "--weight", help="path to weight file")
# ap.add_argument("-tk", "--tkinter", action='store_true', help="setup tkinter")
# ap.add_argument("-a", "--answer", help="answer of prediction")
# ap.add_argument("-d", "--debug", action='store_true', help="Show debug message")
args = vars(ap.parse_args())

if('tkinter' in args):
    import matplotlib
    matplotlib.use('Agg')
# DEBUG = args['debug']
DEBUG = False

import os
import shutil
import itertools
import codecs
import re
import datetime
# import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks

#disable FutureWarning
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

root = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(root, 'image_ocr')
np.random.seed(55)

if DEBUG:
    debug_output = os.path.abspath('./debug')
    if not os.path.exists(debug_output):
        os.makedirs(debug_output)


# character classes and matching regex filter
regex = r'^[A-Z0-9]+$'
alphabet = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Network parameters
conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
minibatch_size = 32    

absolute_max_string_len = 16
words_per_epoch = 16000

random_range=13
# Target parameter
target={
    '19': {
        'TRUTH': '08430097KIA1',
        'TEMPLATE' : 'Image__2018-03-15__15-05-19.bmp',
        'LENGTH' : 12,
        'FONT_SIZE' : 75,
        'BOX' : {'x':60, 'y':282, 'w':700, 'h':140},
        'TARGET_SIZE' : {'w':224, 'h':224},
        'OFFSET' : {'x':17, 'y':32},
        'CHAR_GAP': 6,
        'WEIGHT': '19_490_98.h5'
    },
    '29': {
        'TRUTH': 'PKB79813029G6',
        'TEMPLATE' : 'Image__2018-03-15__14-57-29.bmp',
        'LENGTH' : 13,
        'FONT_SIZE' : 40,
        'BOX' : {'x':200, 'y':240, 'w':400, 'h':64},
        'TARGET_SIZE' : {'w':400, 'h':64},
        'OFFSET' : {'x':19, 'y':15},
        'CHAR_GAP': 5,
        'WEIGHT': '29_400_64.h5'
        # 'BOX' : {'x':200, 'y':150, 'w':400, 'h':300},
        # 'TARGET_SIZE' : {'w':400, 'h':300},
        # 'OFFSET' : {'x':19, 'y':103}
    },
    '54': {
        'TRUTH' : '8LOVO064MMC1',
        'TEMPLATE' : 'Image__2018-03-15__14-54-54.bmp',
        'LENGTH' : 12,
        'FONT_SIZE' : 75,
        'BOX' : {'x':60, 'y':239, 'w':700, 'h':140},
        'TARGET_SIZE' : {'w':490, 'h':98},
        'OFFSET' : {'x':13, 'y':33},
        'CHAR_GAP': 6,
        'WEIGHT': '54_490_98.h5'
    }
}
TRUTH = target[args['target']]['TRUTH']
TEMPLATE = os.path.join(root, 'res', target[args['target']]['TEMPLATE'])
LENGTH = target[args['target']]['LENGTH']
FONT_SIZE = target[args['target']]['FONT_SIZE']
BOX = target[args['target']]['BOX']
TARGET_SIZE = target[args['target']]['TARGET_SIZE']
OFFSET = target[args['target']]['OFFSET']
CHAR_GAP = target[args['target']]['CHAR_GAP']
WEIGHT_FILE = target[args['target']]['WEIGHT']
# WEIGHT_FILE = args['weight']

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import cv2

# resource for training
if not args['predict']:
    font = ImageFont.truetype("./res/SEMI_OCR_Font_document.ttf", FONT_SIZE)
    tm = Image.open(TEMPLATE)

def paint_text(text, w, h, box, ration=1, rotate=False, ud=False, multi_fonts=False):
    im = tm.copy()
    color = random.randrange(0, 128)
    x = random.randrange(box['x']-random_range, box['x']+random_range)
    y = random.randrange(box['y']-random_range, box['y']+random_range)
    # x = box['x']
    # y = box['y']
    # text = TRUTH
    tsize = font.getsize('A')
    chw = tsize[0] - CHAR_GAP
    # draw = ImageDraw.Draw(im)

    if box['w'] < tsize[0]:
        box['w'] = tsize[0]
    if box['h'] < tsize[1]:
        box['h'] = tsize[1]    
    tsize = (chw*len(text), tsize[1])
    cropBox = (box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])
    canvas = Image.new('L', (box['w'], box['h']), 255)
    draw = ImageDraw.Draw(canvas)

    # x_offset = (box['w'] - tsize[0])//2
    # y_offset = (box['h'] - tsize[1])//2
    for i, ch in enumerate(text):
        draw.text((int(OFFSET['x']+i*chw), OFFSET['y']), ch, font=font, fill=color)
    # dilate text to simulate laser dot

    if bool(random.getrandbits(1)):
        buf = np.array(canvas)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        buf = cv2.dilate(buf, kernel)
        canvas = Image.fromarray(buf)
    mask = ImageOps.invert(canvas)
    im.paste(canvas, box=(x, y), mask=mask)
    im_c = im.crop(cropBox)
    im_s = im_c.resize((w, h), Image.BILINEAR)
    if DEBUG:
        debug0 = os.path.join(debug_output, 'train_{0}_img0.png'.format(text))
        if not os.path.exists(debug0):
            canvas.save(os.path.join(debug_output, 'train_{0}_img0.png'.format(text)))
            im.save(os.path.join(debug_output, 'train_{0}_img1.png'.format(text)))
            im_c.save(os.path.join(debug_output, 'train_{0}_img2.png'.format(text)))
            im_s.save(os.path.join(debug_output, 'train_{0}_img3.png'.format(text)))
    a = np.array(im_s)
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    return a

def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = list(range(stop_ind))
    np.random.shuffle(a)
    a += list(range(stop_ind, len_val))
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('`shuffle_mats_or_lists` only supports '
                            'numpy.array and list objects.')
    return ret


# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(alphabet.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


# only a-z and space..probably not to difficult
# to expand to uppercase and symbols

def is_valid_str(in_str):
    search = re.compile(regex, re.UNICODE).search
    return bool(search(in_str))

def get_output_size():
    return len(alphabet) + 1

# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, minibatch_size,
                 img_w, img_h, box, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len
        self.box = box

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words
        for i in range(self.num_words):
            self.Y_len[i] = LENGTH
            code = random.sample(alphabet, LENGTH)
            code = ''.join(code)
            self.Y_data[i, 0:LENGTH] = text_to_labels(code)
            self.X_text.append(code)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0


    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 4, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h, box=self.box,
                                                  rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if 3 <= epoch < 6:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h, box=self.box,
                                                      rotate=False, ud=True, multi_fonts=False)
        elif 6 <= epoch < 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h, box=self.box,
                                                      rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h, box=self.box,
                                                      rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(32000, 12, 0.5)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('Out of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f\n'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        # self.model.save(os.path.join(self.output_dir, 'model%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()

def createModel(img_w, img_h):

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)    
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)

    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirectional GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))

    y_pred = Activation('softmax', name='softmax')(inner)
    model = Model(inputs=input_data, outputs=y_pred)
    if DEBUG:
        model.summary()

    labels = Input(name='the_labels', shape=[absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    return model, input_data, y_pred

def train(run_name, start_epoch, stop_epoch, img_w, img_h, box):
    # Input Parameters
    # img_h = 64
    
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split)) #3200


    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.mythic-ai.com/datasets/wordlists.tgz', untar=True))

    img_gen = TextImageGenerator(minibatch_size=minibatch_size,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words,
                                 absolute_max_string_len=absolute_max_string_len,
                                 box=box
                                 )
    model, input_data, y_pred = createModel(img_w, img_h)

    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
    # captures output of softmax so we can decode the output during visualization

    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    csv_logger = keras.callbacks.CSVLogger(os.path.join(OUTPUT_DIR, run_name, 'history.csv'))

    history = model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size, #(16000 - 3200) // 32
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // minibatch_size,
                        callbacks=[viz_cb, img_gen, csv_logger],
                        initial_epoch=start_epoch)

def loadData(files, w, h, box):
    cropBox = (box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h'])
    size = len(files)
    if K.image_data_format() == 'channels_first':
        X_data = np.ones([size, 1, w, h])
    else:
        X_data = np.ones([size, w, h, 1])
    for i, f in enumerate(files):
        im = Image.open(f)
        im = im.crop(cropBox)
        im = im.resize((w, h), Image.BILINEAR)
        if DEBUG:
            im.save(os.path.join(debug_output, 'predict.png'))
        a = np.array(im)
        a = a.astype(np.float32) / 255
        a = np.expand_dims(a, 0)
        if K.image_data_format() == 'channels_first':
            X_data[i, 0, 0:w, :] = a[0, :, :].T
        else:
            X_data[i, 0:w, :, 0] = a[0, :, :].T
    if DEBUG:
        print(X_data.shape)
    return X_data

def predict(weight, predict, img_w, img_h, box):
    text_X = loadData(predict, img_w, img_h, box)
    model, input_data, y_pred = createModel(img_w, img_h)
    try:
        model.load_weights(weight)
    except IOError:
        print('Can\'t load weight!')
        import sys
        sys.exit()
    test_func = K.function([input_data], [y_pred])
    pred_result = decode_batch(test_func, text_X)[0]

    print('predict = {0}'.format(pred_result))
def batch_predict(weight_folder, predict, img_w, img_h, box):
    model, input_data, y_pred = createModel(img_w, img_h)
    text_X = loadData(predict, img_w, img_h, box)
    i = 0
    while True:
        try:
            a = '{0}/weights{1:0>2d}.h5'.format(weight_folder, i)
            model.load_weights('{0}/weights{1:0>2d}.h5'.format(weight_folder, i))
        except IOError:
            break
        except ValueError as err:
            print('Image dimension error expect={0}x{1}'.format(TARGET_SIZE['w'], TARGET_SIZE['h']))
            break
        except Exception as err:
            print(err)
            break

        test_func = K.function([input_data], [y_pred])
        pred_result = decode_batch(test_func, text_X)[0]

        print('{0:0>3d} predict = {1}'.format(i, pred_result))
        # print('    truth   = {0}'.format(args['answer']))
        # if(pred_result == args['answer']):
        #     print('*********************')
        i +=1

if __name__ == '__main__':
    predict(WEIGHT_FILE, [args['predict']], TARGET_SIZE['w'], TARGET_SIZE['h'], BOX)
    # if args['predict']:
    #     if not args['weight']:
    #         import sys
    #         print('No weight loaded!')
    #         sys.exit()
    #     if args['answer']:
    #         batch_predict(args['weight'], [args['predict']], TARGET_SIZE['w'], TARGET_SIZE['h'], BOX)
    #     else:
    #         predict(args['weight'], [args['predict']], TARGET_SIZE['w'], TARGET_SIZE['h'], BOX)

    # else:
    #     run_name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    #     train(run_name, 0, 400, TARGET_SIZE['w'], TARGET_SIZE['h'], BOX)


    # increase to wider images and start at epoch 20. The learned weights are reloaded
    # train(run_name, 20, 25, 512)