import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback
from tqdm.keras import TqdmCallback
import albumentations as A
import random
from skimage import transform, util
import cv2
import numpy as np


class BoxModel(keras.Model):
    """A model that gets image and mouse pos as inputs and returns a box

    For speed, the encoding part should be saved seperately.
    Therefore, the output tensor's shape should match the input shape
    of the box-detecting model.

    inputs
    ------
    image
        image of the cells
    pos
        mouse position

    output
    ------
    box
        [xmin, ymin, xmax, ymax], normalized to 0.0 ~ 1.0
    """
    def __init__(self, inputs, encoder_function, box_function):
        """
        arguments
        ---------
        inputs : dictionary of keras.Input
            keys:
                'image' : image input
                'pos' : mouse position input

        encoder_function : function 
            Function that uses keras functional API to make a encoder model
            Takes the image input and outputs encoded tensor of the image
        
        box_function : function
            Function that uses keras functional API to make a box model
            Takes the encoded image tensor and 'pos' as inputs.
            Outputs a tensor of [xmin, ymin, xmax, ymax]
        """
        super().__init__()
        image_input = inputs['image']
        pos_input = inputs['pos']

        image_output = encoder_function(image_input)
        self.encoder_model = keras.Model(inputs=image_input, 
                                         outputs=image_output)
        
        encoded_input = keras.Input(image_output.shape[1:])
        box_output = box_function(encoded_input, pos_input)
        self.box_model = keras.Model(inputs=[encoded_input, pos_input],
                                     outputs=box_output)
        
    def call(self, inputs, training=None):
        encoded = self.encoder_model(inputs['image'], training=training)
        return self.box_model([encoded, inputs['pos']], training=training)

class AugGenerator():
    """An iterable generator that makes data

    NOTE: Every img is reshaped to img_size
    NOTE: The position value is like pygame. (width, height),
          which does not match with common image order (height,width)

          Image input is expected to be the shape of (height, width),
          i.e. the transformation to match two is handled in here automatically

    NOTE: THE OUTPUT IMAGE WILL BE (WIDTH, HEIGHT)
          It is because pygame has shape (width, height)

    return
    ------
    X : tuple of (img, pos), dtype= np.uint8, np.int32
        img : np.array
            shape : (WIDTH, HEIGHT, 3)
        pos : tuple
            (WIDTH, HEIGHT)
    Y : tuple of (xmin, ymin, xmax, ymax) dtype= np.int32
    """
    def __init__(self, img, data, img_size):
        """ 
        arguments
        ---------
        img : list
            list of images, in the original size (height, width, 3)
        data : list of dict
            Each dict has :
                'image' : index of the image. The index should match with img
                'mask' : [rr, cc]
                'box' : [[xmin, ymin], [xmax,ymax]]
                'size' : the size of the image that the data was created with
                        IMPORTANT : (WIDTH, HEIGHT)
        img_size : tuple
            Desired output image size
            The axes will be swapped to match pygame.
            IMPORTANT : (WIDTH, HEIGHT)
        """
        self.img = img
        self.data = data
        self.n = len(data)
        self.output_size = img_size
        self.aug = A.Compose([
            A.OneOf([
                A.RandomGamma((40,200),p=1),
                A.RandomBrightness(limit=0.5, p=1),
                A.RandomContrast(limit=0.5,p=1),
                A.RGBShift(p=1),
            ], p=0.8),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            A.Resize(img_size[1], img_size[0]),
        ],
        bbox_params=A.BboxParams(format='albumentations', label_fields=['bbox_classes']),
        keypoint_params=A.KeypointParams(format='xy'),
        )

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        idx = random.randrange(0,self.n)
        datum = self.data[idx]
        image = self.img[datum['image']].copy()

        mask_idx = random.randrange(0,len(datum['mask'][0]))
        pos = (datum['mask'][0][mask_idx], datum['mask'][1][mask_idx])

        box_min = datum['box'][0]
        box_max = datum['box'][1]

        if np.any(np.not_equal(image.shape[:2], np.flip(datum['size']))):
            row_ratio = image.shape[0] / datum['size'][1]
            col_ratio = image.shape[1] / datum['size'][0]
            pos = np.multiply(pos, [col_ratio,row_ratio]).astype(np.int32)
            box_min = np.multiply(box_min, [col_ratio,row_ratio]).astype(np.int32)
            box_max = np.multiply(box_max, [col_ratio,row_ratio]).astype(np.int32)
        
        if np.all(np.greater(box_min,[0,0])):
            crop_min = np.random.randint([0,0],np.flip(box_min))
        else :
            crop_min = [0,0]

        if np.all(np.less(box_max, image.shape[:2])):
            crop_max = np.random.randint(np.flip(box_max), image.shape[:2])
        else :
            crop_max = image.shape[:2]

        cropped_image = image[crop_min[0]:crop_max[0],crop_min[1]:crop_max[1]]
        cropped_box_min = np.subtract(box_min, np.flip(crop_min))
        cropped_box_max = np.subtract(box_max, np.flip(crop_min))
        cropped_box = np.append(cropped_box_min, cropped_box_max)

        # Normalize
        width = cropped_image.shape[1]
        height = cropped_image.shape[0]
        size = [width, height, width, height]
        cropped_box = np.divide(cropped_box,size,dtype=np.float32)

        cropped_pos = np.subtract(pos, np.flip(crop_min))
        transformed = self.aug(
            image=cropped_image,
            bboxes=[cropped_box],
            bbox_classes=['cell'],
            keypoints=[cropped_pos],
        )
        t_img = np.array(transformed['image'],np.uint8).swapaxes(0,1)
        t_pos = np.array(transformed['keypoints'][0], np.int32)
        t_box = np.array(transformed['bboxes'][0], np.float32)
        X = {
            'image' : t_img,
            'pos' : t_pos
        }
        Y = t_box
        return X, Y


def create_train_dataset(img, data, img_size, batch_size):
    autotune = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_generator(
        AugGenerator(img, data, img_size),
        output_types=(
            {
                'image' : tf.uint8,
                'pos' : tf.int32,
            },
            tf.float32,
        ),
        output_shapes=(
            {
                'image' : tf.TensorShape([img_size[0],img_size[1],3]),
                'pos' : tf.TensorShape([2])
            },
            tf.TensorShape([4])
        )
    )
    dataset = dataset.shuffle(min(len(data),1000))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(autotune)
    dataset = dataset.repeat()

    return dataset

def create_val_data(img, data, img_size):
    """No modification to the image, including cropping and rotating
    """
    t_img = []
    T = A.Resize(img_size[1],img_size[0])
    for i in img:
        resized = T(image=i)['image'].astype(np.uint8)
        t_img.append(resized.swapaxes(0,1))
    X = []
    Y = []
    for datum in data:
        t_i = t_img[datum['image']].copy()
        mask_idx = random.randrange(0,len(datum['mask'][0]))

        i_size = np.array(datum['size'])
        pos = np.array([datum['mask'][0][mask_idx], datum['mask'][1][mask_idx]])
        t_pos = pos / i_size
        t_boxmin = datum['box'][0] / i_size
        t_boxmax = datum['box'][1] / i_size
        t_box = np.append(t_boxmin, t_boxmax)
        X.append({
            'image' : t_i,
            'pos' : t_pos.astype(np.float32)
        })
        Y.append(t_box.astype(np.float32))
    return X, Y


def get_model(encoder_f, box_f, img_size):
    """
    To get model only and load weights.
    """
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    inputs = {
        'image' : keras.Input((img_size[0],img_size[1],3)),
        'pos' : keras.Input((2))
    }
    test_model = BoxModel(inputs, encoder_f, box_f)
    test_model.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError(),
    )
    return test_model

def run_training(
        encoder_f,
        box_f, 
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        steps_per_epoch,
        img,
        data,
        img_size, 
        mixed_float = True,
        notebook = True,
    ):
    """
    val_data : (X_val, Y_val) tuple
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = {
        'image' : keras.Input((img_size[0],img_size[1],3)),
        'pos' : keras.Input((2))
    }
    mymodel = BoxModel(inputs, encoder_f, box_f)
    loss = keras.losses.MeanSquaredError()
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        # metrics=[
        #     'mse',
        # ]
    )

    logdir = 'logs/fit/' + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        profile_batch='3,5',
        update_freq='epoch'
    )
    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        save_weights_only=True,
        verbose=1
    )

    if notebook:
        tqdm_callback = TqdmNotebookCallback(metrics=['loss', 'binary_accuracy'],
                                            leave_inner=False)
    else:
        tqdm_callback = TqdmCallback()

    # if augment:
    train_ds = create_train_dataset(img, data, img_size, batch_size)
    mymodel.fit(
        x=train_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            tensorboard_callback,
            lr_callback,
            save_callback,
            tqdm_callback,
        ],
        verbose=0,
        # validation_data=val_data,
    )


    # else:
    #     mymodel.fit(
    #         x=X_train,
    #         y=Y_train,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         callbacks=[
    #             tensorboard_callback,
    #             lr_callback,
    #             save_callback,
    #             tqdm_callback,
    #         ],
    #         verbose=0,
    #         validation_data=val_data
    #     )

    print('Took {} seconds'.format(time.time()-st))

if __name__ == '__main__':
    import os
    import imageio as io
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import draw
    import cv2
    img_names = os.listdir('data/done')
    img = []
    img_name_dict = {}
    for idx, name in enumerate(img_names):
        img.append(io.imread('data/done/'+name))
        img_name_dict[name] = idx

    json_names = os.listdir('data/save')
    data = []
    for name in json_names:
        with open('data/save/'+name,'r') as j:
            data.extend(json.load(j))
    for datum in data :
        datum['image'] = img_name_dict[datum['image']]

    # fig = plt.figure()
    # d_idx = random.randrange(0,len(data)-5)
    # for i, d in enumerate(data[d_idx:d_idx+5]):
    #     image = img[d['image']].copy()
    #     image = cv2.resize(image, (1200,900), interpolation=cv2.INTER_LINEAR)
    #     mask = d['mask']
    #     m_idx = random.randrange(0,len(mask[0]))
    #     pos = (mask[0][m_idx], mask[1][m_idx])
    #     boxmin = d['box'][0]
    #     boxmax = d['box'][1]
    #     rr, cc = draw.disk((pos[1],pos[0]),5)
    #     image[rr, cc] = [0,255,0]
    #     rr, cc = draw.rectangle_perimeter((boxmin[1],boxmin[0]),(boxmax[1],boxmax[0]))
    #     image[rr,cc] = [255,0,0]
    #     image[mask[1],mask[0]] = [100,100,100]
    #     ax = fig.add_subplot(5,1,i+1)
    #     ax.imshow(image)
    # plt.show()

    # gen = AugGenerator(img, data, (400,400))
    # s = next(gen)

    ds = create_train_dataset(img, data, (400,300), 1)
    sample = ds.take(5).as_numpy_iterator()
    fig = plt.figure()
    for i, s in enumerate(sample):
        ax = fig.add_subplot(5,1,i+1)
        img = s[0]['image'][0].swapaxes(0,1)
        pos = s[0]['pos'][0]
        height, width = img.shape[:2]
        xmin,ymin,xmax,ymax = s[1][0] * np.array([width, height, width, height])
        rr, cc = draw.disk((pos[1],pos[0]),5, shape=img.shape[:2])
        img[rr,cc] = [0,255,0]
        rr, cc = draw.rectangle_perimeter((ymin,xmin),(ymax,xmax),shape=img.shape[:2])
        img[rr,cc] = [255,0,0]
        ax.imshow(img)
    plt.show()