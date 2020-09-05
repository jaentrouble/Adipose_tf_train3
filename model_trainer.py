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


class AdiposeModel(keras.Model):
    def __init__(self, inputs, model_function):
        """
        Because of numerical stability, softmax layer should be
        taken out, and use it only when not training.

        arguments
        ---------
        inputs : keras.Input
        model_function 
            function that takes keras.Input and returns

        output
        ------
            output : Tensor
                tensor of logits
        """
        super().__init__()
        outputs = model_function(inputs)
        self.logits = keras.Model(inputs=inputs, outputs=outputs)
        self.logits.summary()
        
    def call(self, inputs, training=None):
        casted = tf.cast(inputs, tf.float32) / 255.0
        if training:
            return self.logits(inputs, training=training)
        return tf.math.sigmoid(self.logits(inputs, training=training))

class AugGenerator():
    """An iterable generator that makes data

    NOTE: Every img is reshaped to img_size
    NOTE: The position value is like pygame. (width, height),
          which does not match with common image order (height,width)

          Image input is expected to be the shape of (height, width),
          i.e. the transformation to match two is handled in here automatically

    return
    ------
    X : tuple of (img, pos), dtype= np.uint8, np.int32
        img : np.array 
        pos : tuple
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
            IMPORTANT : (HEIGHT, WIDTH)
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
            A.Resize(img_size[0], img_size[1]),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bbox_classes']),
        keypoint_params=A.KeypointParams(format='xy'),
        )
        self.resize = A.Resize(img_size[0], img_size[1])

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
        cropped_pos = np.subtract(pos, np.flip(crop_min))
        transformed = self.aug(
            image=cropped_image,
            bboxes=[cropped_box],
            bbox_classes=['cell'],
            keypoints=[cropped_pos],
        )
        t_img = np.array(transformed['image'],np.uint8)
        t_pos = np.array(transformed['keypoints'][0], np.int32)
        t_box = np.array(transformed['bboxes'][0], np.int32)
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
            tf.int32,
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


def get_model(model_f):
    """
    To get model only and load weights.
    """
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    inputs = keras.Input((200,200,3))
    test_model = AdiposeModel(inputs, model_f)
    test_model.compile(
        optimizer='adam',
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.1),
        ]
    )
    return test_model

def run_training(
        model_f, 
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        X_train, 
        Y_train, 
        val_data,
        mixed_float = True,
        notebook = True,
        augment = True,
    ):
    """
    val_data : (X_val, Y_val) tuple
    """
    if mixed_float:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = keras.Input((200,200,3))
    mymodel = AdiposeModel(inputs, model_f)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.1),
        ]
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

    if augment:
        train_ds = create_train_dataset(X_train, Y_train, batch_size)
        mymodel.fit(
            x=train_ds,
            epochs=epochs,
            steps_per_epoch=X_train.shape[0]//batch_size,
            callbacks=[
                tensorboard_callback,
                lr_callback,
                save_callback,
                tqdm_callback,
            ],
            verbose=0,
            validation_data=val_data,
        )


    else:
        mymodel.fit(
            x=X_train,
            y=Y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tensorboard_callback,
                lr_callback,
                save_callback,
                tqdm_callback,
            ],
            verbose=0,
            validation_data=val_data
        )

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

    ds = create_train_dataset(img, data, (300,400), 1)
    sample = ds.take(5).as_numpy_iterator()
    fig = plt.figure()
    for i, s in enumerate(sample):
        ax = fig.add_subplot(5,1,i+1)
        img, pos = s[0]
        img = img[0]
        pos = pos[0]
        xmin,ymin,xmax,ymax = s[1][0]
        rr, cc = draw.disk((pos[1],pos[0]),5, shape=img.shape[:2])
        img[rr,cc] = [0,255,0]
        rr, cc = draw.rectangle_perimeter((ymin,xmin),(ymax,xmax))
        img[rr,cc] = [255,0,0]
        ax.imshow(img)
    plt.show()