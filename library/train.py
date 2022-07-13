import tensorflow as tf
import keras as K
import json
import os


IMG_SHAPE = (640, 640, 1)
IMG_WIDTH = IMG_SHAPE[1]
IMG_HEIGHT = IMG_SHAPE[0]

DATASET_PATH = "/home/brain-matheus/BRAIN-Project/helmet_detector/dataset"


# This function will create a new label file but
# only with useful information from original
# label file.
def _create_json_label(label_dir:str, to_dir:str):
    """
    {
        "image":"image_name.jpg",
        "data":[
            {
                "label":0,
                "point":[
                    0.1,
                    0.2,
                    0.8,
                    0.5
                ]
            }
        ]
    }
    """
    
    __label_name = os.listdir(label_dir)
    __class = {}
    
    for lbl in __label_name:
        __label = os.path.join(label_dir, lbl)
        with open(__label, "r") as file:
            js = json.load(file)
            with open(os.path.join(to_dir, lbl), "w") as new_file:
                new_file.write("{\n")
                new_file.write(f'"image":"{lbl}",\n')
                new_file.write('"data":[\n')
                for data in js["shapes"]:
                    __point = data["points"]
                    __label = data["label"]

                    __x1 = __point[0][0] / IMG_WIDTH
                    __y1 = __point[0][1] / IMG_HEIGHT
                    __x2 = __point[1][0] / IMG_WIDTH
                    __y2 = __point[1][1] / IMG_HEIGHT

                    if __label not in __class.keys():
                        __class[__label] = len(__class.keys())
                    
                    __label = __class[__label]

                    __file_format = """
                    "label":{},\n
                    "point":[\n
                    {},\n
                    {},\n
                    {},\n
                    {}\n
                    ]
                    """.format(__label, __x1, __y1, __x2, __y2)

                    new_file.write("{" + __file_format + "\n}\n")
                new_file.write("]\n")
                new_file.write("}")


# This function will load the images and labels
# from 'images' and 'labels' folder inside the
# DATASET_PATH.
def load_dataset(_train_split:float=0.9, _batch_size:int=8):

    @tf.function
    def load_images(_images):
        return tf.io.decode_jpeg(tf.io.read_file(_images), channels=1)

    def load_labels(_labels):
        with open(_labels.numpy(), "r", encoding="utf-8") as file:
            js = json.load(file)
            print(js)
            return _labels
    
    __images = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, "images"), shuffle=False)
    __labels = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, "labels"), shuffle=False)

    # Preparing data.
    __images = __images.map(load_images)
    __labels = __labels.map(lambda label: tf.py_function(load_labels, [label], [tf.uint8, tf.float16]))

    __dataset = tf.data.Dataset.zip((__images, __labels)).batch(_batch_size).shuffle(1000)

    return __dataset

DATA = load_dataset()