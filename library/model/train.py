#==============================================================================
# RESTARING SESSION.
import tensorflow as tf
import keras

keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            device=gpus[0],
            logical_devices=[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)] # Limiting GPU memory to 1024MB
        )
    except RuntimeError as e:
        logging.warning(e)
        exit(1)
#==============================================================================
from ssd_mobilenetv2 import mobilenetv2, ssd
import logging
import keras
import json
import os


# PARAMETERS.
IMG_SHAPE = (640, 640, 1)
IMG_WIDTH = IMG_SHAPE[1]
IMG_HEIGHT = IMG_SHAPE[0]

DATASET_PATH = "/home/brain-matheus/BRAIN-Project/helmet_detector/dataset"

BATCH = 16
EPOCH = 10

CLASS = {
    0:"motocicleta",
    1:"sem-capacete",
    2:"com-capacete"
}

OPTIMIZER = tf.keras.optimizers.SGD(
    learning_rate=1e-3
)


#==============================================================================
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
                __len = len(js["shapes"])
                for i,data in enumerate(js["shapes"]):
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

                    if i < __len - 1:
                        new_file.write("{" + __file_format + "\n},\n")
                    else:
                        new_file.write("{" + __file_format + "\n}\n")
                new_file.write("]\n")
                new_file.write("}")

def _preprocessing_images():
    __ppi = keras.Sequential(name="preprocessing_model")
    __ppi.add(keras.layers.Resizing(640,640))
    __ppi.add(keras.layers.Rescaling(1./255))
    __ppi.add(tf.keras.layers.RandomBrightness(factor=0.3))
    return __ppi

# This function will load the images and labels
# from 'images' and 'labels' folder inside the
# DATASET_PATH.
def load_dataset(_train_split:float=0.9, _batch_size:int=8):

    __preprocessing_img = _preprocessing_images()

    @tf.function
    def load_images(_images):
        return __preprocessing_img(tf.cast(tf.io.decode_jpeg(tf.io.read_file(_images), channels=1), dtype=tf.float32))

    def load_labels(_labels):
        with open(_labels.numpy(), "r", encoding="utf-8") as file:
            js = json.load(file)
            __all_labels = []
            __all_points = []
            for data in js["data"]:
                __label = data["label"]
                __point = data["point"]
                __all_labels.append(__label)
                __all_points.append(__point)
            return __all_labels, __all_points

    __images = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, "images", "*.jpg"), shuffle=False)
    __labels = tf.data.Dataset.list_files(os.path.join(DATASET_PATH, "labels", "*.json"), shuffle=False)

    # Preparing data.
    __images = __images.map(load_images)
    __labels = __labels.map(lambda label: tf.py_function(load_labels, [label], [tf.int32, tf.float32]))

    __dataset = tf.data.Dataset.zip((__images, __labels)).padded_batch(_batch_size, padded_shapes=(([None, None, 1]),([None],[None,4])), padding_values=((0.0), (-1, -1.0))).shuffle(1000).prefetch(_batch_size // 3)

    _train_split = int(len(__dataset) * _train_split)

    __train = __dataset.take(_train_split)
    __val = __dataset.skip(_train_split).take(-1)

    return __train, __val
#==============================================================================
@tf.function
def train_step():
    pass

@tf.function
def val_step():
    pass

@tf.function
def bbox_loss():
    pass

@tf.function
def cls_loss():
    pass
#==============================================================================

# -- Dataset --
try:
    logging.info("Loading Dataset..")
    TRAIN, VAL = load_dataset(_batch_size=BATCH)
except:
    logging.warning("Could not load the Dataset: ".format(DATASET_PATH))
    exit(1)

# -- Model --
SSD_MOBILENETV2 = keras.Sequential([
    mobilenetv2.mobilenetv2_architecture(_input_shape=IMG_SHAPE)
])
SSD_MOBILENETV2.summary()
