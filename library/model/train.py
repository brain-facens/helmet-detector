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

#------------------------------------------------------------------------------
# PARAMETERS.
IMG_SHAPE = (640, 640, 1)
IMG_WIDTH = IMG_SHAPE[1]
IMG_HEIGHT = IMG_SHAPE[0]

DATASET_PATH = "/home/brain-matheus/BRAIN-Project/helmet_detector/dataset"

BATCH = 16
EPOCH = 10

TRAIN_SPLIT = 0.8

CLASS = {
    1:"motocicleta",
    2:"sem-capacete",
    3:"com-capacete"
}

OPTIMIZER = tf.keras.optimizers.SGD(
    learning_rate=1e-3
)
#------------------------------------------------------------------------------

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
                "label":1,
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
                        __class[__label] = len(__class.keys()) + 1
                    
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

# This function will load the images and labels
# from 'images' and 'labels' folder inside the
# DATASET_PATH.
def load_dataset(_dataset:str, _train_split:float=0.9, _batch_size:int=8):

    assert type(_dataset)==str and type(_train_split)==float and type(_batch_size)==int
    assert os.path.isdir(_dataset)
    assert "images" in os.listdir(_dataset) and "labels" in os.listdir(_dataset)
    assert _train_split<=0.9 and _train_split>=0.5
    assert _batch_size>=3
  
    def load_json_files(_dir:str):
        __json_dir = os.listdir(os.path.join(_dir))
        __labels = []
        __bbox = []
        for json_file in __json_dir:
            try:
                with open(os.path.join(_dir, json_file), "r") as f:
                    js = json.load(f)
                    __labels.append([])
                    __bbox.append([])
                    for data in js["data"]:
                        __labels[-1].append(data["label"])
                        __bbox[-1].append(data["point"])
            except:
                pass
        return __labels, __bbox

    __images = keras.utils.image_dataset_from_directory(
        directory=os.path.join(_dataset, "images"),
        labels=None,
        color_mode="grayscale",
        batch_size=None,
        shuffle=False,
        image_size=(IMG_WIDTH, IMG_HEIGHT)
    )

    __label, __bbox = load_json_files(os.path.join(_dataset, "labels"))

    __label = tf.ragged.constant(__label).to_tensor()
    __bbox = tf.ragged.constant(__bbox).to_tensor()

    __labels = tf.data.Dataset.from_tensor_slices(__label)
    __bbox = tf.data.Dataset.from_tensor_slices(__bbox)

    __dataset = tf.data.Dataset.zip((__images, __labels, __bbox)).batch(_batch_size).shuffle(1000).prefetch(_batch_size // 3).cache()

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

if __name__=="__main__":
    # -- Dataset --
    try:
        logging.info("Loading Dataset..")
        TRAIN, VAL = load_dataset(DATASET_PATH, _train_split=TRAIN_SPLIT, _batch_size=BATCH)
    except:
        logging.warning("Could not load the Dataset: ".format(DATASET_PATH))
        exit(1)

    # -- Model --
    SSD_MOBILENETV2 = keras.Sequential([
        mobilenetv2.mobilenetv2_architecture(_input_shape=IMG_SHAPE)
    ])
    SSD_MOBILENETV2.summary()
