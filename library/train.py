import tensorflow as tf
import keras as K
import json
import os


IMG_SHAPE = (640, 640, 1)
IMG_WIDTH = IMG_SHAPE[1]
IMG_HEIGHT = IMG_SHAPE[0]


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
