from cv2 import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.backend import clear_session
import numpy as np

def classify_one_image(image, image_size):
    """
    对单张图片进行分类
    :param image: 需分类图片
    :param kind_lists: 类型列表
    :return: null
    """
    # 读取图片，并作预处理
    image = cv2.resize(image, image_size)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    model = load_model('models/face_top_model.h5')
    # 将图片送入模型中预测
    result = model.predict(image)
    # 取出相似度最高的一项
    # 取出相似度最高的一项
    proba = np.max(result)
    return int(np.where(result == proba)[1]), proba, result
