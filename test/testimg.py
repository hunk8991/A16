from numpy import expand_dims
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_image_pixels(filename, shape):
        image = load_img(filename, target_size=shape)
        image = img_to_array(image)
        image = expand_dims(image, 0)
        return image
def loadimg(image):
    #model = keras.models.load_model('vactor.h5')
    img = load_image_pixels(image,(150,150))
    a=model.predict(img)
    f = open('label.txt')
    label=[]
    score=[]
    for i in range(0,5):
        ss=f.readline()
        ss=ss.rstrip('\n')
        label.append(ss)
        score.append((a[0][i]+8)/16)
    f.close()
    json_str ='[{"'+str(label[0])+'":['+str(round(score[0],3))+']},{"'+str(label[1])+'":['+str(round(score[1],3))+']},{"'+str(label[2])+'":['+str(round(score[2],3))+']}]'
    python_data = json.loads(json_str)
    #print('object:',python_data)
    return score
for i in range(1,6):
    t = str(i) + ".jpg"
    model = keras.models.load_model('vactor.h5')
    print(loadimg(t))