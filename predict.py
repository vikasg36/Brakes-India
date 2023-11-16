import cv2
import tensorflow as tf
import numpy as np
import io
from PIL import Image

MODEL_PATH="_crack_detection.h5"
model = tf.keras.models.load_model(MODEL_PATH)

#img= cv2.imread(,cv2.IMREAD_UNCHANGED)
#img = cv2.resize(img,(128,128))
def predict(img, model):
    print(img.shape)
    '''img=np.resize(img,(128,128,3))
    img.resize(128,128,3)'''
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x)
    images = np.vstack([x])
   # cv2.imshow("d",img)
    #cv2.waitKey(0)
    result = model.predict(images)
    return result
foo = Image.open("qq.JPG")
foo = foo.resize((128,640),Image.ANTIALIAS)
foo.save("cv17.jpg",quality=95)
img=cv2.imread("cv17.jpg")
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
th,img = cv2.threshold(sobelx,127,255,cv2.THRESH_BINARY_INV)

print(img.shape)
cv2.imwrite("work.jpg",img)
'''gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.zeros_like(img)
img[:,:,0] = gray
img[:,:,1] = gray
img[:,:,2] = gray'''
#img = Image.open(io.BytesIO(img))
prediction = predict(img, model)
print(prediction)
class_name = "NON_Scratch" if prediction[0] < 0.5 else "Scratch"
print( class_name)