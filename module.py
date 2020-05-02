import os
import pickle
import pandas as pd 
import tensorflow as tf
from cv2 import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import keras
import numpy as np
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from flask import Flask,request,render_template
import os
base_dir = "C:\\Users\\abouj\Desktop\\cats_and_dogs_filtered\\cats_and_dogs_filtered\\"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

print(tf.__version__)

# img2 =keras.preprocessing.image.load_img('C:\\Users\\abouj\\Desktop\\cats_and_dogs_filtered\\cat.jpg', target_size=(150,150))
# z2 = keras.preprocessing.image.img_to_array(img2)
# z2 = z2.reshape(1,150,150,3).astype('float')
# z2 /= 255



train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#nchofo image 9bal traitement
for img in os.listdir(train_dogs_dir): 
    img_array = cv2.imread(os.path.join(train_dogs_dir,img))  
    plt.imshow(img_array) 
    plt.show()  
    break 

#s3ib nkhadmo b alwan ankhadmo gha btadaroj dyal l grie
train_datagen = ImageDataGenerator(rescale = 1.0 / 255.)
test_datagen = ImageDataGenerator(rescale = 1.0 / 255.)

#antab9o dakchi 3la tsawr dyalna
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 10, class_mode = 'binary', target_size = (150, 150))
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size = 20, class_mode = 'binary', target_size = (150, 150))
#affichage dyal les class 1,0 dans binary cas
print(validation_generator.class_indices)
#nchofo image ba3d ma9adinaha traitement
x,y = train_generator.next()
for i in range(1,2):
    image = x[i]
    plt.imshow(image)
    plt.show()
    break

#hna ansawbo model CNN dyalna m3a 3 couche
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation = 'relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])


#compilation dyalou m3a tari9a d afficha<ge dyal progression
model.compile(optimizer=RMSprop(lr=0.001),loss='binary_crossentropy',metrics =['acc'])

#demarage dyal test o training
history = model.fit_generator(train_generator,validation_data=validation_generator,steps_per_epoch=100,epochs=20,validation_steps=50,verbose=1)
print('\nhistory dict:', history.history)
#save
model.save('model.h5')



# classes2 = model.predict_classes(z2)

# print(classes2)



# if classes2[0][0]==1:
#     print("kalb")
# else:
#     print("macha")



# app = Flask(__name__)

# @app.route("/",methods=['GET','POST'])
# def prod():
#     resultat=""
#     if(request.method == 'POST'):
#         file = request.files['file']
#         file.save(os.path.join("C:\\Users\\abouj\\Desktop\\cats_and_dogs_filtered\\upload", file.filename))
#         img =keras.preprocessing.image.load_img(os.path.join("C:\\Users\\abouj\\Desktop\\cats_and_dogs_filtered\\upload", file.filename), target_size=(150,150))
#         z = keras.preprocessing.image.img_to_array(img)
#         z = z.reshape(1,150,150,3).astype('float')
#         z /= 255
#         classes = model.predict_classes(z)
#         if classes[0][0]==1:
#             resultat="Dogg"
#         else:
#             resultat ="cat"

#         print(classes)
#         return render_template("file.html",massage=resultat)
#     return render_template("file.html")

# #demarage de server web flask    
# if __name__=='__main__':
# 	app.debug=True
# 	app.run(host='0.0.0.0',port=5000)


# train_datagen = ImageDataGenerator(
# rescale = 1./ 255,
# rotation_range = 40,
# width_shift_range = 0.2,
# height_shift_range = 0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest'
# )

