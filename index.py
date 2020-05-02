from flask import Flask,request,render_template
import os
import tensorflow as tf 
import cv2
import numpy as np
from tensorflow import keras

model = tf.keras.models.load_model('model.h5')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
app = Flask(__name__)
@app.route("/",methods=['GET','POST'])
def prod():
    resultat=[]
    if(request.method == 'POST'):
        if request.files['file']:
            file = request.files['file']
            file.save(os.path.join("C:\\Users\\abouj\\Desktop\\cats_and_dogs_filtered\\static", file.filename))
            img =keras.preprocessing.image.load_img(os.path.join("C:\\Users\\abouj\\Desktop\\cats_and_dogs_filtered\\static", file.filename), target_size=(150,150))
            z = keras.preprocessing.image.img_to_array(img)
            z = z.reshape(1,150,150,3).astype('float')
            z /= 255
            resultat.append(file.filename)
            classes = model.predict_classes(z)
            if classes[0][0]==1:
                resultat.append("Chien")
            else:
                resultat.append("chat")

            print(classes)
            return render_template("file.html",message=resultat)
        else:
            resultat.append("oops ! tu a pas choisir un image")
            return render_template("file.html",message=resultat)
    return render_template("file.html",message=[])


if __name__=='__main__':
	app.debug=True
	app.run(host='0.0.0.0',port=5000)


    