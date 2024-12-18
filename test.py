import pickle
import tensorflow
import numpy as np
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

                        #DATA SET THAT WE HAVE

feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

                            # MODEL IMPORT HERE

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

                #FEATURE EXTRACTION OF NEWLY GIVEN  IMAGE

img= image.load_img('D:\Major Project\pythonProject2\Sample images\girloutfit.jpg',target_size=(224,224))
img_array = image.img_to_array(img)                 #converting the image in array
expanded_img_array = np.expand_dims(img_array,axis=0)  #batch conversion
preprocessed_img = preprocess_input(expanded_img_array)
result=model.predict(preprocessed_img).flatten()
normalized_result= result/norm(result)

                        #To get the closet 5 figures
neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])
print(indices)

for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)
