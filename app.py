import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

                    #Feature list
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

st.title('Fashion Recommender System')
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def extract_features(img_path,model):
    img= image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result=model.predict(preprocessed_img).flatten()
    normalized_result= result/norm(result)
    return normalized_result
#steps
def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices





#file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
   if  save_uploaded_file(uploaded_file):
       display_image=Image.open(uploaded_file)
       st.image(display_image)
            #file hasbeen uploaded
       features = extract_features(os.path.join("uploads",uploaded_file.name),model)
       #st.text(features)
       indices=recommend(features,feature_list)
       col1,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)

       with col1:
           st.image(filenames[indices[0][0]])
       with col2:
           st.image(filenames[indices[0][1]])
       with col3:
           st.image(filenames[indices[0][2]])
       with col4:
           st.image(filenames[indices[0][3]])
       with col5:
           st.image(filenames[indices[0][4]])
       with col6:
           st.image(filenames[indices[0][5]])
       with col7:
           st.image(filenames[indices[0][6]])


   else:
       st.header("Some error occurred in file upload")
#load file -> feature extraction
#recommendation
#show