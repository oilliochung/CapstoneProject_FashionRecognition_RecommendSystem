import streamlit as st
import cv2
import numpy as np
import pandas as pd
import glob
import random
import os
import altair as alt
import pydeck as pdk
from PIL import Image,ImageEnhance
import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import scipy
from IPython.display import display
from scipy import spatial
from io import BytesIO
from IPython.display import HTML 
from urllib.request import urlopen

import keras
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
import tensorflow as tf


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# page_bg_img = '''
# <style>
# body {
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-size: cover;
# }
# </style>
# '''
# st.markdown(page_bg_img , unsafe_allow_html=True)

# SET THE SIDE BOX 1 - Logo_BOX
st.sidebar.title("Welcome :smile:")

st.sidebar.markdown("---")


# SET THE SIDE BOX 2 - UPLOAD_BOX 
st.sidebar.subheader("User Image Upload")
user_input = st.sidebar.file_uploader(" ", type=("png", "jpg", "jpeg"))
if user_input is not None:
    img = Image.open(user_input)
#    st.sidebar.image(img, caption="User Input Image", width=200, height=300)

st.sidebar.markdown("---")


# SET THE SIDE BOX 3 - PRICE RANGE SLIDERBAR
price_range = st.sidebar.slider('Price Range Selection', min_value=0, max_value=5000, value=(0,1000))
# price_sorting = st.sidebar.checkbox("Price low to high")
No_of_product = st.sidebar.slider('No. of Products Selection', min_value=0, max_value=10, value=(5))

st.sidebar.markdown("---")


# SET THE SIDE BOX 4 - Video_BOX
st.sidebar.subheader("Adidas Offical Trailer")
v3 = st.sidebar.video("https://www.youtube.com/watch?v=4hAzZBVq2oE", start_time=2)
# user_input = st.sidebar.file_uploader(" ", type=("mp4"))
# if user_input is not None:
#     vid_file = open(user_input, "rb").read()
#     st.video(vid_file)

#____________________________________________________________________________________________________________________________________

# SET WEBPAGE TITLE
#row0_1, row0_2 = st.beta_columns((1.2, 2.8))

#with row0_1:
# logo = st.image("/Users/kurtischan/Desktop/Xccelerate/Group Project/Capstone Project/PowerPoint/App_logo5a.png", use_column_width=True) 

#

# SET OPENING SENTENCE
st.write(
    """
    Welcome to our "Buytifly" streamlit application! With innovative technology, we aid your eCommerce shopping journey with remarkable experiences, you can fly around with different eCommerce platform easliy to find your interested products with conveninces here!
    """)

st.sidebar.markdown("---")


# SET QUESTION
html_temp = """
<style>.font {font-size:18px !important;}</style>
<div>
<h2 style="color:brown; text-align:left; font-family:georgia,garamond,serif; font-style:italic">
<p class="font">Do you want to process?</p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)


# GET THE VECTOR FROM USER IMAGE
def get_feature_vector(img):
    image = cv2.resize(img, (224, 224),3)
    feature_vector = basemodel.predict(image.reshape( 1, 224, 224,3))
    return feature_vector

# CALCULATE THE SIMILARITY BETWEEN OUR IMAGE AND USER INPUT 
def calculate_similarity(vector1, vector2):
    return 1- scipy.spatial.distance.cosine(vector1, vector2)

# Load the pre-trained model we saved
basemodel=load_model('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vgg16.h5')

#Load DB
DB_shortT=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_shortT.csv')
DB_longT=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_longT.csv')
DB_poloT=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_poloT.csv')
DB_hoodies=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_hoodies.csv')
DB_shorts=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_shorts.csv')
DB_sportbra=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_sportbra.csv')
DB_jacket=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_jacket.csv')
DB_pants=pd.read_csv('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/DB_pants.csv')

# APPLY THE OBJECT DETECTION FUNCTION 
def yolo_v4(image, confidence_threshold=0.1, overlap_threshold=0.3):

	# Load model architecture
    net = cv2.dnn.readNetFromDarknet("yolov4-obj.cfg", "yolov4-obj_best.weights")
    output_layer_names = net.getLayerNames() # ['conv_0', 'bn_0', 'relu_0', 'conv_1', 'bn_1', .......]
    output_layer_names = [output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # get  unconnect layer name e.g.[[200], [227]] i[0]-1  get number inside 'out'  [200][0]=200  layersNames(199)= 'yolo_82'
    # convert image to array format
    image = np.array(image)

    # preprocess image (scale pixel between 0 to 1),(scale image size but not crop),(BGR to RBG)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # set image input to net 
    net.setInput(blob)
    # for calculate detect time 
    start = time.time()
    # net.forward(['conv_0', 'bn_0', ....]) get all output 
    layer_outputs = net.forward(output_layer_names)
    # for calculate detect time
    end = time.time()
    st.write("YOLOv4 took {:.3f} seconds".format(end - start))

    # SET EMPTY LIST 
    global class_IDs
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]

    # For each detected object, compute the box, find the score, ignore if below
    for output in layer_outputs:
        for detection in output: # 5 element of detection [center_x, center_y, width, height, confidences of bounding box(bbox)]
            scores = detection[5:] # confidence of all element
            classID = np.argmax(scores) # get the class index of max confidence
            confidence = scores[classID]# det max confidence according to class index
            if confidence > confidence_threshold: # determine current confidence is biggest than confidence_threshold or not
                # Scale the bounding box coordinates relative to the size of the image
                # Yolo return the bbox center(x, y)
                box = detection[0:4] * np.array([W, H, W, H]) # bbox width and height
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2)) # convert top left corner coordinates of the border
                boxes.append([x, y, int(width), int(height)]) # append top left coordinate , width and height to the list
                confidences.append(float(confidence)) # append confidence to the list
                class_IDs.append(classID) # append max confidence class index to the list

    # WRITE THE NAME OF DETECTED OBJECTS ABOVE IMAGE 
    global f
    f = open("classes.txt", "r")
    f = f.readlines()
    f= [line.rstrip('\n') for line in list(f)]

    global item_got    
    def item_got():
        return list(set([f[obj] for obj in class_IDs]))

    # APPLY NON-MAX SUPPRESSION TO IDENTIFY BEST BOUNDING BOX
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, overlap_threshold)
    UDACITY_LABELS = {
        0: 'Long-T',
        1: 'Hoodie',
        2: 'Pants',
        3: 'Polo-T',
        4: 'Short-T',
        5: 'Shorts',
        6: 'SportBra',
        8: 'Adidas'
    }
    xmin, xmax, ymin, ymax, labels = [], [], [], [], []

    if len(indices) > 0:
        for i in indices.flatten():    
            label = UDACITY_LABELS.get(class_IDs[i], None) 
            if label is None:
                continue
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
            labels.append(label)
    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})

    # ADD A LAYER ON TOP OF A DETECTED OBJECT - COLOR(r,g,b)
    LABEL_COLORS = {
        "Adidas": [255, 0, 0], #red (long-T)
        "Pants": [0, 255, 0], #green (hoodie)  
        "Hoodie": [0, 0, 255], #blue (Pants)
        "Jacket": [244,164,96],  #orange(polo-t)
        "Shorts": [255, 0, 255],  #purple (short-t)
        "SportBra": [255, 255, 0], #yellow(shorts)
        "Short-T": [120, 100, 50],  #brown (jacket)
        "Long-T": [255, 255, 255],  #white (sportsbar)
        "Polo-T": [0,255,255]   #light blue (adidas)
    }
    image_with_boxes = image.astype(np.float64)
    for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
        image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # DISPLAY THE FINAL IMAGE 
    st.image(image_with_boxes.astype(np.uint8), width=300, height=500)

    return item_got

# DEFINE FUNCTION FOR RECOMMENDATIONS
def cal_sim(image):

    #USER UPLOAD PART
    original = np.array(image)
    org2=cv2.resize(original, (800, 800)) 

    #FILL THE EDGE
    org2[0:80,]=0
    org2[720:800,]=0
    org2[:,0:80]=0
    org2[:,720:800]=0
    f1 = get_feature_vector(org2)

    # MAKE LIST FOR cal_sim
    v_shortT=[]
    v_longT=[]
    v_poloT=[]
    v_hoodies=[]
    v_shorts=[]
    v_pants=[]
    v_sportbra=[]
    v_jacket=[]

    #LOAD CLASS FROM YOLO
    yoloclass = item_got()

    #  CALCULATE SIMILARITY AND APPEND TO LIST
    if 'Adidas' in yoloclass:
        for i in yoloclass:
            if i=='Short-T':
                for i in range(1842):
                    v_shortT.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/shortT_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='Long-T':
                for i in range(529):
                    v_longT.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/longT_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='Polo-T':
                for i in range(385):
                    v_poloT.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/poloT_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='Hoodie':
                for i in range(1287):
                    v_hoodies.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/hoodies_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='Shorts':
                for i in range(1359):
                    v_shorts.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/shorts_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='Pants':
                for i in range(1450):
                    v_pants.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/pants_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='SportBra':
                for i in range(54):
                    v_sportbra.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/sportbra_f2_'+str(i)+'.txt', delimiter=','))*100)
            elif i=='Jacket':
                for i in range(614):
                    v_jacket.append(calculate_similarity(f1[0], np.loadtxt('/Users/jeffchan/Documents/Documents/Xccelerate/CoWork./Project-4/Deploy_test001/Gather/vectors/jacket_f2_'+str(i)+'.txt', delimiter=','))*100)
        else: print('Please check our recommendations.')

    def get_redult_from_DB(price_range, No_of_product):
    #APPEND SIMILARITY SCORE TO DB 
        if v_shortT==[] :
            DB_shortT['Sim']=0
        else:
            DB_shortT['Sim']=v_shortT
            for i in range(0,No_of_product):
                # sort by sim 
                DB_shortT.sort_values(by=['Sim'], ascending=False, inplace=True)
                # set price range and open image from csv 
                resp = urlopen(DB_shortT[(DB_shortT['price_hkd']>=min(price_range)) & (DB_shortT['price_hkd']<=max(price_range))]['product_image'][i])
                resp_img = Image.open(resp)
                # show image
                st.image(resp_img, caption=None, width=200, use_column_width=False)
                # show product detail
                st.write(DB_shortT[(DB_shortT['price_hkd']>=min(price_range)) & (DB_shortT['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                # show price
                st.write('HKD$ ' + str(int(DB_shortT[(DB_shortT['price_hkd']>=min(price_range)) & (DB_shortT['price_hkd']<=max(price_range))]['price_hkd'][i])))
                # show product URL
                st.write(DB_shortT[(DB_shortT['price_hkd']>=min(price_range)) & (DB_shortT['price_hkd']<=max(price_range))]['product_url'][i])

        if v_longT==[] :
            DB_longT['Sim']=0
        else:
            DB_longT['Sim']=v_longT
            for i in range(0,No_of_product):
                # sort by sim 
                DB_longT.sort_values(by=['Sim'], ascending=False, inplace=True)
                # set price range and open image from csv 
                resp = urlopen(DB_longT[(DB_longT['price_hkd']>=min(price_range)) & (DB_longT['price_hkd']<=max(price_range))]['product_image'][i])
                resp_img = Image.open(resp)
                # show image
                st.image(resp_img, caption=None, width=200, use_column_width=False)
                # show product detail
                st.write(DB_longT[(DB_longT['price_hkd']>=min(price_range)) & (DB_longT['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                # show price
                st.write('HKD$ ' + str(int(DB_longT[(DB_longT['price_hkd']>=min(price_range)) & (DB_longT['price_hkd']<=max(price_range))]['price_hkd'][i])))
                # show product URL
                st.write(DB_longT[(DB_longT['price_hkd']>=min(price_range)) & (DB_longT['price_hkd']<=max(price_range))]['product_url'][i])


        if v_poloT==[] :
            DB_poloT['Sim']=0
        else:
            DB_poloT['Sim']=v_poloT
            for i in range(0,No_of_product + 1):
                try:
                    # sort by sim 
                    DB_poloT.sort_values(by=['Sim'], ascending=False, inplace=True)
                    # set price range and open image from csv 
                    resp = urlopen(DB_poloT[(DB_poloT['price_hkd']>=min(price_range)) & (DB_poloT['price_hkd']<=max(price_range))]['product_image'][i])
                    resp_img = Image.open(resp)
                    # show image
                    st.image(resp_img, caption=None, width=200, use_column_width=False)
                    # show product detail
                    st.write(DB_poloT[(DB_poloT['price_hkd']>=min(price_range)) & (DB_poloT['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                    # show price
                    st.write('HKD$ ' + str(int(DB_poloT[(DB_poloT['price_hkd']>=min(price_range)) & (DB_poloT['price_hkd']<=max(price_range))]['price_hkd'][i])))
                    # show product URL
                    st.write(DB_poloT[(DB_poloT['price_hkd']>=min(price_range)) & (DB_poloT['price_hkd']<=max(price_range))]['product_url'][i])
                except:
                    pass

        if v_hoodies==[] :
            DB_hoodies['Sim']=0
        else:
            DB_hoodies['Sim']=v_hoodies
            for i in range(0,No_of_product + 1):
                try:
                    # sort by sim 
                    DB_hoodies.sort_values(by=['Sim'], ascending=False, inplace=True)
                    # set price range and open image from csv 
                    resp = urlopen(DB_hoodies[(DB_hoodies['price_hkd']>=min(price_range)) & (DB_hoodies['price_hkd']<=max(price_range))]['product_image'][i])
                    resp_img = Image.open(resp)
                    # show image
                    st.image(resp_img, caption=None, width=200, use_column_width=False)
                    # show product detail
                    st.write(DB_hoodies[(DB_hoodies['price_hkd']>=min(price_range)) & (DB_hoodies['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                    # show price
                    st.write('HKD$ ' + str(int(DB_hoodies[(DB_hoodies['price_hkd']>=min(price_range)) & (DB_hoodies['price_hkd']<=max(price_range))]['price_hkd'][i])))
                    # show product URL
                    st.write(DB_hoodies[(DB_hoodies['price_hkd']>=min(price_range)) & (DB_hoodies['price_hkd']<=max(price_range))]['product_url'][i])
                except:
                    pass

        if v_shorts==[] :
            DB_shorts['Sim']=0
        else:
            DB_shorts['Sim']=v_shorts
            for i in range(0,No_of_product):
                # sort by sim 
                DB_shorts.sort_values(by=['Sim'], ascending=False, inplace=True)
                # set price range and open image from csv 
                resp = urlopen(DB_shorts[(DB_shorts['price_hkd']>=min(price_range)) & (DB_shorts['price_hkd']<=max(price_range))]['product_image'][i])
                resp_img = Image.open(resp)
                # show image
                st.image(resp_img, caption=None, width=200, use_column_width=False)
                # show product detail
                st.write(DB_shorts[(DB_shorts['price_hkd']>=min(price_range)) & (DB_shorts['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                # show price
                st.write('HKD$ ' + str(int(DB_shorts[(DB_shorts['price_hkd']>=min(price_range)) & (DB_shorts['price_hkd']<=max(price_range))]['price_hkd'][i])))
                # show product URL
                st.write(DB_shorts[(DB_shorts['price_hkd']>=min(price_range)) & (DB_shorts['price_hkd']<=max(price_range))]['product_url'][i])

        if v_sportbra==[] :
            DB_sportbra['Sim']=0
        else:
            DB_sportbra['Sim']=v_sportbra
            for i in range(0,No_of_product):
                # sort by sim 
                DB_sportbra.sort_values(by=['Sim'], ascending=False, inplace=True)
                # set price range and open image from csv 
                resp = urlopen(DB_sportbra[(DB_sportbra['price_hkd']>=min(price_range)) & (DB_sportbra['price_hkd']<=max(price_range))]['product_image'][i])
                resp_img = Image.open(resp)
                # show image
                st.image(resp_img, caption=None, width=200, use_column_width=False)
                # show product detail
                st.write(DB_sportbra[(DB_sportbra['price_hkd']>=min(price_range)) & (DB_sportbra['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                # show price
                st.write('HKD$ ' + str(int(DB_sportbra[(DB_sportbra['price_hkd']>=min(price_range)) & (DB_sportbra['price_hkd']<=max(price_range))]['price_hkd'][i])))
                # show product URL
                st.write(DB_sportbra[(DB_sportbra['price_hkd']>=min(price_range)) & (DB_sportbra['price_hkd']<=max(price_range))]['product_url'][i])

        if v_jacket==[] :
            DB_jacket['Sim']=0
        else:
            DB_jacket['Sim']=v_jacket
            try:
                for i in range(0,No_of_product):
                    # sort by sim 
                    DB_jacket.sort_values(by=['Sim'], ascending=False, inplace=True)
                    # set price range and open image from csv 
                    resp = urlopen(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['product_image'][i])
                    resp_img = Image.open(resp)
                    # show image
                    st.image(resp_img, caption=None, width=200, use_column_width=False)
                    # show product detail
                    st.write(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                    # show price
                    st.write('HKD$ ' + str(int(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['price_hkd'][i])))
                    # show product URL
                    st.write(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['product_url'][i])
            except KeyError:
                for i in range(0, No_of_product +1):
                    try:
                        # sort by sim 
                        DB_jacket.sort_values(by=['Sim'], ascending=False, inplace=True)
                        # set price range and open image from csv 
                        resp = urlopen(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['product_image'][i])
                        resp_img = Image.open(resp)
                        # show image
                        st.image(resp_img, caption=None, width=200, use_column_width=False)
                        # show product detail
                        st.write(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                        # show price
                        st.write('HKD$ ' + str(int(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['price_hkd'][i])))
                        # show product URL
                        st.write(DB_jacket[(DB_jacket['price_hkd']>=min(price_range)) & (DB_jacket['price_hkd']<=max(price_range))]['product_url'][i])
                    except: 
                        pass

        if v_pants==[] :
            DB_pants['Sim']=0
        else:
            DB_pants['Sim']=v_pants
            for i in range(0,No_of_product + 3):
                try:
                    # sort by sim 
                    DB_pants.sort_values(by=['Sim'], ascending=False, inplace=True)
                    # set price range and open image from csv 
                    resp = urlopen(DB_pants[(DB_pants['price_hkd']>=min(price_range)) & (DB_pants['price_hkd']<=max(price_range))]['product_image'][i])
                    resp_img = Image.open(resp)
                    # show image
                    st.image(resp_img, caption=None, width=200, use_column_width=False)
                    # show product detail
                    st.write(DB_pants[(DB_pants['price_hkd']>=min(price_range)) & (DB_pants['price_hkd']<=max(price_range))]['product_detail'][i].lower().replace("adidasadidas", "adidas"))
                    # show price
                    st.write('HKD$ ' + str(int(DB_pants[(DB_pants['price_hkd']>=min(price_range)) & (DB_pants['price_hkd']<=max(price_range))]['price_hkd'][i])))
                    # show product URL
                    st.write(DB_pants[(DB_pants['price_hkd']>=min(price_range)) & (DB_pants['price_hkd']<=max(price_range))]['product_url'][i])
                except:
                    pass

    get_redult_from_DB(price_range, No_of_product)

# SET A BUTTON TO PROCESS ALL FUNCTION
if st.button("Click Me!"):
    if user_input is None:
        html_temp = """
        <style>.font {font-size:18px !important;}</style>
        <div> 
        <h2 style="color:blue; text-align:left; font-family:georgia,garamond,serif">
        <p class="font">PLEASE UPLOAD YOUR PHOTO ON THE LEFT!</p>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
    else:
    #   'Wait a moment, We are detecting your image and make some recommendation for you ...'
        # Add a placeholder
        latest_iteration = st.empty()
        progress_bar = st.progress(0)
        for i in range(100):
        # Update the progress bar with each iteration.
            latest_iteration.text(f'Loading your result {i+1}% ...')
            progress_bar.progress(i+1)
            time.sleep(0.04)
        st.balloons()

        # LAYING OUT THE 1ST SECTION OF THE APP
        row1_1, row1_2 = st.beta_columns((2, 2))

        with row1_1:
            st.write(
            """
            **Detected Objects** :mag::mag::mag:
            """)
            yolo_v4(img)

        with row1_2:
            st.write(
            """
            **Details** :memo::memo::memo:
            """)
            if 8 not in set(class_IDs):  
                st.write("Brand: " + 'Non Adidas')
            else:
                st.write("Brand: " + "Adidas")

            for i in set(class_IDs):  
                if i !=8:
                    st.write("Items: " + f[i])
        
        st.markdown("---")


        # LAYING OUT THE 2ND SECTION OF THE APP
        st.write(
        """
        ##
        **Recommendations **
        """)
        cal_sim(img)

        # with row2_2:
        #     st.write(
        #     """
        #     ##
        #     """)

        st.markdown("---")


        # LAYING OUT THE 3RD SECTION OF THE APP
        # row3_1, row3_2, row3_3, row3_4, row3_5, row3_6, row3_7, row3_8, row3_9, row3_10 = st.beta_columns((2,2))
        
        # for i in range(1, No_of_product):
        #     with row3_(i):
        #         st.write(
        #         """
        #         ##
        #         **Recommendations 2**
        #         """)
                
        #     with row3_2:
        #         st.write(
        #         """
        #         ##
                # """)

    
