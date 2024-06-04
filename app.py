# Streamlit server info:
# lsb_release -a
# Debian GNU/Linux 11 (bullseye)
# import sys
# print(sys.version)
# Python version 3.11.9 (main, May 14 2024, 08:23:55) [GCC 10.2.1 20210110]

import os
import random
from random import randrange
import requests

# import boto3
import streamlit as st # 1.35.0
import pandas as pd
from PIL import Image

# session = boto3.Session(
#     aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
#     aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
# s3_client = session.client('s3')

# API Gateway URL format- https://{restapi_id}.execute-api.{region}.amazonaws.com/{stage_name}/{resource_path}
# {restapi_id} for API Gateway REST API
# {region} AWS region
# {stage_name} deployment stage (e.g., prod, dev, etc.)
# {resource_path} the endpoint that triggers the Lambda function
API_URL = 'https://ud4rhytiik.execute-api.us-west-1.amazonaws.com/'
THUMB_DIR="thumb"
S3_IMAGE_BUCKET = 'mhist-streamlit-app'
S3_URL_ORIGINALS = "https://mhist-streamlit-app.s3.us-west-1.amazonaws.com/images/test-set/original/"
S3_DIR_ORIGINALS="images/test-set/original/"
sample_image_path = 'MHIST_bge.png' # for testing purposes

# Metadata about the scans
# 'name' : MHIST_<code>.png    # image codes are 3 letters long
# 'label' = HP or SSA          # binary, categorical label
# 'experts' = 0 through 7      # int
test_df = pd.read_csv('testset_features.csv')

def center_image(path, caption):
    left_col, center_col, right_col = st.columns(3)
    with center_col:
        st.image(path, caption=caption, use_column_width="auto") # fit image within the Streamlit app column width

def center_st(st_object):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st_object

def get_df(label=None): # 'SSA', 'HP', or 'all'
    if label == 'all':
        return test_df
    return test_df.loc[test_df['label']==label, :].drop('label', axis=1)

def random_path(df):
        return random.choice(df['name'].tolist()) # issue: random.choice doesn't work with a dict-like on this server!

def display_all(df):
    with st.container(height=400):  # Automatically scrollable container
        st.image(
            image = df['name'].apply(lambda name: os.path.join(THUMB_DIR, name)).to_list(),
            caption = df['code'].to_list(),
            use_column_width = "auto")

def input_img_code():
    code = st.text_input("Enter an image code (3 letters)", placeholder='example: abc')
    return f'MHIST_{code}.png'

# Initialize state variables
if 'random menu' not in st.session_state:
    st.session_state['random menu'] = None
if 'preview menu' not in st.session_state:
    st.session_state['preview menu'] = None




'### Histopathology Image Analysis'
'Version 0.0.1'
st.caption('The MHIST dataset contains images of tissue sections of colorectal polyps under a microscope. The model is trained on a common and clinically-significant (binary classification) task in gastrointestinal pathology.')
st.caption('There are two possible labels for each image: HP: hyperplastic polyp (benign), and SSA: sessile serrated adenoma (precancerous).')
st.caption('More information on the dataset:\nhttps://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2101.12355')

'**Test the ML model**'
'You can test the model by selecting an image for real-time analysis (model inference). The model has never seen the tissue sections in the following set of images.'

# the first two menu options have the same submenu options
# the third menu option has no submenu
menu_options=['Select a random image', 'Preview the images', 'Enter an image code']
sub_menu_options=['hyperplastic polyp (benign)', 'sessile serrated adenoma (precancerous)', 'any tissue section image']

selected = st.selectbox('', menu_options, index=None,
                        placeholder="Select an option", label_visibility="collapsed", key='menu')
image_path = None


# Print different statements based on the selected option
# Select a random image
if selected is not None and selected == menu_options[0]:
    label = None
    selected_submenu = st.selectbox("", sub_menu_options, index=None,
                                    placeholder="Select an image category", label_visibility="collapsed", key = 'random menu')
    if selected_submenu is not None and selected_submenu == sub_menu_options[0]:
        label = 'HP'
    elif selected_submenu is not None and selected_submenu == sub_menu_options[1]:
        label = 'SSA'
    elif selected_submenu is not None and selected_submenu == sub_menu_options[2]:
        label = 'all'

    if label is not None:
        image_path = random_path(get_df(label))

# Preview the images
elif selected is not None and selected == menu_options[1]:
    label = None
    selected_submenu = st.selectbox("", sub_menu_options, index=None,
                                    placeholder="Select an image category", label_visibility="collapsed", key = 'preview menu')
    if selected_submenu is not None and selected_submenu == sub_menu_options[0]:
        label = 'HP'
    elif selected_submenu is not None and selected_submenu == sub_menu_options[1]:
        label = 'SSA'
    elif selected_submenu is not None and selected_submenu == sub_menu_options[2]:
        label = 'all'

    if label is not None:
        display_all(get_df(label))
        image_path = input_img_code()

# Enter an image code
elif selected is not None:    # Default: type in an image code
    image_path = input_img_code()

if image_path is not None and image_path[6:-4] != "": # to get the image code: strip "MHIST_*.png" (slice first 6 chars and last 4)
    st.write('selected image:', os.path.join(THUMB_DIR, image_path))

if st.button('Analyze'):
        r = None
        if image_path is None or image_path[6:-4] == "": # to get the image code: strip "MHIST_*.png" (slice first 6 chars and last 4)
            st.error("Please select an image, then click \"Analyze.\"")

        else:
            # Display the image from S3 and get the pred from Lambda
            code = image_path[6:-4] # strip "MHIST_*.png" (slice first 6 chars and last 4)
            st.columns(3)[1].image(image=S3_URL_ORIGINALS+image_path, caption=code, use_column_width="auto") # fit image within the Streamlit app column width
            # print('s3 path:', S3_URL_ORIGINALS+image_path)
            # s3_client.download_file(Bucket=S3_IMAGE_BUCKET, Key=S3_DIR_ORIGINALS+image_path, Filename=image_path)
            # st.image(S3_URL_ORIGINALS+image_path, caption=code, use_column_width="auto") # fit image within the Streamlit app column width

            messages = [
                'Sending a JSON request to the Lambda Function',
                'Running real-time inference',
                'This model is performing complex calculations with 125 million parameters!',
                'Doing some dishes while I wait...',
                'Waiting for a response',
                'Does anyone feel like we\'re getting ghosted?',
                'Almost done...',
                'Does you know any good jokes?',
                'AI hasn\'t learned how to tell (funny) jokes yet. <crickets>',
                'Really almost done.',
                'Almost really done.']
            message = messages[randrange(0, len(messages))]
            with st.spinner(message):
                'Get prediction from Lambda'
                # r = requests.post(API_URL+'predict', json={'Image':image_path})
                # print(r.headers['Content-Type']) #application/json
                # print('headers:\n', r.headers) #{'Date': 'Wed, 29 May 2024 03:50:21 GMT', 'Content-Type': 'application/json', 'Content-Length': '48', 'Connection': 'keep-alive', 'Apigw-Requestid': 'Yg7eoiIWSK4EJ8Q='}
                # print(r.encoding) #utf-8
 
            if r is not None and r.status_code == 200:
                # pred = r.json()[0]
                st.balloons()
                # label = pred['label']
                # score = pred['score']
                # '**Label:**', label
                # f'**Score:** {score:14%}'
                st.caption('*The model was trained on a dataset of only 2,162 images, while the ImageNet dataset currently contains 14 million images.')

            else:
                f"Failed to trigger AWS Lambda function. Status code: "#{r.status_code}"

# left_col, center_col, right_col = st.columns(3)
# with center_col:
if st.button('Get model info'):
    with open("README.md", "r") as f:
        mdf = f.read()
    mdf
