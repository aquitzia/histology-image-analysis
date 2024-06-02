import os
import random
import streamlit as st
import requests
from random import randrange
import pandas as pd
from PIL import Image

# API Gateway URL format- https://{restapi_id}.execute-api.{region}.amazonaws.com/{stage_name}/{resource_path}
# {restapi_id} for API Gateway REST API
# {region} AWS region
# {stage_name} deployment stage (e.g., prod, dev, etc.)
# {resource_path} the endpoint that triggers the Lambda function
API_URL = 'https://ud4rhytiik.execute-api.us-west-1.amazonaws.com/'
THUMB_DIR="thumb"
S3_IMAGE_BUCKET="https://mhist-streamlit-app.s3.us-west-1.amazonaws.com/images/test-set/original/"
# aws s3 cp s3://mhist-streamlit-app/images/test-set/thumb/ thumb --recursive
sample_image_path = 'MHIST_bge.png' # for testing purposes

# Metadata about the scans
# 'name' : MHIST_<code>.png    # image codes are 3 letters long
# 'label' = HP or SSA          # binary, categorical label
# 'experts' = 0 through 7      # int
test_df = pd.read_csv('testset_info.csv')

'### Histopathology Image Analysis'
'Version 0.0.1'
st.caption('The MHIST dataset contains images of tissue sections of colorectal polyps under a microscope. The model is trained on a common and clinically-significant (binary classification) task in gastrointestinal pathology.')
st.caption('There are two possible labels for each image: HP: hyperplastic polyp (benign), and SSA: sessile serrated adenoma (precancerous).')
st.caption('More information on the dataset.: https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F2101.12355')

'### Test the ML model:'
'The trained ML model has not been trained on the following \'test set\' of 977 images.'
menu_options=['Precancerous SSA image', 'Benign HP image', 'Select an image by code:']
selected = st.selectbox('Select one of the following for analysis (model inference):', menu_options)
# Print different statements based on the selected option
if selected == menu_options[0]:    # SSA image
    "You selected Option 1."
    ssa_df = test_df[test_df['label']=='SSA', :].drop('label')
    menu_options_SSA = ['Select a random image', 'Preview all 360 SSA images']
    selected_submenu = st.selectbox("Select an option", menu_options_SSA)
    if selected == menu_options[1]: # Preview thumbnails
        'Preview all 360 SSA images'
        # for image_option in ssa_df['name']:
        image_option = sample_image_path
        code = image_option.strip("MHIST_.png")
        st.image(os.path.join(THUMB_DIR, image_option), caption=code, use_column_width="auto") # fit image within the Streamlit app column width
        image_path = image_option
    else: # Default behavior is random choice
        image_path = random.choice(ssa_df['name'])

elif selected == menu_options[1]: # HP image
    "You selected Option 2."
    menu_options_HP = ['Select a random image', 'Preview all 617  HP images']
    hp_df = test_df[test_df['label']=='HP', :].drop('label')
    selected_submenu = st.selectbox("Select an option", menu_options_HP)
    if selected == menu_options[1]: # Preview thumbnails
        'Preview all 360 SSA images'
        # for image_option in hp_df['name']:
        image_option = sample_image_path
        code = image_option.strip("MHIST_.png")
        st.image(os.path.join(THUMB_DIR, image_option), caption=code, use_column_width="auto") # fit image within the Streamlit app column width
        image_path = image_option
    else: # Default behavior is random choice
        image_path = random.choice(hp_df['name'])

else:    # default: type in an image code
    "You selected Option 3."
    code = st.text_input("3-letter image code", height=200)
    image_path = f'MHIST_{code}.png'

print('image_path', image_path)

if st.button('Analyze'):
        r = None
        if image_path == "":
            st.error("Please select an image, then click \"Analyze.\"")

        else:
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
            'Get full-sized image from S3'
            code = image_path.strip("MHIST_.png")
            st.image(S3_IMAGE_BUCKET+image_path, caption=code, use_column_width="auto") # fit within the Streamlit app column width
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

if st.button('Get model info'):
    with open("README.md", "r") as f:
        mdf = f.read()
    mdf
    # with st.spinner("Waiting for server response..."):
    #     'Get model info'
    #     # r = requests.post(API_URL+'info')
    
    # if r.status_code == 200:
    #     # st.write(r.content.decode('utf-8'))
    #     with open("README.md", "r") as f:
    #         mdf = f.read()
    #     mdf
    # else:
    #     f"Failed to trigger AWS Lambda function. Status code: {r.status_code}"