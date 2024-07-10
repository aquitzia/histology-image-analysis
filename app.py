# Streamlit server info:
# lsb_release -a
# Debian GNU/Linux 11 (bullseye)
# import sys
# print(sys.version)
# Python version 3.11.9 (main, May 14 2024, 08:23:55) [GCC 10.2.1 20210110]

import os
import time
import random
from random import randrange
import requests
import json

import streamlit as st # 1.35.0
import pandas as pd # 2.2.2
# from utilities import predict, init_model

# Initialize state (once per app/user session)
if 'first_run' not in st.session_state:
    st.session_state.first_run = False

    # API Gateway URL format- https://{restapi_id}.execute-api.{region}.amazonaws.com/{stage_name}/{resource_path}
    # {restapi_id} for API Gateway REST API
    # {region} AWS region
    # {stage_name} deployment stage (e.g., prod, dev, etc.)
    # {resource_path} the endpoint that triggers the Lambda function
    st.session_state.S3_URL_ORIGINALS = "https://mhist-streamlit-app.s3.us-west-1.amazonaws.com/images/test-set/original/"
    st.session_state.LAMBDA_FUNCTION = 'https://msztnjekn7.execute-api.us-west-1.amazonaws.com/'
    st.session_state.FLASK_ENDPOINT = 'http://54.219.114.203:5050/' #13.52.243.246
    st.session_state.THUMB_DIR="thumb"
    
    # Metadata about the scans
    # 'name' : MHIST_<code>.png    # image codes are 3 letters long
    # 'label' = HP or SSA          # binary, categorical label
    # 'experts' = 0 through 7      # int
    st.session_state.test_df = pd.read_csv('testset_features.csv')
    st.session_state.train_df = pd.read_csv('trainset_features.csv')
    
    # # ONNX Runtime Session loads the model, then can be re-used
    # st.session_state.ort_session = init_model() # This wouldn't work well here with Lambda (session would be limited to 15 min, so would move it to Lambda invocation and add logic)

    # User selections
    st.session_state.random_menu = None
    st.session_state.preview_menu = None
    st.session_state.label = None
    st.session_state.selected_image_code = None
    st.session_state.selected_filename = None

    # print('\nSTATE INITIALIZED')

def get_df(label=None): # 'SSA', 'HP', or 'all'
    if label == 'all':
        return st.session_state.test_df
    return st.session_state.test_df.loc[st.session_state.test_df['label']==label, :].drop('label', axis=1)

def random_path(df):
        return random.choice(df['name'].tolist()) # issue: random.choice doesn't work with a dict-like on this server!

@st.cache_data # cache this container of images in case we use the same one again
def display_thumbnails(df):
    with st.container(height=400):  # Automatically scrollable container
        st.image(
            image = df['name'].apply(lambda name: os.path.join(st.session_state.THUMB_DIR, name)).to_list(),
            caption = df['code'].to_list(),
            use_column_width = "auto")

# Code is not None. If the code is valid, update the filename state variable.
# If the code in not valid, throw error and set filename to None (no selected image).
def validate_code():
    code = st.session_state.selected_image_code # for readability
    if code == '': # user hit Enter/Return without typing a code
        st.error("Please enter a code.") # Errors don't stop/exit/return from anything!
    elif len(code) != 3 or not code.isalpha():
        st.error('Image codes are 3 letters long. Please re-enter the code.')
    else:
        code = code.lower()
        filename = f'MHIST_{code}.png'
        #TODO: check whether the image label matches the selected label (SSA or HP)
        # print('filename', filename)
        test_df = st.session_state.test_df # for readability
        if test_df[test_df['name'] == filename].empty:
            st.error('Please check the image code for typos.')
        else:
            st.session_state.selected_filename = filename

def input_img_code():
    code = st.text_input("Enter an image code (3 letters)", placeholder='example: abc')
    return f'MHIST_{code}.png'


'### Histopathology Image Analysis'
'Version 0.0.1'
st.caption('The MHIST dataset contains images of tissue sections of colorectal polyps under a microscope. The model is trained on a common and clinically-significant (binary classification) task in gastrointestinal pathology.')
st.caption('There are two possible labels for each image:')
st.caption('- **HP**: hyperplastic polyp (benign), and')
st.caption('- **SSA**: sessile serrated adenoma (precancerous)')
st.caption('More information on the dataset:\nhttps://arxiv.org/abs/2101.12355')

'**Test the ML model**'
'You can test the model by selecting an image for real-time analysis (model inference). The model has not been trained on the tissue sections in the following set of images.'

# the first two menu options have the same submenu options
# the third menu option has no submenu
menu_options=['Select a random image', 'Preview the images', 'Enter an image code']
sub_menu_options=['hyperplastic polyp (benign)', 'sessile serrated adenoma (precancerous)', 'any tissue section']
labels = ['HP', 'SSA', 'all']

#TODO reset label, code, filename when menus are changed
selected = st.selectbox('Select', menu_options, index=None,
                        placeholder="Select an option", label_visibility="collapsed", key='menu')


# Print different statements based on the selected option
# Select a random image
if selected is not None and selected == menu_options[0]:
    selected_submenu = st.selectbox("Select", sub_menu_options, index=None,
                                    placeholder="Select an image category", label_visibility="collapsed", key = 'random_menu')
    if st.session_state.random_menu is not None:
        # print('selected random_menu:', st.session_state.random_menu)
        submenu_idx = sub_menu_options.index(st.session_state.random_menu)
        st.session_state.label = labels[submenu_idx] # convert menu text to model label (SSA, HP, or all categories)
        st.session_state.selected_filename = random_path(get_df(st.session_state.label))

# Preview the images
elif selected is not None and selected == menu_options[1]:
    selected_submenu = st.selectbox("Select", sub_menu_options, index=None,
                                    placeholder="Select an image category", label_visibility="collapsed", key = 'preview_menu')
    if st.session_state.preview_menu is not None:
        # print('selected preview_menu:', st.session_state.preview_menu)
        submenu_idx = sub_menu_options.index(st.session_state.preview_menu)
        st.session_state.label = labels[submenu_idx] # convert menu text to model label
        display_thumbnails(get_df(st.session_state.label))
        st.text_input("Enter an image code (3 letters)", placeholder='example: abc', key = 'selected_image_code')

# Enter an image code
elif selected is not None:    # Default: type in an image code
    st.text_input("Enter an image code (3 letters)", placeholder='example: abc', key = 'selected_image_code')


### MLP model ###
code = st.session_state.selected_image_code # for readability
if st.button('Analyze with the MLP model'):
    r = None # No http response (or request)
    if st.session_state.selected_filename is None and code is None:
        st.error("Please select an image, then click \"Analyze.\"")
    else:
        # We don't need to validate the image filename because users can't type in a filename (only a code)
        if code is not None: # user entered a code
            st.session_state.selected_filename = None # delete old filename
            validate_code() # if code is valid, update filename, else throw error
        # if code is None, keep filename

        if st.session_state.selected_filename is not None: # skip this if no file selected yet, or code is invalid
            # Display the image from S3 and get the pred from Lambda
            code = st.session_state.selected_filename[6:-4] # strip "MHIST_" and ".png" (slice first 6 chars and last 4)
            st.columns(3)[1].image(image=st.session_state.S3_URL_ORIGINALS+st.session_state.selected_filename, caption=code, use_column_width="auto") # fit image within the Streamlit app column width
            # print('s3 path:', S3_URL_ORIGINALS+st.session_state.selected_filename)

            messages = [
                'Sending a JSON request to the Lambda Function',
                'Running real-time inference',
                'Doing some dishes while I wait...',
                'It takes about 20 seconds for AWS Lambda to wake up and respond',
                'Does anyone know any good jokes?',
                'A.I. hasn\'t learned how to tell (funny) jokes yet... <crickets>',
                ]
            message = messages[randrange(0, len(messages))]
            with st.spinner(message):
                post_start = time.monotonic()
                r = requests.post(st.session_state.LAMBDA_FUNCTION+'predict', json={'image_filename':st.session_state.selected_filename})
                lambda_runtime = time.monotonic() - post_start
                if r is not None and r.status_code == 200:
                    # print(r.headers['Content-Type']) #application/json
                    # print('headers:\n', r.headers) #{'Date': 'Wed, 29 May 2024 03:50:21 GMT', 'Content-Type': 'application/json', 'Content-Length': '48', 'Connection': 'keep-alive', 'Apigw-Requestid': 'Yg7eoiIWSK4EJ8Q='}
                    # print(r.encoding) #utf-8
                    r_dict = json.loads(r.text)
                    print('AWS Lambda completed inference on image_filename', st.session_state.selected_filename, 'logit', r_dict['logit'])
                    pred_text = sub_menu_options[1] if r_dict['predicted_class'] == 'SSA' else sub_menu_options[0]
                    correct = r_dict['predicted_class'] == st.session_state.label
                    class_type = 'positive' if r_dict['predicted_class'] == 'SSA' else 'negative'

                    '***Results from the multi-layer perceptron (MLP) model running real-time inference on AWS Lambda***'
                    f"**Prediction:** {pred_text}, which is a **{str(correct).lower()} {class_type}**"
                    f"**Model's predicted probability:** {r_dict['probability']*100:.2f}%"
                    f"Preprocessed image in {r_dict['preprocess_time']:.2f} seconds"
                    f"Classified image in **{r_dict['inference_time']:.2f} seconds**"
                    f"Total: {lambda_runtime:.2f} seconds to send and receive the request from AWS Lambda"
                    # st.balloons()
                    st.caption('*The model was trained on a dataset of only 2,162 images, while the ImageNet dataset currently contains 14 million images.')

                else:
                    "Failed to trigger AWS Lambda function."
                    if r is not None:
                        f"Status code: {r.status_code}"
if st.button('Read about the MLP model and AWS Lambda system design'):
    '**Training Data**'
    counts = st.session_state.train_df['label'].value_counts()
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        'Sample Counts'
        st.bar_chart(counts, height=200)
    with col2:
        'Class Percentages'
        total = counts.sum()
        st.bar_chart(counts.apply(lambda x: x / total * 100), height=200)

    with open("mlp_info.md", "r") as f:
        mlp_file = f.read()
    mlp_file


##### ViT model ###
if st.button('Analyze with the ViT model'):
    r = None # No http response (or request)
    if st.session_state.selected_filename is None and code is None:
        st.error("Please select an image, then click \"Analyze.\"")
    else:
        # We don't need to validate the image filename because users can't type in a filename (only a code)
        if code is not None: # user entered a code
            st.session_state.selected_filename = None # delete old filename
            validate_code() # if code is valid, update filename, else throw error
        # if code is None, keep filename

        if st.session_state.selected_filename is not None: # skip this if no file selected yet, or code is invalid
            # print('validated filename', st.session_state.selected_filename)
            # Display the image from S3 and get the pred from Lambda
            code = st.session_state.selected_filename[6:-4] # strip "MHIST_" and ".png" (slice first 6 chars and last 4)
            st.columns(3)[1].image(image=st.session_state.S3_URL_ORIGINALS+st.session_state.selected_filename, caption=code, use_column_width="auto") # fit image within the Streamlit app column width
            # print('s3 path:', S3_URL_ORIGINALS+st.session_state.selected_filename)

            messages = [
                'Sending a JSON request to the Lambda Function',
                'Running real-time inference',
                'Doing some dishes while I wait...',
                # 'Does anyone feel like we\'re getting ghosted?',
                'It takes about 20 second for AWS Lambda to wake up and respond',
                'Does anyone know any good jokes?',
                'A.I. hasn\'t learned how to tell (funny) jokes yet... <crickets>',
                # 'Really almost done.'
                ]
            message = messages[randrange(0, len(messages))]
            with st.spinner(message):
                flask_start = time.monotonic()
                # print('POST to Flask app', st.session_state.FLASK_ENDPOINT+'predict', 'with filename:', st.session_state.selected_filename)
                r = requests.post(st.session_state.FLASK_ENDPOINT+'predict', json={'image_filename':st.session_state.selected_filename})
                # vit_results = predict(st.session_state.selected_filename)
                flask_runtime = time.monotonic() - flask_start
                if r is not None and r.status_code == 200:
                    # print('All HTTP headers:\n', r.headers) #{'Date': 'Wed, 29 May 2024 03:50:21 GMT', 'Content-Type': 'application/json', 'Content-Length': '48', 'Connection': 'keep-alive', 'Apigw-Requestid': 'Yg7eoiIWSK4EJ8Q='}
                    # print("HTTP status_code=200. Content-Type:", r.headers['Content-Type']) #application/json
                    vit_results = r.json() # <class 'dict'>
                    print('Flask app completed inference on image_filename', st.session_state.selected_filename, 'logit', vit_results['logit'])
                    '***Results from the Vision Transformer (ViT) model running real-time inference on EC2***'
                    pred_text = sub_menu_options[1] if vit_results['predicted_class'] == 'SSA' else sub_menu_options[0]
                    correct = vit_results['predicted_class'] == st.session_state.label
                    class_type = 'positive' if vit_results['predicted_class'] == 'SSA' else 'negative'
                    f"**Prediction:** {pred_text}, which is a **{str(correct).lower()} {class_type}**"
                    f"**Model's predicted probability:** {vit_results['probability']*100:.2f}%"
                    # f"Loaded model in {vit_results['model_load_time']:.2f} seconds"
                    f"Preprocessed image in {vit_results['preprocess_time']:.2f} seconds"
                    f"Classified image in **{vit_results['inference_time']:.2f} seconds**"
                    f"Total: {flask_runtime:.2f} seconds"

if st.button('Read about the ViT model and AWS EC2 system design'):
    with open("vit_info.md", "r") as f:
        vit_file = f.read()
    vit_file
# left_col, center_col, right_col = st.columns(3)
# with center_col:

