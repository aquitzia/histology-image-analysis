import os
import time

# To run model
import onnxruntime # 1.18.0
from onnxruntime import InferenceSession

# To download and preprocess an image for real-time inference
import urllib.request
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
import numpy as np # 1.26.4
# Mean and std values are calculated from the training data, to normalize the colors (per channel):
TRAIN_MEAN = [0.738, 0.649, 0.775]
TRAIN_STD =  [0.197, 0.244, 0.17]

import boto3 # 1.34.127
import botocore # handles exceptions
S3_BUCKET = "mhist-streamlit-app"
S3_ORIGINALS_DIR = "images/test-set/original/"
S3_THUMBS_DIR = "images/test-set/thumb/" # To download 977 thumbnails from S3
LOCAL_THUMBS_DIR = "thumb/"
EFS_ACCESS_POINT = '/mnt/efs' # root directory is mounted here
VIT_MODEL_PATH = 'onnx_artifacts/mhist_vit_f1_dynamo_model.onnx' # in EFS
MLFLOW_MODEL_PATH = 'onnx_artifacts' #327.63MB

from flask import Flask, request, jsonify
app = Flask(__name__)
model_path = os.environ.get('VIT_MODEL_PATH')

def init_model():
    print('EFS_ACCESS_POINT contents:', os.listdir(EFS_ACCESS_POINT)) # EFS Access Point has access to the contents of /mhist-lambda
    print('EFS_ACCESS_POINT/onnx_artifacts/ contents:', os.listdir(EFS_ACCESS_POINT+'/onnx_artifacts/')) # EFS Access Point has access to the contents of /mhist-lambda
    onnx_path = os.path.join(EFS_ACCESS_POINT, VIT_MODEL_PATH)
    print('\nLoading model from', os.path.abspath(onnx_path))
    try:
        start_onnx_session = time.monotonic()
        # session_options = onnxruntime.SessionOptions()
        # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        load_time = time.monotonic() - start_onnx_session
    except ort_session.OrtException as e:
        print(f"ONNX Runtime error: {e}")
    except ValueError as e:
        print(f"Value error: {e}") # incorrect input shapes or types
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return ort_session, load_time

# Pre-load the model once (in 5.25 seconds), for quick inference every time
ORT_SESSION, LOAD_TIME = init_model()


def s3_get_object(image_filename):
    try:
        # # Use session class for debugging
        # session = boto3.Session()
        # print("Region:", session.region_name)
        # print("Profile:", session.profile_name)
        # print("Credentials:", session.get_credentials())
        # s3 = session.client('s3')

        image_s3key = os.path.join(S3_ORIGINALS_DIR, image_filename)
        # print('Loading', S3_BUCKET, image_s3key)
        s3 = boto3.client('s3')
        file_obj = s3.get_object(Bucket=S3_BUCKET, Key=image_s3key)
        return file_obj
    except botocore.exceptions.ClientError as e:
        print(f"An AWS error occurred: {e}")
    except botocore.exceptions.NoCredentialsError:
        print("Credentials not available")
    except botocore.exceptions.PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {e}")

        ort_session = init_model()
        # ort_session = InferenceSession(os.path.abspath(onnx_path))#, providers=['CPUExecutionProvider'])


def standardize_image(np_image):
    # Convert lists to numpy
    np_mean = np.array(TRAIN_MEAN, dtype=np.float32).reshape(3, 1, 1) # np_mean.shape (3, 1, 1)
    # print('np_mean shape', np_mean.shape)
    np_std = np.array(TRAIN_STD, dtype=np.float32).reshape(3, 1, 1) # np_std.shape (3, 1, 1)

    # Normalize
    # Operations are performed element-wise using NumPy broadcasting
    np_image = (np_image - np_mean) / np_std
    return np_image


# Model expects the input to be ndarray (150528,), dtype torch.float32
# Images are normalized to range [0., 1.] and standardized by channel
def preprocess(image_filename):
    # Download image (png file) as bytes from S3
    file_obj = s3_get_object(image_filename)
    image_bytes = BytesIO(file_obj['Body'].read())

    # Convert bytes (buffer) to 3-channels, then to ndarray
    # We could do this without PIL (using only NumPy)
    pil_image = Image.open(image_bytes).convert('RGB') # pil_image.size (224, 224) with 3 channels
    # print('pil_image dimensions', pil_image.size)
    np_image = np.array(pil_image, dtype=np.float32) # np_image shape (224, 224, 3) dtype float32
    # print('np_image shape', np_image.shape, 'dtype', np_image.dtype)
    transposed_np = np.transpose(np_image, (2, 0, 1)) # shape (3, 224, 224) max pixel value = 255.
    # print('transposed_np shape', transposed_np.shape, 'max pixel value =', np.max(transposed_np))
    normalized_np = transposed_np / 255.0 # normalize range to [0., 1.]
    # print('normalized_np image shape', normalized_np.shape, 'max pixel value =', np.max(normalized_np))
    standardized_np = standardize_image(normalized_np) # normalize color-channels
    # print('standardized_np image shape', standardized_np.shape)
    return np.expand_dims(standardized_np, axis=0)# .unsqueeze(0) doesn't work with np


def sigmoid(np_outs):
    np_outs = np.clip(np_outs, -50, 50) # prevent np.exp overflow for large values
    return 1 / (1 + np.exp(-np_outs))


@app.route('/predict', methods=['POST'])
def predict(image_filename): # image_url <class '_io.BytesIO'>
    data = request.get_json()
    image_filename = data.get('image_filename')
    if not image_filename:
        return jsonify({"error": "image_filename is required"}), 400
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    start_preprocess = time.monotonic()
    preprocessed_image = preprocess(image_filename)
    preprocess_time = time.monotonic() - start_preprocess

    # Run inference with optimized ONNX model
    # It uses 3.3 GB CPU memory, and 1.4 GB space (for artifacts)
    # Model is already loaded in ORT_SESSION and LOAD_TIME was noted
    start_inference = time.monotonic()
    input_name = ORT_SESSION.get_inputs()[0].name
    ort_outs = ORT_SESSION.run(None, {input_name: preprocessed_image}) # output: [array([-1.2028292], dtype=float32)]
    # result = ORT_SESSION.infer(preprocessed_image)
    inference_time = time.monotonic() - start_inference

    logit = ort_outs[0].item() # <class 'numpy.ndarray'> shape (1,) dtype=float32
    positive_prob = sigmoid(logit).item()
    pred = positive_prob > 0.5
    inference_info = {
        'logit': logit, # check image_filename MHIST_aah.png logit=2.16929292678833
        'predicted_class': 'SSA' if pred else 'HP',
        'probability': positive_prob if pred else 1-positive_prob,
        'model_load_time': LOAD_TIME,
        'preprocess_time': preprocess_time,
        'inference_time': inference_time-preprocess_time,
        }
    return jsonify(inference_info, 200)


# Debug
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
