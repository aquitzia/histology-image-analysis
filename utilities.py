import os
import time
import json

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
# S3_URL_ORIGINALS = "https://mhist-streamlit-app.s3.us-west-1.amazonaws.com/images/test-set/original/"
S3_BUCKET = "mhist-streamlit-app"
S3_ORIGINALS_DIR = "images/test-set/original/"
S3_THUMBS_DIR = "images/test-set/thumb/" # To download 977 thumbnails from S3
LOCAL_THUMBS_DIR = "thumb/"
EFS_ACCESS_POINT = '/mnt/efs/mhist-lambda' # root directory is mounted here
MODEL_PATH = 'onnx_artifacts/mhist_vit_f1_dynamo_model.onnx' # in EFS

# To download artifacts from MLflow (remotely):
import mlflow
MLFLOW_SERVER="http://ec2-3-101-21-63.us-west-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_SERVER)
MLFLOW_RUN = '36bf8fc8ca8b4d8c8788a9b6e74c6099' # run_name = shivering-worm-829
MLFLOW_MODEL_PATH = 'onnx_artifacts' #327.63MB
LOCAL_MODEL_DIR = os.getcwd() # Download MLflow dir to this dest

def init_model():
    # print('EFS_ACCESS_POINT/onnx_artifacts/ contents:', os.listdir(EFS_ACCESS_POINT+'/onnx_artifacts/')) # EFS Access Point has access to the contents of /mhist-lambda
    onnx_path = os.path.join(EFS_ACCESS_POINT, MODEL_PATH)
    print('\nLoading model from', os.path.abspath(onnx_path))
    try:
        start_onnx_session = time.monotonic()
        onnx_dir = os.path.join(MLFLOW_MODEL_PATH, 'mhist_vit_f1_dynamo_model.onnx')
        # session_options = onnxruntime.SessionOptions()
        # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])
        load_time = time.monotonic() - start_onnx_session
    except ort.OrtException as e:
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


def predict(image_filename): # image_url <class '_io.BytesIO'>
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
    # print('logit', logit)
    positive_prob = sigmoid(logit).item()
    pred = positive_prob > 0.5
    # print('ONNX pred =', pred)
    inference_info = {#json.dumps({
        'logit': logit, # check image_filename MHIST_aah.png logit=2.16929292678833
        'predicted_class': 'SSA' if pred else 'HP',
        'probability': positive_prob if pred else 1-positive_prob,
        'model_load_time': LOAD_TIME,
        'preprocess_time': preprocess_time,
        'inference_time': inference_time-preprocess_time,
        }
    return inference_info


# Download ONNX model from MLflow server (called manually)
def download_latest_model():
    print('MLflow Tracking URI:', mlflow.get_tracking_uri())
    run = mlflow.get_run(MLFLOW_RUN)
    print('MLflow runName =', run.data.tags['mlflow.runName'], 'run_id =', run.info.run_id)
    print('Current working directory:', LOCAL_MODEL_DIR)
    # tags = run.data.tags
    # metrics = run.data.metrics

    start_time = time.monotonic()
    mlflow_files = mlflow.artifacts.download_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH, dst_path=EFS_ACCESS_POINT)
    downloaded_time = time.monotonic()
    print('Downloaded model files:\n', mlflow_files)#os.listdir(LOCAL_MODEL_DIR))
    print(f'Downloaded model in {(downloaded_time-start_time):.2f}s')
    # mlflow_files=mlflow.artifacts.list_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH)


# Download files from a directory (prefix) in S3
def s3_download_files(s3_bucket, s3_dir, local_dest_dir):
    s3 = boto3.client('s3')
    print('Downloading from S3 bucket', s3_bucket, 'and filtering by prefix', s3_dir)
    objects = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_dir) # Filter by thumbs prefix

    for i, obj in enumerate(objects['Contents']):
        file_name = os.path.basename(obj['Key'])
        if i == 0:
            print('First filename:', file_name, 'First object key:', obj['Key'])
        local_path = os.path.join(local_dest_dir, file_name)
        s3.download_file(s3_bucket, obj['Key'], local_path)
    print('Downloaded', i+1, 'files to', local_dest_dir)


def get_PILs(filename):
    s3 = boto3.client('s3')

    # Copied from above (it was correct when I used it in Streamlit)
    image_url = S3_URL_ORIGINALS+filename
    print('URL Lib download: \n image_url', image_url)
    with urllib.request.urlopen(image_url) as response:
        image_data = response.read()
    image_file = BytesIO(image_data)
    img1 = Image.open(image_file).convert('RGB') # PIL Image size (224, 224)

    # Used this code in Lambda
    image_s3key = os.path.join(S3_ORIGINALS_DIR, filename)
    print('Loading png from S3', image_s3key, 'from', S3_BUCKET)
    file_obj = s3.get_object(Bucket=S3_BUCKET, Key=image_s3key)
    image_bytes = BytesIO(file_obj['Body'].read())
    img2 = Image.open(image_bytes).convert('RGB') # pil_image.size (224, 224) with 3 channels

    return img1, img2

def compare_PILs(img1=None, img2=None): # requires two PIL images, ex: img1 = Image.open(img1_path)
    if img1 is None and img2 is None:
        TEST_FILENAME = "MHIST_aah.png"
        img1, img2 = get_PILS(TEST_FILENAME)

    if img1.size != img2.size:
        print('images are different sizes')
        return False

    print('Testing pixel by pixel:')
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    return np.array_equal(arr1, arr2)


# def preprocess(image_url):
#     print('image_url', image_url)
#     with urllib.request.urlopen(image_url) as response:
#         image_data = response.read()
#     image_file = BytesIO(image_data)
#     image_PIL = Image.open(image_file).convert('RGB') # PIL Image size (224, 224)
#     # print('image_PIL dimensions', image_PIL.size)

#     # Mean and std values are calculated from the training data, to normalize the colors (per channel):
#     # torchvision transforms output will be a single image: torch.Size([150528]), dtype torch.float32
#     # Model expects the input to be ndarray (150528,)
#     train_mean = [0.738, 0.649, 0.775]
#     train_std =  [0.197, 0.244, 0.17]

#     # torchvision.transforms.ToTensor: Converts a PIL Image or
#     # numpy.ndarray (H x W x C) in the range [0, 255] to a
#     # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
#     val_MHIST_FCN_transforms = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(train_mean, train_std), # Channels must be dim 0
#         # transforms.Lambda(lambda x: torch.flatten(x)) # [3, 224, 224] to [150528]
#     ])

#     preprocessed_image = val_MHIST_FCN_transforms(image_PIL).numpy() # ndarray shape (150528,)
#     print('Flattened preprocessed_image numpy shape', preprocessed_image.shape)
#     return np.expand_dims(preprocessed_image, axis=0)


# # Run inference with local ONNX model
# def predict(image_url, ort_session): # <class '_io.BytesIO'>    
#     start_inference = time.monotonic()
#     preprocessed_image = preprocess(image_url)
#     preprocess_time = time.monotonic()
#     input_name = ort_session.get_inputs()[0].name
#     ort_outs = ort_session.run(None, {input_name: preprocessed_image}) # ort_outs: [array([-1.2028292], dtype=float32)]
#     # result = ort_session.infer(preprocessed_image)
#     inference_time = time.monotonic()

#     # print('logits_numpy', ort_outs[0]) # <class 'numpy.ndarray'> shape (1,) dtype=float32
#     y_pred = torch.sigmoid(torch.from_numpy(ort_outs[0])).item()
#     positive_class = y_pred > 0.5
#     json_info = json.dumps({
#         'preprocess_time': preprocess_time-start_inference,
#         'inference_time': inference_time-preprocess_time,
#         'probability': y_pred if positive_class else 1-y_pred,
#         'predicted_class': 'SSA' if positive_class else 'HP'
#         })
#     return json_info