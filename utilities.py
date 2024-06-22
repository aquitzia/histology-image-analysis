import os
import time
import json

# To run model
import onnxruntime
from onnxruntime import InferenceSession

# To download and preprocess an image for real-time inference
import urllib.request
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

# To download 977 thumbnails from S3
import boto3
S3_BUCKET = "mhist-streamlit-app"
S3_THUMBS_DIR = "images/test-set/thumb/"
LOCAL_THUMBS_DIR = "thumb/"

# To download artifacts from MLflow (remotely):
import mlflow
MLFLOW_SERVER="http://ec2-3-101-21-63.us-west-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_SERVER)
MLFLOW_RUN = "53962bd1fead46f6bd9d647a43e7f492" # run_name = bittersweet-lark-524
MLFLOW_MODEL_PATH = 'onnx_artifacts' #1.4G
LOCAL_MODEL_DIR = os.getcwd() # Download MLflow dir to this dest


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


# Download ONNX model from MLflow server
def download_latest_model():
    print('MLflow Tracking URI:', mlflow.get_tracking_uri())
    run = mlflow.get_run(MLFLOW_RUN)
    print('MLflow runName =', run.data.tags['mlflow.runName'], 'run_id =', run.info.run_id)
    print('Current working directory:', LOCAL_MODEL_DIR)
    # tags = run.data.tags
    # metrics = run.data.metrics

    start_time = time.monotonic()
    mlflow_files = mlflow.artifacts.download_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH, dst_path=LOCAL_MODEL_DIR)
    downloaded_time = time.monotonic()
    print('Downloaded model files:\n', mlflow_files)#os.listdir(LOCAL_MODEL_DIR))
    print(f'Downloaded model in {(downloaded_time-start_time):.2f}s')
    # mlflow_files=mlflow.artifacts.list_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH)


def preprocess(image_url):
    # print('image_url', image_url)
    with urllib.request.urlopen(image_url) as response:
        image_data = response.read()
    image_file = BytesIO(image_data)
    image_PIL = Image.open(image_file).convert('RGB') # PIL Image size (224, 224)
    # print('image_PIL dimensions', image_PIL.size)

    # Mean and std values are calculated from the training data, to normalize the colors (per channel):
    # torchvision transforms output will be a single image: torch.Size([150528]), dtype torch.float32
    # Model expects the input to be ndarray (150528,)
    train_mean = [0.738, 0.649, 0.775]
    train_std =  [0.197, 0.244, 0.17]
    val_MHIST_FCN_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    preprocessed_image = val_MHIST_FCN_transforms(image_PIL).numpy() # ndarray shape (150528,)
    # print('preprocessed_image shape', preprocessed_image.shape)
    return preprocessed_image


def init_model():
    onnx_dir = os.path.join(MLFLOW_MODEL_PATH, 'mhist_dynamo_model.onnx')
    # session_options = onnxruntime.SessionOptions()
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])
    return ort_session


# Run inference with local ONNX model
def predict(image_url, ort_session): # <class '_io.BytesIO'>    
    start_inference = time.monotonic()
    preprocessed_image = preprocess(image_url)
    preprocess_time = time.monotonic()
    input_name = ort_session.get_inputs()[0].name
    ort_outs = ort_session.run(None, {input_name: preprocessed_image}) # ort_outs: [array([-1.2028292], dtype=float32)]
    # result = ort_session.infer(preprocessed_image)
    inference_time = time.monotonic()

    # print('logits_numpy', ort_outs[0]) # <class 'numpy.ndarray'> shape (1,) dtype=float32
    y_pred = torch.sigmoid(torch.from_numpy(ort_outs[0])).item()
    positive_class = y_pred > 0.5
    json_info = json.dumps({
        'preprocess_time': preprocess_time-start_inference,
        'inference_time': inference_time-preprocess_time,
        'probability': y_pred if positive_class else 1-y_pred,
        'predicted_class': 'SSA' if positive_class else 'HP'
        })
    return json_info
