# This function requires EFS and S3
import os
import time
import json

import boto3
from io import BytesIO
from PIL import Image
import numpy as np
from onnxruntime import InferenceSession

# Mean and std values are calculated from the training data, to normalize the colors (per channel):
# Model expects the input to be ndarray (150528,), dtype torch.float32
TRAIN_MEAN = [0.738, 0.649, 0.775]
TRAIN_STD =  [0.197, 0.244, 0.17]

### MLflow information required for downloading artifacts:
# MLFLOW_SERVER="http://ec2-54-215-248-114.us-west-1.compute.amazonaws.com:5000"
# mlflow.set_tracking_uri(MLFLOW_SERVER)
# MLFLOW_RUN = "53962bd1fead46f6bd9d647a43e7f492" # run_name = bittersweet-lark-524
EFS_ACCESS_POINT = '/mnt/efs' # root directory is mounted here
MODEL_PATH = 'onnx_artifacts/mhist_dynamo_model.onnx'
S3_BUCKET = "mhist-streamlit-app"
S3_ORIGINALS_DIR = "images/test-set/original/"
# S3_PREPROCESSED_DIR = "images/test-set/preprocessed/"

PREDICT_PATH = '/predict'
INFO_PATH = '/info'


def normalize_image(image_bytes):
    # Convert bytes (buffer) to 3-channels, then to ndarray
    # We could do this without PIL (using only NumPy)
    pil_image = Image.open(image_bytes).convert('RGB') # pil_image.size (224, 224) with 3 channels
    # print('pil_image dimensions', pil_image.size)
    np_image = np.array(pil_image, dtype=np.float32) # np_image.shape (224, 224, 3) np_image.dtype float32
    # print('np_image shape', np_image.shape)
    # print('np_image dtype', np_image.dtype)

    # Convert lists to numpy
    np_mean = np.array(TRAIN_MEAN, dtype=np.float32).reshape(1, 1, 3) # np_mean.shape (1, 1, 3)
    # print('np_mean shape', np_mean.shape)
    np_std = np.array(TRAIN_STD, dtype=np.float32).reshape(1, 1, 3) # np_std.shape (1, 1, 3)

    # Normalize
    # Operations are performed element-wise using NumPy broadcasting
    np_image = (np_image - np_mean) / np_std
    return np_image


def preprocess(image_filename):
    # Download image (png file) as bytes from S3
    image_s3key = os.path.join(S3_ORIGINALS_DIR, image_filename)
    print('Loading', image_s3key, 'from', S3_BUCKET)

    s3 = boto3.client('s3')
    file_obj = s3.get_object(Bucket=S3_BUCKET, Key=image_s3key)
    image_bytes = BytesIO(file_obj['Body'].read())
    preprocessed_np_image = normalize_image(image_bytes)

    preprocessed_flattened = np.ravel(preprocessed_np_image)  # ndarray shape (150528,)
    # print('preprocessed_flattened shape', preprocessed_flattened.shape)
    return preprocessed_flattened


def sigmoid(np_outs):
    # Clip the values to a reasonable range
    np_outs = np.clip(np_outs, -50, 50) # prevent np.exp overflow for large values
    return 1 / (1 + np.exp(-np_outs))


def predict(image_filename): # image_url <class '_io.BytesIO'>
    # print('EFS_ACCESS_POINT contents:', os.listdir(EFS_ACCESS_POINT)) # EFS Access Point has access to the contents of /mhist-lambda

    # Run inference with optimized ONNX model
    # It only uses 3.3 GB CPU memory, and 1.4 GB space (for artifacts)
    onnx_path = os.path.join(EFS_ACCESS_POINT, MODEL_PATH)
    print('Loading model from', os.path.abspath(onnx_path))
    # session_options = onnxruntime.SessionOptions()
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = InferenceSession(os.path.abspath(onnx_path))#, providers=['CPUExecutionProvider'])

    start_inference = time.monotonic()
    preprocessed_image = preprocess(image_filename)
    preprocess_time = time.monotonic()
    input_name = ort_session.get_inputs()[0].name
    ort_outs = ort_session.run(None, {input_name: preprocessed_image}) # output: [array([-1.2028292], dtype=float32)]
    # result = ort_session.infer(preprocessed_image)
    inference_time = time.monotonic()

    logit = ort_outs[0].item() # <class 'numpy.ndarray'> shape (1,) dtype=float32
    positive_prob = sigmoid(logit).item()
    pred = positive_prob > 0.5
    # print('ONNX pred =', pred)
    inference_info = {#json.dumps({
        'logit': logit,
        'predicted_class': 'SSA' if pred else 'HP',
        'probability': positive_prob if pred else 1-positive_prob,
        'preprocess_time': preprocess_time-start_inference,
        'inference_time': inference_time-preprocess_time,
        }
    return inference_info


def json_response(inference_info):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(inference_info)
    }


def lambda_handler(event, context):
    # print('Event:', event)
    # action = event['action'] # API Gateway uses different JSON
    action = event['rawPath'] # API Gateway uses different JSON
    # print('action', action)

    if action == PREDICT_PATH:
        decodedEvent = json.loads(event['body']) # API Gateway uses different JSON
        # image_filename = event['image_filename']
        image_filename = decodedEvent['image_filename']
        # print('Lambda starting inference on ', image_filename)
        inference_info = predict(image_filename)
        return json_response(inference_info)

    elif action == INFO_PATH:
        # re_train()
        return "This model was created by Angelita Garcia-Stonehocker. I used a Pre-trained RoBERTa model, which uses byte-pair encodings (BPEs) on a masked-language modeling (MLM) objective. RoBERTa has a similar architecture to BERT, which is a powerful Natural Language Processing (NLP) transformer model. However, RoBERTa is more lightweight with similar results on measures of model performance. I used the Huggingface transformers library to download the pretrained model from Huggingface Hub, easily add a classification head for this text-classification task, and modify the defaults, such as class labels, before fine-tuning the model on a publicly available dataset of published articles. I used the Huggingface Trainer and ran several experiments. In my first experiment, I trained all weights with a large learning rate, then compared that result with an experiment where I used PyTorch to manually freeze all weights except the head (last layer), then lowered the learning rate and fine-tuned all (hundreds of thousands of) parameters. I used a Huggingface Trainer with PyTorch, subclassing Trainer to define a custom loss function and passed in a custom method for evaluating the model using my own calculations as well as sklearn's functions to compute accuracy, precision, recall, and F1-Score on the evaluation set, each epoch. The dataset is balanced, so I chose to use the model with the lowest evaluation loss, rather than the additional metrics, for evaluating the best model, but it was helpful to compare and log all the metrics for further information on the model performance. I ran enough training epochs with the dataset such that the loss and evaluation decreased and the model overfit. As the loss approached zero, the evaluation did become worse, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters that I used. I logged the training runs within multiple experiments in MLflow and logged the artifacts of each fine-tuned model along with the training and evaluation metrics by using the Huggingface MLflow integration autologging, as well as a custom callback. I set up the MLflow tracking server on AWS EC2, used S3 for storing artifacts, and PostGresQL for the model metrics and other metadata. I also set up an MLflow Model Registry for versioning and the ease of use of model aliases for easily and seamlessly deploying the latest model. However, the Huggingface model is not optimized for inference. I converted the model with Huggingface Optimum, which uses an ONNX computational graph for serialization. I also used ONNX model optimizations and then used ONNX runtime for inference. I used an AWS Lambda function and API Gateway as a serverless solution, which is more cost-effective on AWS than a long-running server like EC2. I deployed the optimized ONNX model to a custom Lambda Docker image, which has a limit of 250MB for (read-only) code and files within the container, but it is configurable to use up to 10GB of ephemeral storage. I used boto3, AWS CLI v3, and the AWS console to set up and monitor various services, including IAM Roles, VPCs, security groups with custom inbound and outbound rules, CloudWatch logs, alarms, and budgets. I created a Streamlit app to set up a simple front-end for making post requests. Streamlit is easily adapted or changed for the purpose of validating a new model and ease of access to real-time inference. CI/CD is implemented with GitHub Actions and version control with GitHub. I trained the models in Google Colab, on CPU as well as with L4 and A100 GPUs, as available. Development was done with VScode, Google Colab, Jupyter Notebooks/Lab, and vim in Terminal. I used Terminal to SSH as well as the Remote Explorer extension on VScode."

    else:
        return "Please provide a valid parameter"
