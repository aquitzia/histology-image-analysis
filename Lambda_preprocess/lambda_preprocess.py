import os
import json

# To download image, upload bytes
import boto3
from io import BytesIO

# To preprocess for inference
from PIL import Image
import numpy as np
# from torchvision import transforms

# Mean and std values are calculated from the training data, to normalize the colors (per channel):
# Model expects the input to be ndarray (150528,), dtype torch.float32
TRAIN_MEAN = [0.738, 0.649, 0.775]
TRAIN_STD =  [0.197, 0.244, 0.17]

EFS_ACCESS_POINT = '/mnt/efs' # root directory is mounted here
S3_BUCKET = "mhist-streamlit-app"
S3_ORIGINALS_DIR = "images/original/"
S3_PREPROCESSED_DIR = "images/test-set/preprocessed/"

PREPROCESS_PATH = '/preprocess'
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
    # print('Getting', image_s3key, 'from', S3_BUCKET)

    s3 = boto3.client('s3')
    file_obj = s3.get_object(Bucket=S3_BUCKET, Key=image_s3key)
    print('Loaded image from S3', image_s3key)
    image_bytes = BytesIO(file_obj['Body'].read())
    preprocessed_np_image = normalize_image(image_bytes)

    # val_MHIST_FCN_transforms = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(train_mean, train_std),
    #     # transforms.Lambda(lambda x: torch.flatten(x))
    # ])
    # preprocessed_image = val_MHIST_FCN_transforms(image_PIL).numpy()
    preprocessed_flattened = np.ravel(preprocessed_np_image)  # ndarray shape (150528,)
    # print('preprocessed_flattened shape', preprocessed_flattened.shape)

    # Convert to image back to bytes buffer, upload to a different dir on S3
    preprocessed_bytes = preprocessed_flattened.tobytes()
    preprocessed_buffer = BytesIO(preprocessed_bytes)
    s3_obj_key = os.path.join(S3_PREPROCESSED_DIR, image_filename)
    s3.upload_fileobj(preprocessed_buffer, S3_BUCKET, s3_obj_key) # images/test-set/preprocessed/MHIST_*.png

    # Verify file was uploaded
    if key_in_bucket(s3, s3_obj_key):
        print('Uploaded image to s3 with key', s3_obj_key)
    else:
        print('Failed to upload image buffer to S3')

    return s3_obj_key


def key_in_bucket(s3, key):
    # There are more than 1000 objects in the bucket. Use paginator.
    page_iterator = s3.get_paginator('list_objects_v2').paginate(Bucket=S3_BUCKET)
    for page in page_iterator:
        for obj in page['Contents']:
            if obj['Key'] == key: # "images/test-set/preprocessed/MHIST_*.png"
                return True
    return False


def lambda_handler(event, context):
    # print('Event:', event)
    # action = event['action'] # API Gateway uses different JSON
    action = event['rawPath'] # API Gateway uses different JSON
    # print('action', action)

    if action == PREPROCESS_PATH:
        decodedEvent = json.loads(event['body']) # API Gateway uses different JSON
        # image_filename = event['image_filename']
        image_filename = decodedEvent['image_filename']
        # print('Preprocesing', image_filename)
        s3_path = preprocess(image_filename)

        # We don't use this
        print('EFS_ACCESS_POINT contents:', os.listdir(EFS_ACCESS_POINT)) # EFS Access Point has access to the contents of /mhist-lambda

        return s3_path

    elif action == INFO_PATH:
        # re_train()
        return "This model was created by Angelita Garcia-Stonehocker. I used a Pre-trained RoBERTa model, which uses byte-pair encodings (BPEs) on a masked-language modeling (MLM) objective. RoBERTa has a similar architecture to BERT, which is a powerful Natural Language Processing (NLP) transformer model. However, RoBERTa is more lightweight with similar results on measures of model performance. I used the Huggingface transformers library to download the pretrained model from Huggingface Hub, easily add a classification head for this text-classification task, and modify the defaults, such as class labels, before fine-tuning the model on a publicly available dataset of published articles. I used the Huggingface Trainer and ran several experiments. In my first experiment, I trained all weights with a large learning rate, then compared that result with an experiment where I used PyTorch to manually freeze all weights except the head (last layer), then lowered the learning rate and fine-tuned all (hundreds of thousands of) parameters. I used a Huggingface Trainer with PyTorch, subclassing Trainer to define a custom loss function and passed in a custom method for evaluating the model using my own calculations as well as sklearn's functions to compute accuracy, precision, recall, and F1-Score on the evaluation set, each epoch. The dataset is balanced, so I chose to use the model with the lowest evaluation loss, rather than the additional metrics, for evaluating the best model, but it was helpful to compare and log all the metrics for further information on the model performance. I ran enough training epochs with the dataset such that the loss and evaluation decreased and the model overfit. As the loss approached zero, the evaluation did become worse, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters that I used. I logged the training runs within multiple experiments in MLflow and logged the artifacts of each fine-tuned model along with the training and evaluation metrics by using the Huggingface MLflow integration autologging, as well as a custom callback. I set up the MLflow tracking server on AWS EC2, used S3 for storing artifacts, and PostGresQL for the model metrics and other metadata. I also set up an MLflow Model Registry for versioning and the ease of use of model aliases for easily and seamlessly deploying the latest model. However, the Huggingface model is not optimized for inference. I converted the model with Huggingface Optimum, which uses an ONNX computational graph for serialization. I also used ONNX model optimizations and then used ONNX runtime for inference. I used an AWS Lambda function and API Gateway as a serverless solution, which is more cost-effective on AWS than a long-running server like EC2. I deployed the optimized ONNX model to a custom Lambda Docker image, which has a limit of 250MB for (read-only) code and files within the container, but it is configurable to use up to 10GB of ephemeral storage. I used boto3, AWS CLI v3, and the AWS console to set up and monitor various services, including IAM Roles, VPCs, security groups with custom inbound and outbound rules, CloudWatch logs, alarms, and budgets. I created a Streamlit app to set up a simple front-end for making post requests. Streamlit is easily adapted or changed for the purpose of validating a new model and ease of access to real-time inference. CI/CD is implemented with GitHub Actions and version control with GitHub. I trained the models in Google Colab, on CPU as well as with L4 and A100 GPUs, as available. Development was done with VScode, Google Colab, Jupyter Notebooks/Lab, and vim in Terminal. I used Terminal to SSH as well as the Remote Explorer extension on VScode."

    else:
        return "Please provide a valid parameter"
