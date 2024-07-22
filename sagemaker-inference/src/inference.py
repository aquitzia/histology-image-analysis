import os
import json
import time

# For preprocessing
import boto3  # 1.34.127
from io import BytesIO
from PIL import Image
import numpy as np # 1.26.4
s3 = boto3.client('s3')

# Mean and std values are calculated from the training data, to normalize the colors (per channel):
TRAIN_MEAN = [0.738, 0.649, 0.775]
TRAIN_STD =  [0.197, 0.244, 0.17]

# For loading model
import onnxruntime # 1.18.0
from onnxruntime import InferenceSession
MODEL_PATH = 'MHIST_ViT_v13_dynamo_model.onnx'

'''
Model loading is the process of deserializing your saved model back into a PyTorch model.
Serving is the process of translating InvokeEndpoint requests to inference calls on the loaded model.

Model loading:
model_fn: SageMaker PyTorch model server loads your model by invoking model_fn.

Model serving:
input_fn: Takes request data and deserializes the data into an object for prediction.
predict_fn: Takes the deserialized request object and performs inference against the loaded model.
output_fn: Takes the result of prediction and serializes this according to the response content type.
'''

# Load the model from a file (ex: model.pth) from model_dir
def model_fn(model_dir):
    model_path = os.path.join(model_dir, MODEL_PATH)
    session = InferenceSession(model_path)
    return session


##### Preprocess #####
def standardize_image(np_image):
    # Convert lists to numpy
    np_mean = np.array(TRAIN_MEAN, dtype=np.float32).reshape(3, 1, 1) # np_mean.shape (3, 1, 1)
    # print('np_mean shape', np_mean.shape)
    np_std = np.array(TRAIN_STD, dtype=np.float32).reshape(3, 1, 1) # np_std.shape (3, 1, 1)

    # Normalize: operations are performed element-wise using NumPy broadcasting
    np_image = (np_image - np_mean) / np_std
    return np_image

def preprocess(image_bytes):
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
    return np.expand_dims(standardized_np, axis=0)


##### Predict #####
# Where request_body is a byte buffer and request_content_type is a Python string.
#
# The request Content-Type, for example “application/x-npy”
# The request data body, a byte array
def input_fn(serialized, content_type):
    if content_type == 'application/json':
        input_data = json.loads(serialized)
        bucket = input_data['bucket']
        key = input_data['key']
        
        # Download image (png file) as bytes from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        image_file = response['Body'].read()
        return preprocess(BytesIO(image_file))        

    else:
        raise ValueError(f"Unsupported content type: {content_type}")
        
# Predict with ONNX model
def predict_fn(input_data, model):
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    result = model.run([output_name], {input_name: input_data})
    return result[0]


##### Post-process #####
def sigmoid(np_outs):
    np_outs = np.clip(np_outs, -50, 50) # prevent np.exp overflow for large values (probably an error)
    return 1 / (1 + np.exp(-np_outs))


def output_fn(predictions, content_type):
    # if content_type == 'application/json':
    # predictions = json.dumps(prediction_output.tolist())
    logit = predictions[0].item() # <class 'numpy.ndarray'> shape (1,) dtype=float32
    positive_prob = sigmoid(logit).item()
    pred = positive_prob > 0.3
    # self.logger.info(f'ONNX model ran inference on image_filename{image_filename} logit {logit}')
    inference_info = {
        'logit': logit, # check image_filename MHIST_aah.png logit=2.16929292678833
        'predicted_class': 'SSA' if pred else 'HP',
        'probability': positive_prob if pred else 1-positive_prob,
        # 'model_load_time': self.load_time, # 1.1504546720000235
        # 'preprocess_time': preprocess_time,
        # 'inference_time': inference_time-preprocess_time,
        }
    return json.dumps(inference_info), content_type
    # raise ValueError(f"Unsupported content type: {content_type}")