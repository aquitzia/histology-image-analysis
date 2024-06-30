import os
import time

# To download and preprocess an image for real-time inference
import boto3 # 1.34.127
import botocore # handles exceptions
S3_BUCKET = "mhist-streamlit-app"
S3_ORIGINALS_DIR = "images/test-set/original/"
from io import BytesIO
from PIL import Image
import numpy as np # 1.26.4

# Mean and std values are calculated from the training data, to normalize the colors (per channel):
TRAIN_MEAN = [0.738, 0.649, 0.775]
TRAIN_STD =  [0.197, 0.244, 0.17]

# To run model
import onnxruntime # 1.18.0
from onnxruntime import InferenceSession
# VIT_MODEL_PATH = '/mnt/efs/mhist-lambda/onnx_artifacts/mhist_vit_f1_dynamo_model.onnx' # set by docker-compose


class ModelHandler:
    def __init__(self, model_path, logger):
        try:
            self.logger = logger
            start_onnx_session = time.monotonic()
            # session_options = onnxruntime.SessionOptions()
            # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = InferenceSession(model_path, providers=['CPUExecutionProvider'])
            # ort_session = InferenceSession(os.path.abspath(onnx_path))#, providers=['CPUExecutionProvider'])
            self.load_time = time.monotonic() - start_onnx_session
            logger.info(f'Loaded model in {self.load_time} seconds') # 1.1504546720000235
        except ValueError as e:
            logger.info(f"Value error: {e}") # incorrect input shapes or types
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")

    @staticmethod
    def __s3_get_object(image_filename):
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
            logger.info(f"An AWS error occurred: {e}")
        except botocore.exceptions.NoCredentialsError:
            logger.info("Credentials not available")
        except botocore.exceptions.PartialCredentialsError:
            logger.info("Incomplete credentials provided")
        except Exception as e:
            logger.info(f"An error occurred: {e}")


    @staticmethod
    def __standardize_image(np_image):
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
    @staticmethod
    def __preprocess(image_filename):
        # Download image (png file) as bytes from S3
        file_obj = ModelHandler.__s3_get_object(image_filename)
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
        standardized_np = ModelHandler.__standardize_image(normalized_np) # normalize color-channels
        # print('standardized_np image shape', standardized_np.shape)
        return np.expand_dims(standardized_np, axis=0)# .unsqueeze(0) doesn't work with np

    @staticmethod
    def __sigmoid(np_outs):
        np_outs = np.clip(np_outs, -50, 50) # prevent np.exp overflow for large values (probably an error)
        return 1 / (1 + np.exp(-np_outs))


    def predict(self, image_filename):
        start_preprocess = time.monotonic()
        preprocessed_image = self.__preprocess(image_filename)
        preprocess_time = time.monotonic() - start_preprocess

        # Run inference with optimized ONNX model
        # It uses 3.3 GB CPU memory, and 1.4 GB space (for artifacts)
        start_inference = time.monotonic()
        input_name = self.ort_session.get_inputs()[0].name
        ort_outs = self.ort_session.run(None, {input_name: preprocessed_image}) # output: [array([-1.2028292], dtype=float32)]
        # result = self.ort_session.infer(preprocessed_image)
        inference_time = time.monotonic() - start_inference

        logit = ort_outs[0].item() # <class 'numpy.ndarray'> shape (1,) dtype=float32
        positive_prob = self.__sigmoid(logit).item()
        pred = positive_prob > 0.5
        self.logger.info(f'ONNX model ran inference on image_filename{image_filename} logit {logit}')
        inference_info = {
            'logit': logit, # check image_filename MHIST_aah.png logit=2.16929292678833
            'predicted_class': 'SSA' if pred else 'HP',
            'probability': positive_prob if pred else 1-positive_prob,
            # 'model_load_time': self.load_time, # 1.1504546720000235
            'preprocess_time': preprocess_time,
            'inference_time': inference_time-preprocess_time,
            }
        return inference_info
