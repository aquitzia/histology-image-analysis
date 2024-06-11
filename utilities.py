# Run inference with ONNX model downloaded from MLflow server:
# about 4.3 GB memory use???
import os
import time

# To download and run model
import mlflow
import onnxruntime
from onnxruntime import InferenceSession

# To download and preprocess image for inference
import urllib.request
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
# ONNX_MODEL_PATH = 'artifacts/mhist_dynamo_model.onnx' #1.4G

### MLflow information required for downloading artifacts:
MLFLOW_SERVER="http://ec2-3-101-143-87.us-west-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_SERVER)
MLFLOW_RUN = "4b90e0f629f948a4845a0533aa795b01" # run_name = orderly-shrike-695
MLFLOW_MODEL_PATH = 'onnx_artifacts'
LAMBDA_TMP = 'lambda_tmp/'

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

def predict(image_url): # <class '_io.BytesIO'>
    # Download model files from MLflow server to Lambda tmp/
    print('MLflow Tracking URI:', mlflow.get_tracking_uri())
    model_uri = f"runs:/{MLFLOW_RUN}/{MLFLOW_MODEL_PATH}"
    # print('model_uri', model_uri)
    start_time = time.monotonic()
    model = mlflow.onnx.load_model(model_uri=model_uri, dst_path=LAMBDA_TMP) # returns <class 'onnx.onnx_ml_pb2.ModelProto'> <class 'mlflow.pyfunc.PyFuncModel'>
    downloaded_time = time.monotonic()
    serialized = model.SerializeToString() # <class 'bytes'>
    serialized_time = time.monotonic()
    print(f'downloaded model in {(start_time-downloaded_time):.2f}s')
    print(f'serialized model in {(downloaded_time-serialized_time):.2f}s')

    # mlflow_files = mlflow.artifacts.download_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH, dst_path=LAMBDA_TMP)
    print('Downloaded model files:\n', os.listdir(LAMBDA_TMP))
    # mlflow_files=mlflow.artifacts.list_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH)

    # Get MLflow model run info:
    run = mlflow.get_run(MLFLOW_RUN)
    # print('MLflow Run ID:', run.info.run_id)
    run_name = run.data.tags['mlflow.runName']
    print('Run name:', run_name)
    # tags = run.data.tags
    # metrics = run.data.metrics

    # Run inference with downloaded ONNX model
    onnx_dir = os.path.join(LAMBDA_TMP, MLFLOW_MODEL_PATH)
    # session_options = onnxruntime.SessionOptions()
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = InferenceSession(serialized, providers=['CPUExecutionProvider'])
    # result = onx.infer(preprocessed_image)
    input_name = ort_session.get_inputs()[0].name
    start_inference = time.monotonic()
    preprocessed_image = preprocess(image_url)
    preprocess_time = time.monotonic()
    ort_outs = ort_session.run(None, {input_name: preprocessed_image})
    inference_time = time.monotonic()
    # print('ort_outs', ort_outs) # [array([-1.2028292], dtype=float32)]
    print(f'preprocessed image in {(start_inference-preprocess_time):.2f}s')
    print(f'classified image in {(preprocess_time-inference_time):.2f}s')

    logit = ort_outs[0].item() # <class 'numpy.ndarray'> shape (1,) dtype=float32
    # print('ONNX model logit =', logit)
    return 'SSA' if logit > 0 else 'HP'
