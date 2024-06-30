import os
from model_handler import ModelHandler
from flask import Flask, request, jsonify

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FlaskApp')
model_logger = logging.getLogger('Model')

app = Flask(__name__)

# Pre-load the model once (in 1.2 seconds), for faster inference
model_path = os.environ['VIT_MODEL_PATH']
# model_path = os.environ.get('VIT_MODEL_PATH', '/mnt/efs/mhist-lambda/onnx_artifacts/mhist_vit_f1_dynamo_model.onnx')
# print(f"All environment variables: {os.environ}")
logger.info(f'\nLoading model from {model_path}')#os.path.abspath(model_path))
model = ModelHandler(model_path, model_logger)


@app.route('/predict', methods=['POST'])
def predict(): # image_url <class '_io.BytesIO'>
    if not request.is_json:
        return jsonify({"Error": "Request must be JSON"}), 400
    data = request.get_json() # request is a "context local" (global within the thread, so it's safe for multi-threaded env) Flask object
    # logger.info(f"Received data: {data}, len {len(data)}")

    image_filename = data['image_filename'] # data is <class 'dict'>
    if not image_filename:
        return jsonify({"Error": "image_filename is required"}), 400
    logger.info(f"image_filename: {image_filename}")

    inference_info = model.predict(image_filename)
    return jsonify(inference_info), 200


# # Debug
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
