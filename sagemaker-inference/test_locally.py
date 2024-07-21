import os
import json
from src.inference import model_fn, input_fn, predict_fn, output_fn

TEST_INPUT = {
    'bucket': 'mhist-streamlit-app',
    'key': 'images/original/MHIST_aah.png'
}

model = model_fn(os.getcwd())
input_data = input_fn(json.dumps(TEST_INPUT), 'application/json')
prediction = predict_fn(input_data, model)
output = output_fn(prediction, 'application/json')
print("Output:", output)
