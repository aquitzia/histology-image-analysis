import os
import mlflow
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline

# print('Transformers cache dir:', os.getenv('TRANSFORMERS_CACHE')) # Will be deprecated
# print('Huggingface home dir:', os.getenv('HF_HOME'))

### MLflow information required for downloading artifacts:
MLFLOW_SERVER="http://ec2-3-101-143-87.us-west-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_SERVER)
MLFLOW_RUN = "4f9a2ca283f141769647f773ae61c15a" # run_name = "languid-dolphin-519"
MLFLOW_MODEL_PATH = 'huggingface_optimum_artifacts'
LAMBDA_TMP = '/tmp/'    # use /tmp/ for downloading large (ephemeral) files in a Lambda function

PREDICT_PATH = 'predict'
INFO_PATH = 'info'

def predict(articles):

    # Download model files from MLflow server
    print('MLflow Tracking URI:', mlflow.get_tracking_uri())
    mlflow_files = mlflow.artifacts.download_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH, dst_path=LAMBDA_TMP)
    print('Downloaded model files:\n', os.listdir(mlflow_files))
    # mlflow_files=mlflow.artifacts.list_artifacts(tracking_uri=MLFLOW_SERVER, run_id=MLFLOW_RUN, artifact_path=MLFLOW_MODEL_PATH)

    # Get MLflow model run info:
    run = mlflow.get_run(MLFLOW_RUN)
    print('MLflow Run ID:', run.info.run_id)
    run_name = run.data.tags['mlflow.runName']
    print('Run name:', run_name)
    # tags = run.data.tags
    # metrics = run.data.metrics

    # Run inference with optimized ONNX model and ONNX RunTime pipeline:
    # It only uses 3.3 GB CPU memory, and 480 MB space (for artifacts)
    onnx_dir = os.path.join(LAMBDA_TMP, MLFLOW_MODEL_PATH)
    optimized_model = ORTModelForSequenceClassification.from_pretrained(os.path.abspath(onnx_dir))
    ort_pipe = pipeline("text-classification", model=optimized_model, accelerator="ort")
    return ort_pipe(articles)

def lambda_handler(event, context):
    # print(event)
    action = event['Action'] # API Gateway uses different JSON

    if action == PREDICT_PATH:
        
        articles = event['Text']
        print(articles)
        results = predict(articles)
        print('Model output:', results)
        return results

    elif action == INFO_PATH:
        # re_train()
        return "This model was created by Angelita Garcia-Stonehocker. I used a Pre-trained RoBERTa model, which uses byte-pair encodings (BPEs) on a masked-language modeling (MLM) objective. RoBERTa has a similar architecture to BERT, which is a powerful Natural Language Processing (NLP) transformer model. However, RoBERTa is more lightweight with similar results on measures of model performance. I used the Huggingface transformers library to download the pretrained model from Huggingface Hub, easily add a classification head for this text-classification task, and modify the defaults, such as class labels, before fine-tuning the model on a publicly available dataset of published articles. I used the Huggingface Trainer and ran several experiments. In my first experiment, I trained all weights with a large learning rate, then compared that result with an experiment where I used PyTorch to manually freeze all weights except the head (last layer), then lowered the learning rate and fine-tuned all (hundreds of thousands of) parameters. I used a Huggingface Trainer with PyTorch, subclassing Trainer to define a custom loss function and passed in a custom method for evaluating the model using my own calculations as well as sklearn's functions to compute accuracy, precision, recall, and F1-Score on the evaluation set, each epoch. The dataset is balanced, so I chose to use the model with the lowest evaluation loss, rather than the additional metrics, for evaluating the best model, but it was helpful to compare and log all the metrics for further information on the model performance. I ran enough training epochs with the dataset such that the loss and evaluation decreased and the model overfit. As the loss approached zero, the evaluation did become worse, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters that I used. I logged the training runs within multiple experiments in MLflow and logged the artifacts of each fine-tuned model along with the training and evaluation metrics by using the Huggingface MLflow integration autologging, as well as a custom callback. I set up the MLflow tracking server on AWS EC2, used S3 for storing artifacts, and PostGresQL for the model metrics and other metadata. I also set up an MLflow Model Registry for versioning and the ease of use of model aliases for easily and seamlessly deploying the latest model. However, the Huggingface model is not optimized for inference. I converted the model with Huggingface Optimum, which uses an ONNX computational graph for serialization. I also used ONNX model optimizations and then used ONNX runtime for inference. I used an AWS Lambda function and API Gateway as a serverless solution, which is more cost-effective on AWS than a long-running server like EC2. I deployed the optimized ONNX model to a custom Lambda Docker image, which has a limit of 250MB for (read-only) code and files within the container, but it is configurable to use up to 10GB of ephemeral storage. I used boto3, AWS CLI v3, and the AWS console to set up and monitor various services, including IAM Roles, VPCs, security groups with custom inbound and outbound rules, CloudWatch logs, alarms, and budgets. I created a Streamlit app to set up a simple front-end for making post requests. Streamlit is easily adapted or changed for the purpose of validating a new model and ease of access to real-time inference. CI/CD is implemented with GitHub Actions and version control with GitHub. I trained the models in Google Colab, on CPU as well as with L4 and A100 GPUs, as available. Development was done with VScode, Google Colab, Jupyter Notebooks/Lab, and vim in Terminal. I used Terminal to SSH as well as the Remote Explorer extension on VScode."

    else:
        return "Please provide a valid parameter"
