### **Fake News Detector**

This is an end-to-end project, which is meant to demonstrate a popular NLP model using several ML Ops tools. This project was created by Angelita Garcia-Stonehocker, an ML Engineer who also has experience being a member of a board, teacher, and university lecturer. She has a degree in Computer Science from Stanford University.

LinkedIn: https://www.linkedin.com/in/aqgarcia/

GitHub: https://github.com/aquitzia

**Model**
I used a Pre-trained RoBERTa model, which uses byte-pair encodings (BPEs) on a masked-language modeling (MLM) objective. RoBERTa has a similar architecture to BERT, which is a powerful Natural Language Processing (NLP) transformer model. However, RoBERTa is more lightweight with similar results on measures of model performance.

**Fine-Tuning**
I used the Huggingface transformers library to download the pretrained model from Huggingface Hub, easily add a classification head for this (binary) text-classification task, and modify the defaults, such as class labels, before fine-tuning the model on a publicly available dataset of published articles.

**Dataset**
The initial dataset was in a CSV file. I used Pandas, Numpy, and Python for exploration and preprocessing. I pickled the files for long-term use and portability. I converted the pickled file to a Huggingface Dataset, which uses Arrow files in the implementation and works well with the rest of the Huggingface framework. Arrow is columnar and loads only when necessary, easing memory requirements, which is important with large amounts of data, and they make training lighting-fast (Well, relatively speaking)! I didn't use data versioning for this project.

**Training**
I used the Huggingface transformers Trainer and ran several experiments. In my first experiment, I trained all weights with a large learning rate, then compared that result with an experiment where I used PyTorch to manually freeze all weights except the head (last layer), then lowered the learning rate and fine-tuned all (hundreds of thousands of) parameters. I sub-classed Trainer to define a custom loss function and passed in a custom method for evaluating the model using my own calculations as well as sklearn's functions to compute accuracy, precision, recall, and F1-Score on the evaluation set, each epoch. (Yes, that's overkill, but I love to analyze all the numbers!) The dataset is balanced, so I chose to use the model with the lowest evaluation loss, rather than the additional metrics, for evaluating the best model, but it was helpful to compare and log all the metrics for further information on the model performance. I ran enough training epochs with the dataset such that the loss and evaluation loss decreased as much as possible. As the training loss approached zero, the evaluation loss worsened, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters that I used.

**Tracking**
I logged the training runs within multiple experiments in MLflow and logged the artifacts of each fine-tuned model along with the training and evaluation metrics by using the Huggingface MLflow integration autologging, as well as a custom callback passed to transformers Trainer.

**Back end**
I hosted the MLflow tracking server on AWS EC2, used S3 for storing artifacts, and PostGresQL for the model metrics, parameters, and other metadata. I also set up an MLflow Model Registry for versioning and the ease of use of model aliases for easily and seamlessly deploying the latest model.

**Inference**
Huggingface models are not optimized for inference. I converted the model with Huggingface Optimum, which uses an ONNX computational graph for serialization. I also used ONNX model optimizations and then used ONNX runtime for inference. I used an AWS Lambda function and API Gateway as a serverless solution. Lambda has a very limited runtime max of 15 min, but it is more cost-effective for running inference in this case with a mid-sized model and dataset and low traffic, compared to a long-running solution like EC2. AWS Fargate is a similar serverless solution, but better suited for batch-inference or models that require more resources. I deployed a custom Lambda Docker image using ECR (AWS Elastic Container Registry). Lambda Functions have a limit of 250MB for (read-only) code and files within the container, but it is configurable to use up to 10GB of ephemeral storage. In the Docker container, I used MLflow (backed by S3) to deploy the best model and used Optimum's ONNX runtime to the model that is optimized inference.

**AWS Cloud and ML Ops Tools**
I used boto3, AWS CLI v3, and the AWS console to set up and monitor various services, including IAM Roles, VPCs, security groups with custom inbound and outbound rules, CloudWatch logs, alarms, and budgets. I created a Streamlit app to set up a simple front-end for making post requests. Streamlit is an easy solution for prototyping or even validating a new model making it easy to access real-time inference, easy formatting, and even beautiful visualizations. CI/CD is implemented with GitHub Actions and version control with GitHub. I trained the models in Google Colab with L4 and A100 GPUs, as available. Development was done with VScode, Google Colab, Jupyter Notebooks/Lab, and vim in Terminal. I also used Terminal to SSH as well as the Remote Explorer extension on VScode.