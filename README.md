This is an end-to-end project, which is meant to compare common techniques for implementing basic computer vision models using several ML Ops tools. This project was created by Angelita Garcia-Stonehocker, a software engineer who also has experience being a member of a board, teacher, and university lecturer. She has a degree in Computer Science from Stanford University.

LinkedIn: https://www.linkedin.com/in/aqgarcia/

GitHub: https://github.com/aquitzia

**Files included:**
`Google_Colab_training` Google Colab notebook for data exploration, training, hyperparameter tuning with MLflow for tracking metrics, logging best models, and model evaluation and analysis
`sagemaker-inference` and `artifacts` for running inference in SageMaker
`flask_files` contain file for the Flask app, Docker image, and Docker Compose
`Lambda_predict` contains files for the AWS Lambda Function
`.github/workflows` contains two files for GitHub Actions to deploy Docker Containers. One is for the Lambda Function the other is for the Docker container running on EC2.
Most of the other files and folders in this directory are for the Streamlit frontend files: `mhist-app, thumb, diagrams, mhist-venv, markdown files, and CSV files`

## **Dataset**
The images in the dataset are fixed-size, due to the limitations of the model and the standard in computer vision. The color in the images is mostly shades of purple (against a white background), so I opted to normalize the colors and used torchvision transforms to effectively apply normalization, standardization, as well as other steps required for input into the model. In a couple of my experiments, I added synthetic data with 90-degree rotations and flipping, but there was limited payoff, compared to more traditional methods of fine-tuning, such as adjusting the learning rate or using a different scheduler.

## **Models**

### **Multi-level Perceptron**

I trained two models for this project. This first model is a simple Multi-level Perceptron without dropout, no CNN, nothing fancy. My goal was to train a fully-connected model on images with the purpose of gaining an intuition for the capabilities and pitfalls of such a design. The most obvious drawback is the size of this learning machine! It requires about five times more storage space than the ViT model I trained, even after compression and optimization with ONNX!

**Fine-Tuning**
The primary challenge in training an MLP, particularly on an imbalanced dataset, is rapid overfitting over the course of even a few epochs. Adding a drop-out layer would likely offer a great improvement. I trained on various metrics, which each quickly improved, to the detriment of other metrics. The model had very mediocre results overall, and not nearly good enough results to use in a medical setting. This model is powerful, but like a souped-up car, it just has too much power to meet safety standards!

**Training**
I created a custom trainer object (in Pytorch) to suit my purposes and ran several experiments. The custom trainer was very helpful for comparing models. I found it even easier to use than an off-the-shelf trainer like Huggingface offers because I designed it efficiently and with just enough flexibility that it is useful without having an overwhelming number of options or requirements for adapting it to a new model.

For each run, I iterated over the training set with enough epochs that the loss and evaluation loss decreased as much as possible before evaluation loss began to increase (the model overfit). As the training loss approached zero, the evaluation loss worsened, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters and methods that I used.

I tried several FC models of varying numbers of layers and nodes, but I was surprised to find that one of the simplest architectures is the best among the designs I tried. It has only three hidden layers! The sizes of each layer are: 150528 (input), 2352, 2352, 294, and 1 (output) node. The input images are all 224 x 224 pixels by 3 channels (8-bit color depth in RGB format).

### **Vision Transformer**

This model is based on ViT, which stands for Vision Transformer. It was designed by Google in 2020 and it was pre-trained on images that are 224 x 224 pixels. Before transformers became popular, CNN (convolutional neural network) models set the standard in computer vision. We have, now, adapted transformers (similar to ChatGPT), to be used effectively on images as well as text. ViT breaks up an image into patches, which are 16x16 pixels. These patches are processed linearly and treated as a sequence of tokens, similar to NLP transformers.

**Fine-Tuning**
I used the timm library, rather than Huggingface, to download the pretrained model and easily add a classification head for this (binary) classification task, before fine-tuning the model on a publicly available dataset: MHIST. The data is quite imbalanced due to the nature of cancer screenings. There are many more negative than positive samples, which makes training more complex. For this dataset, F1-score should have been a great metric to guide training because it is a more balanced measure than accuracy or other standard metrics. However, it was necessary to focus more on the true-positive rate while training on this dataset to compel the model to focus on false negatives, in particular. A false negative can be quite detrimental in pathology because it would give a patient a false sense of security. A patient that has a negative result would be treated as quickly, possibly leading to dire consequences.

However, when I focused the training on loss, while saving models on an evaluation metric of true-positive rate, the model became overly focused on true positive samples, quickly overfitting and the F1-score, ROC-AUC, and other balanced metrics were quite low. Weighted and balanced metrics were often near 50/50 chance, which is close to random. As expected, it was important to train on multiple metrics simultaneously. It was helpful for me to create my own method of training that keeps various cases in mind so that, as TPR increased, it wouldn't excessively skew training or prevent an undesirable drop in the balanced metrics.

I also needed to adjust the threshold to squeeze the most out of training and inference on this imbalanced dataset with a transformer that can't "see" the same way a CNN can. I was quite pleased to find out that adjusting my threshold from the typical 0.5 to 0.4 and sometimes as low as 0.3 would significantly increase TPR and some of the balanced metrics. For example, ROC-AUC increased from 0.5 to 0.8764. Of course, I could have decreased it even more to get 100% TPR, but that negatively impacts other metrics.

**Training**
I created a custom trainer object (in Pytorch) to suit my purposes and ran several experiments. When I trained all weights with a large learning rate, I quickly lost the benefit of using pre-trained weights and the model overfit quickly.

I had much better results when I froze all weights except the classification head, then lowered the learning rate and fine-tuned all parameters. I was surprised to find that a high learning rate was important for training this model, as well as using a very focused metric, true-positive rate, as described above. I ran enough training epochs with the training set such that the loss and evaluation loss decreased as much as possible before evaluation loss began to increase. As the training loss approached zero, the evaluation loss worsened, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters and methods that I used. I found that learning rates as high as 0.01 to 0.1 gave me the best results.

## **Tracking**
I set up an MLflow remote tracking server to log the training runs within a single "experiment" for easy comparison and visualization. I kept careful records of model parameters, information and even some images (as artifacts) for each fine-tuned model. I logged copious metrics for each run during training and evaluation as well as for analysis later. MLflow is an excellent tool for making decisions about the dynamic interplay of metrics during model training and evaluation.

**Back end**
I hosted the MLflow tracking server on AWS EC2, used S3 for storing artifacts, and PostGresQL (on RDS) for the model metrics, parameters, and other metadata. I prefer not to use the MLflow model registry because I find that it is not flexible enough for my use. I used ONNX runtime which is highly efficient for running inference and it has the capability of pre-loading a model, which makes repeated real-time inference or batch inference very efficient. I was able to reduce inference time from minutes to tenths of a second by using ONNX runtime. After converting the model to ONNX, I log the model as an artifact in the appropriate MLflow run.

## **Network architecture and ML Ops Tools**

**ONNX**

PyTorch models are not optimized for inference and they can become unwieldy due to high memory requirements and large size. I converted each model to ONNX, which optimizes the computational graph upon serialization. This reduced the file size, but more importantly it reduced the size of the dependencies, which are particularly large for training PyTorch models on a GPU. The CUDA libraries are multiple GB, which makes loading take several minutes. ONNX runtime for CPU, on the other hand, take up less than 100MB.

**AWS Lambda Function**

I used an AWS Lambda function and API Gateway as a serverless solution, but I wanted to compare that to using a traditional long-running server with EC2. Lambda is much more cost-effective for running inference and it can provision "elastic" resources sufficiently for this mid-sized model and dataset. AWS Fargate is a similar serverless solution, but better suited for batch-inference or models that require more resources. I deployed a custom Lambda Docker image using ECR (AWS Elastic Container Registry). Lambda Functions have a limit of 250MB for (read-only) code and files within the container, but it is configurable to use up to 10GB of ephemeral storage. In the Docker container.

**Flask** and **Docker Compose**

On EC2, I used the same model in a Flask app and deployed the endpoint with Docker-compose. I set up Docker-compose to use a volume so that the Docker container and the Lambda function have access to the same model without the need for managing multiple copies. I stored the model on EFS (Elastic File System), which is an NFS (Network File System) with incredible capabilities: low-latency, high-throughput, durability, availability, and it can scale up to petabytes of data! I also set up a VPC endpoint so that Lambda (and EC2) can easily access data from an S3 bucket (or other AWS services), which improves response time without sacrificing security.

I created a **Streamlit** app as a simple front-end for making post requests. Streamlit is a quick solution for prototyping or even validating a new model making it easy to run real-time inference, as well as easy text formatting and beautiful visualizations.

I used boto3, AWS CLI v3, and the AWS console to set up and monitor various **AWS services**, including IAM Roles, VPC, public and private subnets in multiple availability zones, security groups with custom inbound and outbound rules, CloudWatch logs, alarms, and AWS Budgets with Anomaly Detection.

The AWS Lambda Function runs in a **private subnet** which it connects to via an ENI (Elastic Network Interface) to access both the VPC Gateway (see above) and the EFS mount target for that subnet (and availability zone). EFS be access across availability zones, but I only needed one for this project. I kept data in private subnets with access to the internet through a NAT Gateway, when necessary.

I implemented CI/CD with **GitHub Actions** and version control with GitHub (using a linear commit history), which build each container in parallel either on EC2 (through SSH with a private key from GitHub Secrets) or on the linux runner and pushes the container to ECR for Lambda. I also set up a security group for EC2 to ensure that only the runner has access to EC2 by adding the runner’s IP to the security group in the deployment process.

I trained the models in Google Colab with L4 and A100 GPUs, as available. Development was done with VScode, Google Colab, SageMaker, Jupyter Notebooks/Lab, and vim in Terminal. I also used Terminal to SSH as well as the Remote Explorer extension on VScode along with typical virtual environments (conda or venv) and tools like pipreqs for setting up dependencies.

