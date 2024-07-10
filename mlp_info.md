### **Multi-level Perceptron**

This is an end-to-end project, which is meant to compare common techniques for implementing basic computer vision models using several ML Ops tools. This project was created by Angelita Garcia-Stonehocker, a software engineer who also has experience being a member of a board, teacher, and university lecturer. She has a degree in Computer Science from Stanford University.

LinkedIn: https://www.linkedin.com/in/aqgarcia/

GitHub: https://github.com/aquitzia

**Model**

I trained two models for this project. This first model is a simple Multi-level Perceptron without dropout, no CNN, nothing fancy. My goal was to train a fully-connected model on images with the purpose of gaining an intuition for the capabilities and pitfalls of such a design. The most obvious drawback is the size of this learning machine! It requires about five times more storage space than the ViT model I trained, even after compression and optimization with ONNX!

**Fine-Tuning**
The primary challenge in training an MLP, particularly on an imbalanced dataset, is rapid overfitting over the course of even a few epochs. Adding a drop-out layer would likely offer a great improvement. I trained on various metrics, which each quickly improved, to the detriment of other metrics. The model had very mediocre results overall, and not nearly good enough results to use in a medical setting. This model is powerful, but like a souped-up car, it just has too much power to meet safety standards!

**Dataset**
The images in the dataset are fixed-size, which is the standard for computer vision, but the benefit of this model is that it can accept an image with any dimensions, especially given more memory. The color in the images is mostly shades of purple (against a white background), so I opted to normalize the colors with torchvision transforms to effectively apply normalization, standardization, as well as other steps required for input into the Pytorch model. In a couple of my experiments, I added synthetic data with 90-degree rotations and flipping, but there was limited payoff, compared to more traditional methods of fine-tuning, such as adjusting the learning rate or using a different scheduler, loss function, etc.

**Training**
I created a custom trainer object (in Pytorch) to suit my purposes and ran several experiments. The custom trainer was very helpful for comparing models. I found it even easier to use than an off-the-shelf trainer like Huggingface offers because I designed it efficiently and with just enough flexibility that it is useful without having an overwhelming number of options or requirements for adapting it to a new model.

For each run, I iterated over the training set with enough epochs that the loss and evaluation loss decreased as much as possible before evaluation loss began to increase (the model overfit). As the training loss approached zero, the evaluation loss worsened, as expected, assuring me that the model was able to get the most out of the training with the hyperparameters and methods that I used.

I tried several FC models of varying numbers of layers and nodes, but I was surprised to find that one of the simplest architectures is the best among the designs I tried. It has only three hidden layers! The sizes of each layer are: 150528 (input), 2352, 2352, 294, and 1 (output) node. The input images are all 224 x 224 pixels by 3 channels (8-bit color depth in RGB format).

**Tracking**
I set up an MLflow remote tracking server to log the training runs within a single "experiment" for easy comparison and visualization. I kept careful records of model parameters, information and even some images (as artifacts) for each fine-tuned model. I logged copious metrics for each run during training and evaluation for saving the best model as well as for further analysis later. MLflow is an excellent tool for making decisions about the dynamic interplay of hyperparameters and metrics during model training and evaluation.

**Back end**
I hosted the MLflow tracking server on AWS EC2, used S3 for storing artifacts, and PostGresQL (on RDS) for the model metrics, parameters, and other metadata. I prefer not to use the MLflow model registry because I find that it is not flexible enough for my use.

PyTorch models are not optimized for inference and they can become unwieldy due to high memory requirements and large size. I selected ONNX for serialization, which optimizes the computational graph. An ONNX runtime session has the capability of pre-loading a model, which makes repeated real-time inference or batch inference very efficient. I was able to reduce inference time from minutes to tenths of a second by using ONNX runtime. I logged potential best models as artifacts in the appropriate MLflow run.

**Inference**
I used an AWS Lambda function and API Gateway as a serverless solution, but I wanted to compare serverless to using a traditional long-running server with EC2. Lambda is much more cost-effective for running inference and it can provision "elastic" resources sufficiently for this mid-sized model and dataset. AWS Fargate is a similar serverless solution, but better suited for batch-inference or models that require more resources. I deployed a custom Lambda Docker image using ECR (AWS Elastic Container Registry). Lambda Functions have a limit of 250MB for (read-only) code and files within the container, but it is configurable to use up to 10GB of storage. In the Docker container.

On EC2, I deployed the same model in a Flask app and deployed the endpoint with Docker-compose. I set up Docker-compose to use a volume so that the Docker container and the Lambda function have access to the same model without the need for managing multiple copies. I stored the model on EFS (Elastic File System), which is an NFS (Network File System) with incredible capabilities: durability, availability, and it can provision enough resources to store petabytes of data! I also set up a VPC endpoint so that Lambda (and EC2) can easily access S3 buckets (or other AWS services), which improves response time without sacrificing security.

**AWS Cloud and ML Ops Tools**
I used boto3, AWS CLI v3, and the AWS console to set up and monitor various services, including IAM Roles, VPCs, security groups with custom inbound and outbound rules, CloudWatch logs, alarms, and budgets. I created a Streamlit app as a simple front-end for generating POST requests from images. Streamlit is a quick solution for prototyping or even validating a new model making it easy to run real-time inference, as well as easy text formatting and beautiful visualizations. I implemented CI/CD with GitHub Actions and version control with GitHub (using a linear commit history). I trained the models in Google Colab with L4 and A100 GPUs, as available. Development was done with VScode, Google Colab, Jupyter Notebooks/Lab, and vim in Terminal. I also used Terminal to SSH as well as the Remote Explorer extension on VScode along with typical virtual environments and tools like pipreqs for setting up dependencies.


