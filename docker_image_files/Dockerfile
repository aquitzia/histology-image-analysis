# Use AWS image for AWS Lambda environment
FROM public.ecr.aws/lambda/python:3.12

# ${LAMBDA_TASK_ROOT} is an environment variable set by AWS Lambda
# to /var/task/, the default working directory.
RUN python -m pip install --upgrade pip
RUN pip3 install --upgrade setuptools
COPY requirements.txt  .
RUN pip3 install --upgrade -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code. /var/task/ is read-only.
COPY lambda_function_APIGateway.py ${LAMBDA_TASK_ROOT}
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_home

# Use CMD to run the handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function_APIGateway.lambda_handler" ]
