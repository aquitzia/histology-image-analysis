# Use AWS image for AWS Lambda environment
FROM public.ecr.aws/lambda/python:3.12

# ${LAMBDA_TASK_ROOT} is an environment variable set by AWS Lambda
# to /var/task/, the default working directory.
RUN python -m pip install --upgrade pip setuptools wheel
COPY requirements.txt  .
RUN pip install --upgrade -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code. /var/task/ is read-only.
COPY lambda_predict.py ${LAMBDA_TASK_ROOT}

# Use CMD to run the handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_predict.lambda_handler" ]
