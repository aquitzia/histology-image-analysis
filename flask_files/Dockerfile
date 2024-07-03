# Use the minimal Debian Bullseye (Debian 11), multi-architecture (works on ARM)
FROM python:3.11-slim-bullseye

WORKDIR /code
COPY requirements.txt .
# RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Use the application named app located in the server module
# gunicorn -b :$PORT server:app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
