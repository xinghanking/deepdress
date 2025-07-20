FROM python:3.11.13
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && rm -fr /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt