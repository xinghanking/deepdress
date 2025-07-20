FROM python:3.11.13
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && rm -fr /var/lib/apt/lists/*
RUN mkdir core/checkpoints
RUN gdown --fuzzy https://drive.google.com/file/d/1lZIISdSpssZwt9yVC0SuVISciWb3jCV3/view?usp=drive_link -O core/checkpoints/deepdress.pt
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt