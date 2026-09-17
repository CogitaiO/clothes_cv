FROM python:3.14.7-slim-bookworm
WORKDIR /clothes_cv
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
