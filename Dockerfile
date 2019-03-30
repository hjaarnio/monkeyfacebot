FROM python:3.6

# Install dependencies
RUN apt-get update && apt-get install -y cmake

COPY requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip install dlib==19.17.0
RUN pip install -r requirements.txt

COPY app /app/app

CMD python app/server.py
