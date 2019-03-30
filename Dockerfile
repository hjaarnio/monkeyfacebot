FROM python:3.6

# Install dependencies
RUN apt-get update

COPY requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip install -r requirements.txt

COPY app /app/app

CMD python app/server.py
