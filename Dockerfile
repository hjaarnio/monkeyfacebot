FROM python:3.6

# Install dependencies
RUN apt-get update && apt-get install -y cmake

RUN pip install dlib==19.17.0

# Install requirements
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Install newer scikit-image
RUN apt-get install -y python3-matplotlib python3-numpy python3-pil python3-scipy python3-tk
RUN apt-get install -y build-essential cython3
RUN git clone https://github.com/scikit-image/scikit-image.git
RUN cd scikit-image && pip3 install -e .


COPY app /app/app

CMD python app/server.py
