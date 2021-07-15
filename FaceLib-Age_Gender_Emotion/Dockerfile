FROM python:3.8-slim

WORKDIR /app

# RUN apt-get update && apt-get install -y python3-opencv
#RUN apt-get install -y python3-opencv

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python","master_server.py"]