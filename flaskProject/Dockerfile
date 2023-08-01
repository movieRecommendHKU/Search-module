FROM python:3.9.17
MAINTAINER Huang
WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
COPY . .

ENV FLASK_APP app

CMD ["python","-m","flask","run","--host=0.0.0.0","--port=5000"]
