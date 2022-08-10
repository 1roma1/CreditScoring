FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY run.sh .
COPY model/credit_scoring .

RUN chmod +x run.sh

CMD ["./run.sh"]