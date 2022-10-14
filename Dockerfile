FROM python:3.9-slim

RUN apt update && apt install -y curl

COPY . .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["/run.sh"]