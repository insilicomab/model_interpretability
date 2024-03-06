FROM python:3.11.6-slim

WORKDIR /opt

ADD requirements.txt /opt/
RUN pip install -r requirements.txt

CMD ["tail", "-f", "/dev/null"]