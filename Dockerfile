FROM python:3.12



RUN apt-get update && apt-get install -y vim


RUN mkdir /app
WORKDIR /app

COPY requirements.txt requirements.txt
COPY requirements_pyg.txt requirements_pyg.txt

RUN python -m pip install -r requirements.txt
RUN python -m pip install -r requirements_pyg.txt
RUN rm requirements.txt
RUN rm requirements_pyg.txt

