FROM python:3.12


COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
RUN rm requirements.txt

