ARG BASE=latest
FROM tensorflow/tensorflow:$BASE

WORKDIR /workarea

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /workarea

RUN mkdir -p /vol/run_artifacts
RUN mkdir -p /workarea/logs

ENTRYPOINT ["python"] 
