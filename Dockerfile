FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y python3-pip && apt-get clean
RUN pip3 install --no-cache-dir pipenv

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

WORKDIR /app

ENV PIPENV_CACHE_DIR /tmp/pipenv
COPY Pipfile* /app/
RUN pipenv install --system --deploy && rm -r $PIPENV_CACHE_DIR

RUN python3 -m spacy download en_core_web_sm

ENV QA_CACHE_PATH="/cache" QA_DATA_PATH="/data"
ENV PYTORCH_PRETRAINED_BERT_CACHE="${QA_CACHE_PATH}/bert" \
	QA_VECTOR_CACHE="${QA_CACHE_PATH}/vectors.pickle"

ADD . /app

EXPOSE 8765

CMD [ "/app/entrypoint.sh", "bootstrap" ]
