FROM pytorch/torchserve:latest
WORKDIR /home/model-server

ENV MODEL_FOLDERS="./MODEL1|./MODEL2"
ENV MODEL_NAMES="model1|model2"
ENV MODEL_VERSIONS="1.0|1.1"
ENV TS_CONFIG_FILE=/home/model-server/config.properties

COPY --chmod=755 create_mar_files.sh .

CMD ["/home/model-server/create_mar_files.sh"]