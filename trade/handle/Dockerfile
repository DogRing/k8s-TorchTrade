FROM python:3.9-slim
WORKDIR /source
ENV PYTHONUNBUFFERED=1
COPY ./requirements.txt source/requirements.txt
RUN pip --no-cache-dir install -r source/requirements.txt
COPY app.py /source/app.py
COPY ttrade.py /source/ttrade.py
CMD ["python", "/source/app.py"]