FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /source
COPY ./requirements.txt .
RUN pip --no-cache-dir install -r ./requirements.txt
COPY app.py /source/app.py
CMD ["python", "/source/app.py"]