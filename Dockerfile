# predict() in a container: Python 3.11, CPU only, nothing leaves the machine.
#   docker build -t echogesture .
#   docker run --rm -v "$PWD/clips:/clips" -v "$PWD/results:/out" echogesture /clips --out /out/predictions.jsonl
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY src/requirements.txt src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt
COPY src/ src/
ENV GLOG_minloglevel=3
ENTRYPOINT ["python", "src/main.py"]
