# RAG system made by SH4DOW
FROM python:3.13-slim

# 設定環境變數，避免 Python 輸出緩衝
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libomp-dev \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    && apt-get -f install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 複製你的專案到容器內
COPY . /app

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 開放 API Port (for human friendly，沒寫沒差)
EXPOSE 5000

# 預設啟動命令（啟動 API 模式）
CMD ["python", "app.py"]
