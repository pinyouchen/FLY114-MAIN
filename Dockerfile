# 用官方 Python 3.11 精簡版
FROM python:3.11-slim

# 讓 print 立刻輸出、不buffer
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Taipei

# 安裝一些基本套件 & libgomp1（給 lightgbm / xgboost 用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 工作目錄
WORKDIR /app

# 先只複製 requirements，讓 Docker 可以 cache 這層
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 再把整個專案複製進來
COPY . .

# 預設執行你的腳本
CMD ["python", "test2_alldata_binary_update.py"]