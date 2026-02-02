# Python 3.9 tabanlı imaj
FROM python:3.9-slim

# Sistem bağımlılıklarını yükle (dlib ve opencv için şart)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Önce gereksinimleri kopyala ve kur (cache avantajı için)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Veritabanı ve logların kalıcı olması için klasörü oluştur
RUN mkdir -p data

# Ana uygulamayı çalıştır
CMD ["python", "main.py"]