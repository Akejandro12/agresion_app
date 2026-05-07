FROM python:3.11-slim

# Dependencias del sistema para OpenCV y ffmpeg
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar e instalar dependencias primero (aprovecha caché de Docker)
COPY agresion_app/agresion_app/requirements.txt ./requirements.txt
COPY agresion_app_v2/requirements_extended.txt ./requirements_extended.txt

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r requirements_extended.txt

# Copiar todo el proyecto
COPY agresion_app/   ./agresion_app/
COPY agresion_app_v2/ ./agresion_app_v2/

# Directorio de trabajo donde arranca la app extendida
WORKDIR /app/agresion_app_v2

EXPOSE 5000

CMD ["python", "app_extended.py"]