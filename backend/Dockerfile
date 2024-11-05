# backend/Dockerfile

# Imagen base de Python
FROM python:3.11.5-slim

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias del backend y entrenamiento
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copia el código de la aplicación Flask, modelos y código de entrenamiento
COPY . .

# Expone el puerto que usa Flask
EXPOSE 5000

# Ejecuta el servidor Flask
CMD ["python", "app.py"]
