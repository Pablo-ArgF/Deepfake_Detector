# backend/Dockerfile

# Imagen base de Python
FROM tensorflow/tensorflow:2.15.0-gpu

# Establece el directorio de trabajo
WORKDIR /app

# Instala las dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala las dependencias del backend y entrenamiento
COPY requirements.txt .
RUN pip install --ignore-installed -r requirements.txt

# Copia el código de la aplicación Flask, modelos y código de entrenamiento
COPY . .

# Create a directory for storing static image data
RUN mkdir -p /app/static/images/generated

# Expone el puerto que usa Flask
EXPOSE 5000

# Ejecuta el servidor Flask
CMD ["python", "app.py"]
