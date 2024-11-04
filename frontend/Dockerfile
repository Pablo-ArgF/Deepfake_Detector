# frontend/Dockerfile

# Imagen base de Node.js para construir el frontend
FROM node:14-alpine AS build

# Establece el directorio de trabajo
WORKDIR /app

# Copia el package.json y package-lock.json e instala dependencias
COPY package*.json ./
RUN npm install

# Copia el resto de los archivos del frontend
COPY . .

# Compila el frontend para producción
RUN npm run build

# Servidor Nginx para servir el frontend compilado
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
