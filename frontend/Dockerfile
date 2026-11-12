# Build del frontend y servidor estático en una sola imagen.
# El contexto de build es la raíz del repositorio (ver docker-compose.yml).
FROM node:20-alpine AS build

WORKDIR /app

# Copiar primero los manifiestos permite reutilizar la capa de dependencias
# mientras package.json/package-lock.json no cambien.
COPY frontend/package*.json ./

# npm puede encontrarse con fallos DNS transitorios (EAI_AGAIN) en servidores.
# Los reintentos evitan que un fallo puntual invalide todo el despliegue.
# CRA 5 e i18next declaran rangos opcionales de TypeScript incompatibles;
# el proyecto no contiene código TypeScript, por lo que fijamos esa resolución.
RUN npm ci --legacy-peer-deps --no-audit --no-fund \
    --fetch-retries=5 \
    --fetch-retry-factor=2 \
    --fetch-retry-mintimeout=1000 \
    --fetch-retry-maxtimeout=120000

COPY frontend/ ./

RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
# La configuración necesita resolver el servicio backend de Compose.
COPY nginx/nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
