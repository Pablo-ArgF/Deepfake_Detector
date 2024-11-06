# DeepFake Detector

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/fcd8b946-2d97-46ff-8c02-81da5be11414" alt="Icon" width="80" height="80"></td>
    <td>
      <p>Continuación de trabajo Fin de Grado<br>
      Cursado en 2023-2024, obteniendo la calificación de 10 con Matrícula de honor<br>
      Ingeniería Informática del Software<br>
      Universidad de Oviedo</p>
    </td>
  </tr>
</table>

### Objetivo

El objetivo principal de este proyecto es desarrollar una herramienta de inteligencia artificial capaz de detectar y prevenir la manipulación de contenido multimedia mediante la identificación de deepfakes. Los deepfakes son vídeos en los que se muestran imágenes falsas, habitualmente del rostro de una persona, generadas mediante técnicas avanzadas de inteligencia artificial y aprendizaje profundo.

En la era digital actual, la capacidad de las IAs generativas para crear contenido extremadamente realista ha aumentado significativamente. Esto plantea serios desafíos en términos de seguridad y veracidad de la información. La detección de deepfakes es crucial para proteger la integridad de la información y prevenir el uso malintencionado de estas tecnologías, que pueden ser utilizadas para difundir desinformación, cometer fraudes, o dañar la reputación de individuos.

El objetivo final de este proyecto es el estudio, análisis y entrenamiento de posibles modelos de predicción basados en Inteligencia Artificial, principalmente redes neuronales convolucionales, que detecten de forma lo más precisa posible vídeos manipulados de rostros humanos. Esto incluye:

- **Investigación y análisis**: Estudiar las técnicas actuales de generación de deepfakes y los métodos existentes para su detección.
- **Desarrollo de modelos**: Crear y entrenar modelos de inteligencia artificial que puedan identificar características distintivas de los deepfakes.
- **Evaluación y mejora**: Probar la efectividad de los modelos desarrollados y mejorar su precisión y robustez mediante iteraciones sucesivas.
- **Implementación práctica**: Desarrollar una herramienta accesible y fácil de usar que pueda ser utilizada por individuos y organizaciones para verificar la autenticidad de los vídeos.

### ¿Qué es un Deepfake?

Un deepfake es una técnica de síntesis de imágenes y vídeos que utiliza algoritmos de inteligencia artificial, específicamente redes neuronales profundas, para crear contenido audiovisual falso pero extremadamente realista. El término "deepfake" proviene de la combinación de "deep learning" (aprendizaje profundo) y "fake" (falso).

Los deepfakes se generan mediante el uso de algoritmos de aprendizaje profundo, como las redes generativas antagónicas (GANs), que pueden aprender y replicar patrones complejos en datos visuales y auditivos. Estos algoritmos pueden tomar imágenes o vídeos de una persona y superponerlos en otro vídeo, creando la ilusión de que esa persona está diciendo o haciendo algo que en realidad nunca ocurrió.

### Aplicaciones y Riesgos

**Aplicaciones legítimas**:
- **Cine y entretenimiento**: Para crear efectos visuales impresionantes y realistas.
- **Educación y formación**: Para generar simulaciones y escenarios de aprendizaje interactivos.
- **Restauración de medios**: Para restaurar y mejorar la calidad de vídeos antiguos o dañados.

**Riesgos y usos malintencionados**:
- **Desinformación**: Crear y difundir noticias falsas o propaganda.
- **Fraude y extorsión**: Utilizar deepfakes para engañar a personas o cometer delitos.
- **Daño a la reputación**: Crear vídeos falsos para dañar la imagen pública de individuos.

La detección de deepfakes es, por tanto, una tarea crítica para mantener la confianza en los medios digitales y proteger a las personas y organizaciones de los posibles abusos de esta tecnología.

## Instalación

Para poder ejecutar este proyecto, es necesario tener **Docker** instalado en tu máquina.

### Requisitos previos

1. **Docker**: Si no tienes Docker instalado, puedes descargarlo desde [aquí](https://www.docker.com/get-started) y seguir las instrucciones de instalación para tu sistema operativo.

### Pasos para ejecutar el proyecto

1. Clona este repositorio en tu máquina:
   ```bash
   git clone <url_del_repositorio>
   cd <directorio_del_repositorio>
   ```
2. En el directorio raíz del repositorio, ejecuta el siguiente comando para construir y levantar los contenedores de Docker:
   ```bash
   docker-compose up --build
   ```
3. Una vez que el contenedor esté corriendo, la aplicación estará disponible y podrás interactuar con ella accediendo a http://localhost





