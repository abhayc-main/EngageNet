

<h1 align="center">EngageNet</h1>
<p align="center">
  Medición de participación de multitudes/audiencias que aprovecha cámaras cenitales con agrupación (clustering) y métricas euclidianas.
</p>
<p align="center">
  <a href="#installation"><strong>Instalación</strong></a> ·
  <a href="#usage"><strong>Desarrollo local</strong></a> ·
  <a href="#cli"><strong>Herramientas CLI</strong></a> ·
  <a href="#code-flow"><strong>Métricas/Flujo de código</strong></a>
</p>

<br/>

![Screenshot 2023-09-10 at 8 49 58 PM](https://github.com/abhayc-glitch/EngageNet/assets/78511893/60ad3ee4-78ce-409a-bd7a-7e96dd917f0d)

#### Mejora de las interacciones científicas en conferencias: Un enfoque novedoso utilizando estimación de pose con cámara cenital, agrupación y métricas euclidianas

#### Características
**Estimación de pose con cámara cenital:** Utilizando la estimación de pose con cámara cenital, el sistema rastrea y analiza efectivamente la dinámica de las discusiones científicas, proporcionando una visión sin precedentes de los patrones de interacción.

**Análisis de interacciones en vivo:** Cámaras estratégicamente colocadas sobre el recinto de la conferencia capturan video en vivo de los investigadores durante momentos clave de sus discusiones, ofreciendo una visión en tiempo real del corazón de la conferencia.

**Mejora de conferencias basada en datos:** Más allá de simplemente capturar interacciones, este proyecto permite a los organizadores utilizar los datos adquiridos para tomar decisiones informadas, optimizar la disposición de la conferencia y mejorar la experiencia general de los asistentes.

**Puntuación avanzada usando métricas euclidianas:** Al evaluar parámetros como la proximidad, la duración y la frecuencia de las interacciones, calcula una 'puntuación de interacción' integral que ofrece conocimientos cuantificables sobre la efectividad de los intercambios entre investigadores.

## Cómo comenzar

### Instalación

#### Modelos
1. Obtén los modelos entrenados (archivos .pt de YOLO) -> envíame un correo a abhichebium@gmail.com para obtener acceso
2. Coloca los modelos descargados en la carpeta `./engagenet/models/`
3. Asegúrate de que los nombres sean `best.pt` y `angle_best.pt`

#### Interfaz de usuario (UI)
1. Clona todo el repositorio y navega a la carpeta `./client`
2. Asegúrate de tener `Node.js` y `npm/yarn` instalados.
3. Ejecuta `npm install` en el directorio raíz de `./client` para instalar todas las dependencias

#### Código de detección/Algoritmo
1. Navega a la carpeta `./engagenet`
2. Asegúrate de tener instalado `python 3.9.6`.
3. Activa un entorno virtual si lo prefieres
4. Ejecuta `pip install -r requirements.txt`

### Uso
###### En este punto, tu carpeta debería verse así
<img width="126" alt="image" src="https://github.com/abhayc-glitch/EngageNet/assets/78511893/3796a406-edd9-47bd-9a43-cc76e11aa714">


1. Recomiendo tener 3 ventanas de terminal diferentes: cliente, servidor de sockets y código de detección en Python
2. En la primera ventana, navega al directorio `./client` y ejecuta `npm run dev` para iniciar la aplicación `NextJS`
3. En la segunda ventana y en el mismo directorio, ejecuta `npm run start:socket`, esto iniciará el servidor de socketIO
4. Ahora navega al directorio `./engagenet` y ejecuta `python main.py`
> ⚠️: **Asegúrate de conectar tu webcam/cámara a través de USB**: Cambia la fuente de la cámara en `main.py` si tu cámara no es reconocida
5. Ahora arrastra la ventana de Python que aparece con la transmisión en vivo sobre la caja de la UI que dice place here
> Al final debería verse así
> 
![Screenshot 2023-09-10 at 8 27 50 PM](https://github.com/abhayc-glitch/EngageNet/assets/78511893/8b7a01a3-9fb6-4e21-90f8-29790208eb2c)


### CLI
Si deseas usar solo el CLI y no la UI, navega al directorio `./engagenet` y ejecuta `python cli.py`
> Debería terminar viéndose así
>
<img width="796" alt="image" src="https://github.com/abhayc-glitch/EngageNet/assets/78511893/7af4c46b-4ac2-45aa-9b9a-383e48690d90">


## Flujo de código

### Sistema de detección de cuadros en tiempo real

#### 1. Recopilación de datos:
- **Objetivo:** Recopilar y anotar un conjunto de datos para entrenar un modelo que evalúe la participación de la multitud y la salud de las interacciones.
- **Método:**
  - Recopila un conjunto de datos de aproximadamente 200 videos de multitudes desde una perspectiva de cámara cenital (top-down).
  - Anota los videos con etiquetas de referencia (ground truth) que indiquen niveles de participación o salud de la interacción.
  - Utiliza este conjunto de datos tanto para el cálculo del ángulo de la cabeza como para la detección de cabezas.

#### 2. Métricas para la detección (Resumen):
- **Rama de proximidad:** 
  - Calcula una puntuación de proximidad que indique qué tan cerca están los individuos unos de otros.
- **Rama de ángulo/cluster:** 
  - Determina la puntuación de participación basada en los ángulos de las cabezas y la agrupación espacial (clustering).
- **Rama de conteo de cabezas:** 
  - Cuenta el número de cabezas en la imagen para evaluar el tamaño de la multitud.
- **Puntuación general:** 
  - Combina las puntuaciones de las ramas anteriores para un análisis integral de la participación.

#### 3. Métricas para la detección - Enfoque detallado:
- **Orientación de la cabeza (50% de la puntuación total):** 
  - Cuenta las cabezas totales (N) y las cabezas orientadas hacia el grupo (F).
  - Puntuación = (F / N) * 100.
- **Análisis de proximidad (40% de la puntuación total):** 
  - Calcula las distancias entre las cabezas utilizando una matriz de distancias.
  - Identifica y puntúa los clusters de cabezas basándose en la proximidad.
  - Normaliza las puntuaciones entre 0 (proximidad mínima) y 1 (proximidad máxima).
- **Rama de conteo de cabezas (5% de la puntuación total):** 
  - Simplemente cuenta el número de cabezas en la escena.

#### 4. Clases de ángulo:
- Representación de los ángulos de orientación de la cabeza y sus direcciones correspondientes:

| Ángulo | Dirección  |
| ----- | ---------- |
| 0     | Abajo      |
| 45    | Abajo-Derecha |
| 90    | Derecha    |
| 135   | Arriba-Derecha |
| 180   | Arriba     |
| 225   | Arriba-Izquierda |
| 270   | Izquierda  |
| 315   | Abajo-Izquierda |

#### 5. Datos de ángulo:
- Distribución de archivos en diferentes clases de ángulo:

| Ángulo | Número de archivos |
| ----- | --------------- |
| 0     | 2870            |
| 45    | 2996            |
| 90    | 2806            |
| 135   | 2986            |
| 180   | 2832            |
| 225   | 2932            |
| 270   | 2800            |
| 315   | 2978            |
