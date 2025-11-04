# Laboratorio 4: Señales electromiográficas EMG

## Integrantes
* Laura Valentina Velásquez Castiblanco (5600846)
* Carol Valentina Cruz Becerra (5600845)
* Carlos Felipe Moreno Guzmán (5600881)

## Objetivos:
* Aplicar el filtrado de señales continuas para procesar una señal electromiográfica (EMG).
* Detectar la aparición de fatiga muscular mediante el análisis espectral de contracciones musculares individuales.
* Comparar el comportamiento de una señal emulada y una señal real en términos de frecuencia media y mediana.
* Emplear herramientas computacionales para el procesamiento, segmentación y análisis de señales biomédicas.

## Diagramas de flujo

> ### Parte A
> ### Parte B
> ### Parte C

## Configuración inicial

Para la parte A, B y C se necesita el uso de librerias: 

```python
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter, firwin, lfilter
from google.colab import drive
import scipy.signal as sig
drive.mount('/content/drive')
```

* ``numpy (np)`` : Se usa para realizar operaciones matemáticas y manejar arreglos numéricos (vectores, matrices).
* ``matplotlib.pyplot (plt)``: Se utiliza para generar gráficas y visualizar datos.
* ``scipy.fft (fft, fftfreq)``: Se emplea para aplicar la transformada rápida de Fourier y analizar señales en el dominio de la frecuencia.
* ``scipy.signal (butter, filtfilt, find_peaks, savgol_filter, firwin, lfilter)``: Se usa para procesar señales, incluyendo diseño y aplicación de filtros, detección de picos y suavizado.
  * ``butter``: Diseña filtros Butterworth.
  * ``filtfilt``: Aplica el filtro sin alterar la fase.
  * ``find_peaks``: Identifica picos en una señal.
  * ``savgol_filter``: Suaviza señales.
  * ``firwin``: Diseña filtros FIR.
  * ``lfilter``: Aplica un filtro lineal.
* ``google.colab.drive``: Se usa para conectar el entorno de Google Colab con Google Drive y acceder a archivos almacenados allí.
* ``scipy.signal as sig``: Sirve para acceder a las funciones de procesamiento de señales de SciPy mediante un alias más corto.

# **Parte A**

# **Parte B**
### **1. Registro Electromiográfico**
Para esta parte del laboratorio se registró la actividad electromiográfica (EMG) del músculo bíceps braquial durante contracciones repetidas hasta la fatiga. Los electrodos se ubicaron sobre el vientre muscular, siguiendo el eje de las fibras, mientras que el electrodo de referencia se colocó en una prominencia ósea del codo (epicóndilo lateral o proceso del olécranon), garantizando una señal estable y libre de interferencias.

El bíceps braquial fue elegido por su fácil acceso, su papel definido en la flexión del codo y la posibilidad de generar contracciones controladas para observar la fatiga muscular. La referencia en el codo proporciona un punto de potencial cero adecuado debido a la mínima actividad muscular en esa zona.

Durante el protocolo, el voluntario realizó contracciones repetidas hasta la fatiga, definida como la incapacidad de mantener el nivel de fuerza requerido o continuar con las contracciones, permitiendo así, observar las características típicas de la fatiga muscular en la señal EMG.

### **2. Filtro Pasa Banda (20 - 450 Hz)**
>### 2.1. Importación de la Señal
Para la implementación en *Google Colab*, en primera instancia se cargaron los archivos ``.txt`` en la unidad de *Drive* para posteriormente ser leídos desde el entorno. La vinculación del *Drive* se realizó de la siguiente manera:

```python
from google.colab import drive
drive.mount('/content/drive')
```
La lectura de los datos electromiográficos se efectuó mediante la apertura directa del archivo de texto, convirtiendo cada línea en un valor numérico tipo *float*. El siguiente fragmento muestra la sintaxis utilizada:

```python
# Cargar señal
camino2 = '/content/drive/MyDrive/Colab Notebooks/Lab Procesamiento Digital de Señales/PDS - Lab 4/Paciente EMG5000.txt'

with open(camino2, 'r') as f:
    emg_paciente = [float(line) for line in f if line.strip()]
```
Este procedimiento permite importar los registros de la señal EMG almacenados en formato ``.txt``, generando un vector de datos listo para su posterior procesamiento.

>### 2.2. Aplicación Filtro Pasa Banda
Se aplicó un filtro pasa banda FIR (Finite Impulse Response) con ventana de Hamming entre ``20 y 450 Hz`` para eliminar componentes no relacionadas con la actividad electromiográfica. Las frecuencias por debajo de 20 Hz corresponden a ruido de movimiento, interferencia de línea base y artefactos mecánicos, mientras que las frecuencias por encima de 450 Hz se asocian a ruido de alta frecuencia y contenido no fisiológico. Al restringir el espectro a la banda donde se concentran las señales EMG producidas por la activación de unidades motoras, se mejora la relación señal-ruido y se asegura que los análisis posteriores representen de manera precisa la actividad muscular real.

Primero, se definieron los parámetros básicos del filtro, incluyendo la frecuencia de muestreo y las frecuencias de corte:

```python
# Frecuencia de muestreo
fs = 1000  # Hz

# Frecuencias de corte del filtro pasa-banda
f1 = 20    # Hz
f2 = 450   # Hz
```
Posteriormente, se calcularon los valores necesarios en frecuencia discreta y el orden del filtro, estableciendo finalmente un tamaño de ``N = 150``, lo que proporciona una respuesta estable y con adecuada atenuación fuera del rango de interés:

```python
# Parámetros discretos
w1 = 2 * np.pi * f1 / fs
w2 = 2 * np.pi * f2 / fs

# Orden del filtro
N = 150
```
Con estos parámetros, se construyó el filtro mediante la función ``firwin()`` de la librería ``scipy.signal``, utilizando la ventana Hamming, la cual reduce las oscilaciones indeseadas en la respuesta en frecuencia:

```python
# Aplicación FIR BP con Hamming
b = firwin(N, [f1, f2], pass_zero=False, fs=fs, window='hamming')
```
Finalmente, se aplicó el filtro a la señal EMG cargada previamente, obteniendo una versión limpia y lista para los procesos de segmentación y análisis:

```python
# Aplicación FIR a la señal EMG
emg_paciente_filtrada = lfilter(b, [1], emg_paciente)
```
>### 2.3. Visualización de la señal EMG filtrada
Una vez aplicado el filtro pasa banda, se procedió a representar gráficamente la señal resultante para verificar visualmente la efectividad del proceso de filtrado. El siguiente fragmento de código muestra la rutina utilizada para generar la gráfica:

```python
# Gráfica de Señal Filtrada
plt.figure(figsize=(10, 4))
plt.plot(emg_paciente_filtrada, color='royalblue', linewidth=1)
plt.title('Señal EMG Filtrada')
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.xlim(50000, 80000)
plt.ylim(-0.25, 0.25)
plt.show()
```
En la gráfica obtenida se observa la señal electromiográfica del paciente tras el filtrado en el rango de 20 a 450 Hz. La forma de onda presenta una distribución más limpia y definida, con menor presencia de ruido de baja frecuencia y oscilaciones de alta frecuencia. Este resultado confirma que el filtro FIR con ventana de Hamming logró suprimir eficazmente los artefactos de movimiento y las interferencias eléctricas, permitiendo resaltar únicamente la actividad muscular real.

La ventana mostrada (entre las muestras 50 000 y 80 000) evidencia de manera clara los periodos de activación y reposo del músculo, conservando la variabilidad natural de la señal EMG sin distorsionar su amplitud. Esta etapa de preprocesamiento es fundamental, debido a que proporciona una señal óptima para las fases posteriores de segmentación, detección de contracciones y análisis espectral asociados al estudio de la fatiga muscular.

<img width="1000" height="390" alt="image" src="https://github.com/user-attachments/assets/32a5197c-3e44-49d3-8b13-8ffe56f048d1" />


# **Parte C**
