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

Para esta parte del laboratorio se registró la actividad electromiográfica (EMG) del músculo bíceps braquial durante contracciones repetidas hasta la fatiga. Los electrodos se ubicaron sobre el vientre muscular, siguiendo el eje de las fibras, mientras que el electrodo de referencia se colocó en una prominencia ósea del codo (epicóndilo lateral o proceso del olécranon), garantizando una señal estable y libre de interferencias.

El bíceps braquial fue elegido por su fácil acceso, su papel definido en la flexión del codo y la posibilidad de generar contracciones controladas para observar la fatiga muscular. La referencia en el codo proporciona un punto de potencial cero adecuado debido a la mínima actividad muscular en esa zona.

Durante el protocolo, el voluntario realizó contracciones repetidas hasta la fatiga, definida como la incapacidad de mantener el nivel de fuerza requerido o continuar con las contracciones, permitiendo así, observar las características típicas de la fatiga muscular en la señal EMG.

# **Parte C**
