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
### 1. Registro Electromiográfico
Para esta parte del laboratorio se registró la actividad electromiográfica (EMG) del músculo bíceps braquial durante contracciones repetidas hasta la fatiga. Los electrodos se ubicaron sobre el vientre muscular, siguiendo el eje de las fibras, mientras que el electrodo de referencia se colocó en una prominencia ósea del codo (epicóndilo lateral o proceso del olécranon), garantizando una señal estable y libre de interferencias.

El bíceps braquial fue elegido por su fácil acceso, su papel definido en la flexión del codo y la posibilidad de generar contracciones controladas para observar la fatiga muscular. La referencia en el codo proporciona un punto de potencial cero adecuado debido a la mínima actividad muscular en esa zona.

Durante el protocolo, el voluntario realizó contracciones repetidas hasta la fatiga, definida como la incapacidad de mantener el nivel de fuerza requerido o continuar con las contracciones, permitiendo así, observar las características típicas de la fatiga muscular en la señal EMG.

Para la adquisición de los datos se utilizó un sistema de adquisición de datos (DAQ), junto con un código de captura implementado en tiempo real que permitió el registro continuo de la señal electromiográfica durante todo el protocolo experimental. Este sistema garantizó la digitalización y almacenamiento de la señal EMG con una frecuencia de muestreo adecuada para preservar las características relevantes de la actividad muscular y facilitar el análisis posterior.

### 2. Filtro Pasa Banda (20 - 450 Hz)
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

### 3. Segmentación de la Señal 
>### 3.1. Definición de parámetros generales

```python
FS = 1000.0
MIN_DUR_S = 0.02
SG_WIN = 101
SG_POLY = 3
DERIV_STD_MULT = 3.0
MERGE_GAP_S = 0.15
P2P_MULT_NOISE = 4.0
RMS_MULT_NOISE = 2.0
```

En esta sección se establecen los parámetros globales que controlan la detección de contracciones musculares. Se define la frecuencia de muestreo ``(FS = 1000 Hz)``, la duración mínima de una contracción ``(20 ms)`` y los criterios de suavizado mediante el filtro de *Savitzky–Golay* ``(SG_WIN, SG_POLY)``. Los multiplicadores de desviación estándar ``(DERIV_STD_MULT)`` y de energía ``(P2P_MULT_NOISE, RMS_MULT_NOISE)`` permiten ajustar la sensibilidad del algoritmo, asegurando que solo se detecten eventos con amplitud y duración características de verdaderas contracciones musculares.

>### 3.2. Carga de señal y preprocesamiento

La señal filtrada previamente se copia en una nueva variable para el análisis. Posteriormente, se aplica un suavizado Savitzky–Golay, que reduce pequeñas oscilaciones y ruido sin alterar la forma general de la señal.
Luego, se calcula la derivada temporal, la cual resalta los cambios bruscos de amplitud asociados con el inicio y el final de cada contracción muscular.

```python
emg = emg_paciente_filtrada.copy()
t = np.arange(len(emg)) / FS
smoothed = savgol_filter(emg, SG_WIN, SG_POLY)
deriv = np.gradient(smoothed) * FS
```

>### 3.3. Detección inicial de eventos
Se establecieron umbrales positivo y negativo sobre la derivada de la señal, calculados a partir de la media y desviación estándar de la misma. Los puntos donde la derivada supera el umbral positivo se identifican como inicios de contracción muscular (onsets), mientras que aquellos que descienden por debajo del umbral negativo corresponden a los finales de contracción (offsets). Este criterio basado en la variación temporal de la señal permite delimitar de manera objetiva los segmentos de actividad muscular, diferenciándolos de los periodos de reposo y facilitando la segmentación automática de las contracciones registradas. El siguiente fragmento muestra la sintaxis utilizada:

```python
mu = np.mean(deriv)
sigma = np.std(deriv)
thr_pos = mu + DERIV_STD_MULT * sigma
thr_neg = mu - DERIV_STD_MULT * sigma
```

>### 3.4. Emparejamiento y consolidación de contracciones

```python
segments = []
i = 0
while i < len(onsets_idx) and i < len(offsets_idx):
    s = onsets_idx[i]
    offs_after = offsets_idx[offsets_idx > s]
    if len(offs_after) == 0:
        break
    e = offs_after[0]
    if e - s >= min_samples:
        segments.append((s, e))
    i += 1
```
En este bloque se construye la lista inicial de segmentos de activación muscular a partir de los índices de inicio ``(onsets_idx)`` y final ``(offsets_idx)`` detectados previamente. Para cada evento, el código busca el primer punto de finalización posterior al inicio y calcula la duración del intervalo en número de muestras. Solo aquellos tramos cuya longitud supera un umbral mínimo ``(min_samples)`` son considerados válidos, con el fin de eliminar artefactos breves o fluctuaciones no representativas. El resultado es una lista segments que agrupa las posibles contracciones registradas en la señal EMG.

El siguiente bloque de código implementa la función encargada de fusionar los segmentos de activación que se encuentran separados por intervalos muy cortos, unificando así contracciones continuas que pudieron haber sido divididas por pausas breves en la señal:

```python
# ---------- Merge ----------
def merge_segments(segs, gap_samples):
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x[0])
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        prev = merged[-1]
        if s <= prev[1] + gap_samples:
            prev[1] = max(prev[1], e)
        else:
            merged.append([s, e])
    return [(int(a), int(b)) for a, b in merged]

segs_merged = merge_segments(segments, int(round(MERGE_GAP_S * FS)))
```

En este fragmento se ejecuta la función ``merge_segments``, que fusiona los intervalos cercanos entre sí dentro de una separación máxima definida por ``MERGE_GAP_S``. Este proceso evita que una misma contracción muscular —que puede presentar leves interrupciones en la señal— sea contabilizada como múltiples eventos separados. De esta manera, los segmentos contiguos que se encuentran a una distancia menor al umbral especificado ``(gap_samples)`` se combinan en uno solo, obteniendo un conjunto de intervalos unificados ``(segs_merged)`` que representan contracciones completas y continuas.

A continuación, se calcula una estimación robusta del nivel de ruido de la señal, con el fin de establecer los umbrales de detección adecuados:

```python
residual = emg - savgol_filter(emg, 1001 if len(emg)>1001 else SG_WIN, SG_POLY)
mad = np.median(np.abs(residual - np.median(residual)))
noise_std = mad / 0.6745
p2p_thresh = P2P_MULT_NOISE * noise_std
rms_thresh = RMS_MULT_NOISE * noise_std
```
Este bloque calcula el residuo entre la señal original y su versión suavizada mediante el filtro de *Savitzky–Golay*, estimando la desviación estándar del ruido ``(noise_std)`` a partir de la mediana de las desviaciones absolutas (MAD). Con ello se definen los umbrales mínimos de amplitud pico a pico ``(p2p_thresh)`` y de energía RMS ``(rms_thresh)``, que sirven como referencia para descartar eventos de baja intensidad no asociados a verdadera actividad muscular.

>### 3.5. Filtrado final de contracciones válidas

```python
filtered_segs = []
for s, e in segs_merged:
    seg_y = emg[s:e]
    seg_ptp = seg_y.max() - seg_y.min()
    seg_rms = np.sqrt(np.mean(seg_y**2))
    dur = (e - s) / FS
    # condiciones: duración y energía/ amplitud
    if dur >= MIN_DUR_S and seg_ptp >= p2p_thresh and seg_rms >= rms_thresh:
        filtered_segs.append((s, e))
```

Dentro del ciclo ``for s, e in segs_merged``, se analiza cada intervalo fusionado extrayendo la porción correspondiente de la señal EMG y calculando sus características principales: amplitud pico a pico (diferencia entre valores máximo y mínimo), valor RMS (energía media del segmento) y duración en segundos. Estos indicadores permiten evaluar la intensidad y extensión temporal de cada contracción.

Posteriormente, se validaron las contracciones mediante tres criterios de aceptación: (1) duración mínima suficiente para descartar artefactos transitorios, (2) amplitud pico a pico significativa que refleje activación muscular genuina y (3) energía RMS superior al nivel de ruido basal. Únicamente las contracciones que satisfacen simultáneamente estos requisitos se conservan como eventos válidos, garantizando la exclusión de falsas detecciones. El conjunto ``filtered_segs`` reúne así los intervalos correspondientes a activaciones musculares reales, proporcionando una base confiable para el análisis cuantitativo y temporal posterior.

>### 3.6. Visualización de resultados

En esta parte, se genera la visualización global de la señal EMG procesada, mostrando el trazado completo de la señal filtrada junto con las contracciones válidas resaltadas en color naranja. Cada zona sombreada corresponde a un intervalo identificado como actividad muscular efectiva, es decir, aquellos tramos que superaron simultáneamente los umbrales de duración, amplitud y energía establecidos durante el filtrado.

```python
plt.figure(figsize=(12,4))
plt.plot(t, emg, label='EMG filtrada', linewidth=0.8)
for s,e in filtered_segs:
    plt.axvspan(s/FS, e/FS, color='orange', alpha=0.3)
plt.title('Señal EMG — Detección final (filtrado de eventos pequeños)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud (V)')
plt.legend()
plt.tight_layout()
plt.show()
```
La gráfica permite realizar una validación visual del desempeño del algoritmo, evidenciando la correspondencia entre los segmentos detectados y las regiones de mayor densidad e intensidad de la señal electromiográfica. De esta forma, se confirma que el método logra una detección precisa y estable de los periodos activos, evitando la inclusión de picos espurios o ruido residual. El resultado es una representación clara y depurada del patrón de activación muscular a lo largo del tiempo.
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/d6245ea7-5425-4787-8c9d-c9606879a96f" />

Posteriormente, cada contracción muscular validada se representa de manera individual para analizar con mayor detalle su morfología y evolución temporal. A continuación se presenta el código implementado:

```python
for i,(s,e) in enumerate(filtered_segs):
    seg_t = t[s:e]
    seg_y = emg[s:e]
    sm_seg = savgol_filter(seg_y, window_length=11 if len(seg_y)>11 else len(seg_y)|1, polyorder=3)
    plt.figure(figsize=(6,2))
    plt.plot(seg_t, sm_seg)
    plt.title(f'Contracción {i+1} — {seg_t[0]:.3f}s a {seg_t[-1]:.3f}s')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (V)')
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.85, bottom=0.25)
    plt.margins(x=0.02, y=0.2)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.show()
```
El código recorre los intervalos previamente aceptados ``(filtered_segs)``, extrae el segmento correspondiente de la señal y aplica nuevamente un suavizado local mediante el *filtro de Savitzky–Golay*, con el fin de resaltar la forma general de la contracción y atenuar pequeñas oscilaciones de alta frecuencia. 

Cada gráfico muestra el intervalo temporal y la variación de amplitud de la contracción, lo que permite realizar un análisis comparativo entre eventos sucesivos.Dado que el algoritmo detectó un total de 116 contracciones válidas, para efectos de visualización se presentan únicamente las cinco primeras y las cinco últimas contracciones. Esta selección ilustra de manera representativa la evolución de la señal a lo largo del registro, destacando cómo hacia las etapas finales de fatiga se observan mayores amplitudes e irregularidades en la forma de onda, reflejo del incremento en la activación de unidades motoras y de la pérdida de estabilidad muscular.

>### Primeras 5 Contracciones
<img width="545" height="205" alt="image" src="https://github.com/user-attachments/assets/f44cf6dc-808e-45a6-a8fc-471a9d612510" />
<img width="552" height="205" alt="image" src="https://github.com/user-attachments/assets/9683e125-1e83-41fb-a2eb-24d055ae494c" />
<img width="527" height="205" alt="image" src="https://github.com/user-attachments/assets/35e00e7e-7de2-4b17-afac-08703f13132e" />
<img width="527" height="205" alt="image" src="https://github.com/user-attachments/assets/a801c7a7-87e1-44ea-8a87-44c783298534" />
<img width="529" height="205" alt="image" src="https://github.com/user-attachments/assets/3bbd1d28-d840-4fb3-b025-f25aa075dd80" />

>### Últimas 5 Contracciones
<img width="545" height="205" alt="image" src="https://github.com/user-attachments/assets/8bddfbbf-299c-4caf-b573-dcbf843a4a80" />
<img width="527" height="205" alt="image" src="https://github.com/user-attachments/assets/f15aa274-5fd5-46a0-a169-5679fe25644c" />
<img width="540" height="205" alt="image" src="https://github.com/user-attachments/assets/fc1ad60f-db6b-48ca-bdcf-f4f25cf8a4a4" />
<img width="537" height="205" alt="image" src="https://github.com/user-attachments/assets/17e6d0c3-eccf-48d8-94a2-fc596a2e4f80" />
<img width="540" height="205" alt="image" src="https://github.com/user-attachments/assets/e7d0bfd7-dac5-4151-8cef-d3ea41faadef" />

>### Análisis

En las primeras cinco contracciones (``60–66 s`` aproximadamente), la señal EMG exhibe mayor estabilidad temporal y menor variabilidad en amplitud. Las formas de onda presentan morfología definida con transiciones suaves, reflejando una actividad muscular controlada en ausencia de fatiga significativa. Las amplitudes pico a pico son moderadas y relativamente simétricas respecto al eje de referencia, mientras que la densidad de oscilaciones es reducida, sugiriendo contracciones eficientes con bajo contenido de ruido fisiológico.

Por el contrario, las últimas cinco contracciones (``266–270 s`` aproximadamente) muestran marcada irregularidad morfológica. Se evidencian picos más abruptos, fluctuaciones asimétricas y un incremento considerable en la amplitud de la señal, particularmente en las contracciones 112 y 115. Este patrón es consistente con los cambios electromiográficos característicos de la fatiga muscular, donde el aumento de amplitud y la pérdida de consistencia temporal se asocian con la desincronización progresiva de las unidades motoras y el reclutamiento de fibras musculares adicionales para compensar la pérdida de fuerza.

Adicionalmente, la dispersión en la morfología de las últimas contracciones revela una mayor variabilidad inter-evento, atribuible tanto a fluctuaciones en la fuerza aplicada como a la degradación de la eficiencia neuromuscular conforme avanza el protocolo de fatiga. Esta evolución en las características de la señal confirma la transición desde un estado de contracción óptima hacia un estado de fatiga muscular manifiesta. En conjunto, este procedimiento ofrece una visión detallada del comportamiento electromiográfico durante el desarrollo de la fatiga, complementando los análisis numéricos realizados previamente.

### 4. Cálculo de Frecuencia Media y Mediana por Contracción 

La  Frecuencia Media y Mediana representan la distribución de energía en el dominio de la frecuencia: la frecuencia media indica el centro de gravedad del espectro de potencia, mientras que la frecuencia mediana corresponde al punto que divide la energía espectral acumulada en dos mitades iguales. Ambos parámetros son indicadores clave del estado fisiológico del músculo, debido a que su desplazamiento hacia frecuencias más bajas suele asociarse con la fatiga muscular o con una reducción en la velocidad de conducción de las fibras.

El siguiente fragmento de código calcula la frecuencia media y frecuencia mediana de cada contracción muscular detectada.

```python
f_mean_list = []
f_median_list = []

for i, (s, e) in enumerate(filtered_segs):
    seg_y = emg[s:e]

    N = len(seg_y)
    if N < 10:
        continue  # descarta segmentos muy cortos

    Y = np.fft.rfft(seg_y)
    Pxx = np.abs(Y)**2 / N
    f = np.fft.rfftfreq(N, 1 / FS)

    f_mean = np.sum(f * Pxx) / np.sum(Pxx)
    cumsum = np.cumsum(Pxx)
    f_median = f[np.where(cumsum >= cumsum[-1] / 2)[0][0]]

    f_mean_list.append(f_mean)
    f_median_list.append(f_median)
```
En el código, se itera sobre cada contracción identificada en la lista filtered_segs, extrayendo su correspondiente porción de la señal ``(seg_y)``. Para cada segmento se calcula la Transformada Rápida de Fourier (FFT), a partir de la cual se obtiene el espectro de potencia ``(Pxx = |Y|² / N)``, que refleja la energía distribuida a lo largo de las distintas frecuencias. A partir de este espectro se determina la frecuencia media mediante un promedio ponderado ``(np.sum(f * Pxx) / np.sum(Pxx))``, y la frecuencia mediana se calcula identificando el punto en el que la energía acumulada alcanza la mitad del total ``(np.cumsum(Pxx))``. Los valores resultantes se almacenan en las listas ``f_mean_list y f_median_list``, respectivamente, permitiendo analizar la evolución de ambas métricas a lo largo del registro.

> [!NOTE]
> Dado que el registro completo contiene un total de 116 contracciones musculares detectadas, se seleccionaron únicamente las cinco primeras y las cinco últimas para su presentación y análisis comparativo. Esta selección representa de manera sintética la evolución temporal del comportamiento electromiográfico, evitando una sobrecarga visual de información.

>### Tabla Frecuencia Media y Mediana

| Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz) |
| :---------: | :-------------------: | :---------------------: |
|      1      |         123.67        |          88.24          |
|      2      |         116.40        |          57.54          |
|      3      |         106.56        |          57.92          |
|      4      |         124.84        |          71.77          |
|      5      |         100.12        |          47.77          |
|     ...     |          ...          |           ...           |
|     112     |         81.25         |          34.25          |
|     113     |         110.54        |          61.81          |
|     114     |         75.18         |          38.01          |
|     115     |         77.27         |          28.17          |
|     116     |         93.97         |          54.55          |

>### Cálculo y Gráfica Transformada de Fourier
>### Primeras 5 Contracciones
<img width="581" height="190" alt="image" src="https://github.com/user-attachments/assets/376e2e7e-cd9e-404c-977b-55660ed9313a" />
<img width="589" height="190" alt="image" src="https://github.com/user-attachments/assets/205b330e-df1a-4bae-aee9-194cb4b5ad51" />
<img width="598" height="190" alt="image" src="https://github.com/user-attachments/assets/3490fa6a-0386-4a4f-9d65-c909b54d8987" />
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/ec1a69bd-e2a0-4b84-88ad-4e61e993be83" />
<img width="572" height="190" alt="image" src="https://github.com/user-attachments/assets/887d463e-2d9f-40ca-886d-5e8e0d0dcec4" />

>### Últimas 5 Contracciones
<img width="572" height="190" alt="image" src="https://github.com/user-attachments/assets/60519fcc-5bb5-441a-a210-2aae19a6bf00" />
<img width="572" height="190" alt="image" src="https://github.com/user-attachments/assets/9ef4d1c2-6e79-4983-89ef-3e5077314d61" />
<img width="589" height="190" alt="image" src="https://github.com/user-attachments/assets/834efe64-a1e4-4450-9125-5669a44b0d33" />
<img width="581" height="190" alt="image" src="https://github.com/user-attachments/assets/a98b524f-c1c3-436f-96a4-3566f1585349" />
<img width="581" height="190" alt="image" src="https://github.com/user-attachments/assets/38e8ec74-1d4b-4e23-806d-387f414c7331" />



>### Análisis

A partir del cálculo de la frecuencia media y mediana para las 116 contracciones detectadas, se observa una variabilidad significativa en la distribución de energía espectral entre los distintos eventos musculares. Las primeras contracciones muestran valores más elevados de frecuencia media (en torno a ``120–140 Hz``) y mediana (entre ``80–100 Hz``), lo cual refleja una mayor actividad de fibras rápidas y una eficiente conducción de los potenciales de acción musculares.

Sin embargo, conforme avanza el registro, ambas frecuencias tienden a disminuir progresivamente, alcanzando valores medios más bajos (entre ``80–100 Hz`` para la frecuencia media y 40–60 Hz para la mediana). Este descenso es característico de la aparición de fatiga muscular, fenómeno asociado a la reducción en la velocidad de conducción de las fibras, al aumento de la sincronización de unidades motoras y al predominio de fibras lentas de tipo I durante los periodos prolongados de esfuerzo.

La diferencia entre las frecuencias media y mediana también se amplía en algunos tramos, lo que indica una redistribución asimétrica de la energía espectral hacia componentes de menor frecuencia, evidenciando un desplazamiento del contenido espectral típico de la fatiga. Así, la disminución sostenida de la frecuencia media y mediana constituye un marcador objetivo del proceso de fatiga, coherente con los patrones esperados en señales electromiográficas durante ejercicios de contracción repetitiva.

### 4. Tendencia de la Frecuencia Media y Mediana 
El siguiente bloque de código realiza la representación gráfica de la tendencia de la frecuencia media y mediana a lo largo de las contracciones musculares detectadas. Estos indicadores permiten observar la evolución espectral de la señal EMG, particularmente útil para evaluar la presencia de fatiga muscular, fenómeno que se manifiesta como una disminución progresiva de la frecuencia durante repeticiones continuas.

```python
if len(f_mean_list) > 1:
    f_mean_arr = np.array(f_mean_list)
    f_median_arr = np.array(f_median_list)
    contracciones = np.arange(1, len(f_mean_arr) + 1)

    slope_mean, intercept_mean, _, _, _ = linregress(contracciones, f_mean_arr)
    slope_median, intercept_median, _, _, _ = linregress(contracciones, f_median_arr)

    tend_mean = slope_mean * contracciones + intercept_mean
    tend_median = slope_median * contracciones + intercept_median
```
En este bloque se convierten las listas ``f_mean_list`` y ``f_median_lis``t en arreglos numéricos (NumPy arrays) y se asocia cada valor con su respectiva contracción. Luego, mediante la función linregress, se calcula la pendiente (slope) y el intercepto de la recta de regresión lineal que modela la evolución de ambas frecuencias a lo largo del número de contracciones.

Las pendientes obtenidas (``slope_mean`` y ``slope_median``) permiten cuantificar la dirección y magnitud del cambio espectral. Una pendiente negativa indica una reducción de la frecuencia conforme avanzan las contracciones, lo que se interpreta como progresión de la fatiga muscular, asociada al reclutamiento de fibras más lentas y a la disminución de la velocidad de conducción.
  
```python
    print(f"\nPendiente frecuencia media: {slope_mean:.3f} Hz/contracción")
    print(f"Pendiente frecuencia mediana: {slope_median:.3f} Hz/contracción")

    if slope_mean < 0 and slope_median < 0:
        print("\nAmbas frecuencias disminuyen con el número de contracciones → indica progresión de la fatiga muscular.")
    elif slope_mean < 0 or slope_median < 0:
        print("\n Solo una frecuencia muestra tendencia decreciente → posible inicio de fatiga.")
    else:
        print("\n No se observa tendencia descendente clara → no hay evidencia de fatiga significativa.")
else:
    print("\n Solo se detectó una contracción — no se puede analizar tendencia de frecuencia.")
```
Gráficamente, el código genera una figura donde se representan las frecuencias media y mediana por contracción, junto con sus respectivas rectas de tendencia (líneas punteadas). Esta visualización facilita la interpretación de los cambios en el dominio de la frecuencia y proporciona evidencia visual del fenómeno de fatiga.

Finalmente, según los valores de las pendientes, se imprime una interpretación automática del resultado:

* Si ambas pendientes son negativas, se confirma una tendencia decreciente clara, indicativa de fatiga progresiva.
* Si solo una pendiente es negativa, se sugiere un inicio de fatiga incipiente.
* Si ninguna presenta tendencia descendente, no se observa evidencia significativa de fatiga.
  
En la ejecución del bloque anterior, se obtiene una pendiente negativa tanto para la frecuencia media como para la frecuencia mediana:

* Pendiente frecuencia media: −0.165 Hz/contracción
* Pendiente frecuencia mediana: −0.205 Hz/contracción

Este comportamiento indica que, a medida que avanza el número de contracciones, ambas frecuencias disminuyen progresivamente, lo que se interpreta como una tendencia clara hacia la fatiga muscular. En términos fisiológicos, esta reducción refleja una disminución en la velocidad de conducción de las fibras musculares y una menor sincronización de las unidades motoras, efectos característicos del agotamiento progresivo durante esfuerzos repetitivos.

<img width="889" height="490" alt="image" src="https://github.com/user-attachments/assets/36a12fb8-c395-4d42-b296-dfd803334c7f" />

La visualización de las líneas de tendencia en la gráfica respalda este hallazgo, mostrando un descenso gradual de la frecuencia media y mediana a lo largo de la secuencia de contracciones, confirmando así el deterioro funcional asociado a la fatiga.

### 5. Relación entre los cambios de frecuencia y la fisiología de la fatiga muscular
Los resultados obtenidos muestran una **tendencia decreciente** tanto en la frecuencia media ``(−0.165 Hz/contracción)`` como en la frecuencia mediana ``(−0.205 Hz/contracción)`` a lo largo de las contracciones analizadas. Este comportamiento es un indicador clásico de la progresión de la fatiga muscular, fenómeno que se manifiesta en el dominio de la frecuencia del electromiograma (EMG) como un desplazamiento del espectro hacia valores más bajos.

Desde el punto de vista fisiológico, este descenso en la frecuencia está asociado principalmente a la **disminución de la velocidad** de conducción de las fibras musculares, consecuencia de la acumulación de metabolitos como el lactato, el ion hidrógeno (H⁺) y el fosfato inorgánico durante contracciones sostenidas o repetitivas. Estos subproductos alteran el equilibrio iónico y reducen la eficiencia del acoplamiento excitación-contracción, generando una transmisión más lenta del potencial de acción a lo largo de la fibra.

Asimismo, la fatiga produce una **pérdida gradual de la sincronización y el reclutamiento eficiente de las unidades motoras**, lo que contribuye a la **disminución de la potencia espectral** en las frecuencias altas. En conjunto, estos efectos se reflejan en la reducción progresiva de la frecuencia media y mediana del EMG, tal como se observó en el análisis, evidenciando un deterioro funcional en la capacidad contráctil del músculo.

En síntesis, la disminución de las componentes de alta frecuencia constituye un marcador cuantitativo de la fatiga muscular, permitiendo relacionar las variaciones del espectro de la señal EMG con los mecanismos fisiológicos subyacentes al agotamiento del tejido muscular.

# **Parte C**
