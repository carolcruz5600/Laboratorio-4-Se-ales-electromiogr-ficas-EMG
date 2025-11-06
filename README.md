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

## **1. Captura de la señal**

Se configura el generador de señales para una señal de EMG (electromiografía) para poder simular aproximademente cinco contracciones musculares. Este tipo de señal muestra la actividad eléctrica generada por las fibras musculares durante la contracción, observando las variaciones de amplitud y frecuencia asociadas al esfuerzo muscular. Se ajusto la frecuencia de 1 Hz en el generador. Posterior, se realiza la captura de la señal a través del DAQ, permitiendo obtener y almacenar la señal EMG simulada para poder realizar el análisis. 

## **2. Carga de la señal**

Se realiza la carga de la señal EMG que se encuentra en un archivo txt, dando parámetros básicos. Se da la ruta del archivo para poder cargar los valores numéricos cada uno por linea. Se abre el archivo y se lee todos los datos linea por linea, conviertiendo cada dato en un número decimal. El uso de la función ``if line.strip()`` asegura que el código ignore líneas vacías, los datos leidos se guardan en la definición ``emg_generador``, tomando toda la señal completa. 

En el código se calcula el número total de muestras usando ``len(emg_generador)`` y se almacena en la variable ``n_generador``, también se da la definición del tiempo usado de $10 segundos$ en ``duración_generador`` y una frecuencia de muestreo de $f_s=5000Hz$ en ``fs_generador``, indicando que se tomaron 5000 datos por segundo. Finalmente estos valores se muestran.

```python
emg_file_path = '/content/drive/MyDrive/Colab Notebooks/Lab Procesamiento Digital de Señales/Señal EMG5000.txt' 

with open(emg_file_path, 'r') as f:
    emg_generador = [float(line) for line in f if line.strip()]

n_generador = len(emg_generador)
duracion_generador = 10
fs_generador = 5000

print("Muestras:", n_generador)
print("Duración:", duracion_generador, "s")
print("fs:", fs_generador, "Hz")
```
## **3. Gráfica de la señal generada**

Con el código se muestra la gráfica de la señal EMG. Tomando los valores de la señal que se habían cargado anteriormente creando un eje ene le tiempo correspondiente con una medición de voltaje. Así, trazando la amplitud en función del tiempo obteniendo una forma de onda que refleja las contracciones simuladas por el generador. El código ajusta los límites del eje temporal y de amplitud para visualizar con mayor detalle los primeros segundos del registro (de 0 a 5 s) y evitar que la señal se vea saturada o muy comprimida.

Se observa la señal electromiográfica. En el eje horizontal se representa el tiempo (s), de 0 a 10 segundos y en el vertical se muestra la amplitud (v).

```python
# Gráfica de la señal del generador
t_generador = np.arange(n_generador)/fs_generador # Crea el vector de tiempo
plt.figure(figsize=(12,4))
plt.plot(t_generador,emg_generador)
plt.axis([0,duracion_generador,-10,10])
plt.xlim(0,5)
plt.ylim(-1.5,1.5)
plt.grid()
plt.title(f"EMG Generador")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (V)")
plt.show()
```

<img width="1254" height="393" alt="image" src="https://github.com/user-attachments/assets/fe132966-184e-4422-88ab-5352e25e8d78" />

## 4. Segmentación

>### **4.1 Parámetros Iniciales**

Se establecen los parámetros que controlan la detección de las contracciones en el EMG. Se define la frecuencia de muestreo $FS$, se fija una duración minima de contracción equivalente a 20ms evitando que se vean falsos positivos. Los valores definidos en ``SG_WIN=51`` y ``SG_POLY=3`` corresponden a la configuración del filtro **Savitzky-Golay**, empleado para suavizar la señal sin distorsionar la forma. Por su parte,`` DERIV_STD_MULT = 3.0`` establece el nivel de sensibilidad del detector, considerando todo cambio que supere tres veces la desviación estándar de la derivada. Finalmente, ``MERGE_GAP_S = 0.2`` define el tiempo máximo entre contracciones consecutivas para unirlas.

```python
FS = 5000
MIN_DUR_S = 0.02
SG_WIN = 51
SG_POLY = 3
DERIV_STD_MULT = 3.0
MERGE_GAP_S = 0.2
```

>### **4.2 Carga de la señal**

En este bloque se prepara la señal y se genera su eje temporal. La variable ``emg`` almacena la señal previamente cargada desde el archivo de texto, mientras que ``t = np.arange(len(emg)) / FS`` construye un vector de tiempo que asigna un instante en segundos a cada muestra. Este paso sirve para que la señal se logre representar correctamente en función del tiempo, calculando la duración exacta de cada evento detectado más adelante.

```python
# carga
emg = emg_generador
t = np.arange(len(emg)) / FS
```

>### **4.3 Suavizado y calculo de la derivada**

Aquí se aplica el filtro mencionado anteriormente a partir de la función ``savgol_filter()``, para poder reducir el ruido de la señal EMG sin alterar su formal. Este suavizado ayuda con la detección de cambios reales de amplitud. Luego, se calcula la derivada numérica con ``np.gradient(smoothed) * FS``, midiendo el cambio en el tiempo de la señal. En una señal EMG, las contracciones se manifiestan como variaciones rápidas en la amplitud, por lo que analizar la derivada permite localizar los momentos donde la actividad eléctrica del músculo aumenta o disminuye bruscamente.

```python
smoothed = savgol_filter(emg, SG_WIN, SG_POLY)
deriv = np.gradient(smoothed) * FS 
```
>### **4.4 Cálculo de umbrales para la detección**

Se determina los límites que definirán cuándo ocurre la contracción muscular. Se calcula la media ``(mu)`` y la desviación estándar ``(sigma)`` de la derivada, y a partir de estos valores se definen los umbrales superior ``(thr_pos)`` e inferior ``(thr_neg)``. Cuando la derivada supera el umbral positivo, sera el inicio de una contracción, mientras que cuando cae por debajo del umbral negativo, sera el final de la contracción.

```python
mu = np.mean(deriv)
sigma = np.std(deriv)
thr_pos = mu + DERIV_STD_MULT * sigma
thr_neg = mu - DERIV_STD_MULT * sigma
```
>### **4.5 Detección de onsets y offsets**

Los puntos donde la derivada es mayor que ``thr_pos`` se guardan en ``onsets_idx`` y guardando los inicios de las contracciones, mientras que los puntos donde la derivada es menor que ``thr_neg`` se almacenan en ``offsets_idx`` y guardan los finales de las contracciones.

```python
onsets_idx = np.where(deriv > thr_pos)[0]
offsets_idx = np.where(deriv < thr_neg)[0]
```
> ### **4.6 Emparejar inicios y finales de contracciones**

Luego se empareja los puntos de inicio y fin para formar cada uno de lossegmentos de contracción válidos. Para cada inicio detectado, busca el primer final que ocurre después, calculando la duración del evento. Si la diferencia entre ambos supera el tiempo mínimo establecido anteriormente en los parámetros iniciales ``(MIN_DUR_S)``, el segmento se guarda como una contracción real. Esto ayuda a la vez a evitar tomar contracciones muy cortas. El resultado es una variable que contienen las posiciones exactas, en número de muestra, de cada contracción identificada.

```python
min_samples = int(round(MIN_DUR_S*FS))
segments = []
i=0
while i < len(onsets_idx) and i < len(offsets_idx):
    s = onsets_idx[i]
    offs_after = offsets_idx[offsets_idx > s]
    if len(offs_after)==0: break
    e = offs_after[0]
    if e - s >= min_samples:
        segments.append((s,e))
    i += 1
```

> ### **4.7 Unión de contracciones**

Se incluye una función llamada merge_segments() que revisa la lista de contracciones y fusiona las que están muy próximas entre sí. Si el intervalo entre el final de una contracción y el inicio de la siguiente es menor que MERGE_GAP_S (0.2 s), se unen en una sola contracción.

```python
def merge_segments(segs, gap_samples):
    if not segs: return []
    segs = sorted(segs, key=lambda x: x[0])
    merged = [list(segs[0])]
    for s,e in segs[1:]:
        prev = merged[-1]
        if s <= prev[1] + gap_samples:
            prev[1] = max(prev[1], e)
        else:
            merged.append([s,e])
    return [(int(a),int(b)) for a,b in merged]

segs_merged = merge_segments(segments, int(round(MERGE_GAP_S*FS)))
```

> ### **4.8 Resultados detectados**

Para cada contracción se muestra el tiempo de inicio, el tiempo de fin y la duración total en segundos. Esto permite verificar que la detección se haya realizado de manera correcta.
```Python
print("Deriv-based events:", len(segs_merged))
for i,(s,e) in enumerate(segs_merged):
    print(i+1, f"{s/FS:.4f}s - {e/FS:.4f}s, dur={(e-s)/FS:.4f}s")
```

<p align="center"><b>Resultados</b></p>

<div align="center">
<pre>
Deriv-based events: 10

1 0.1232s - 0.6412s, dur=0.5180s
2 0.8694s - 1.6412s, dur=0.7718s
3 1.8694s - 2.6412s, dur=0.7718s
4 2.8694s - 3.6412s, dur=0.7718s
5 3.8694s - 4.6412s, dur=0.7718s
6 4.8692s - 5.6410s, dur=0.7718s
7 5.8692s - 6.6410s, dur=0.7718s
8 6.8692s - 7.6410s, dur=0.7718s
9 7.8692s - 8.6410s, dur=0.7718s
10 8.8692s - 9.6410s, dur=0.7718s
</pre>
</div>

>### **4.9 Gráficas**

>#### **4.9.1 Gráfica con contracciones detectadas**

Se genera una gráfica de la señal EMG completa en la que se pueden observar las contracciones detectadas. La señal se da en el eje tiempo-amplitud, y las regiones correspondientes a contracciones se muestran con color azul usando ``plt.axvspan()``. Esta representación permite identificar fácilmente cuándo y por cuánto tiempo se produce cada contracción dentro de la gráfica.

```python
plt.figure(figsize=(12,3))
plt.plot(t, emg, label='EMG')

for s,e in segs_merged: plt.axvspan(s/FS, e/FS, color='cyan', alpha=0.25)
plt.title('EMG — detec. por derivada (suavizada)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.tight_layout()
plt.show()
```
<p align="center"><b>Gráfica</b></p>

<img width="1254" height="393" alt="image" src="https://github.com/user-attachments/assets/97d9641c-38ef-4d7e-9d5d-18733a710c6e" />

>#### **4.9.2Gráficas de contracciones**

Para finalizar, el código extrae y grafica por separado cada contracción detectada. Para cada segmento, se obtiene la parte correspondiente de la señal original y se aplica nuevamente un suavizado para mejorar su apariencia visual. Luego, se muestra en una imagen independiente con su respectivo intervalo de tiempo y número de contracción. Esto permitio observar con mayor detalle la forma de cada contracción, su amplitud y duración, facilitando el análisis más preciso de las características electromiográficas.

```python
for i, (s, e) in enumerate(segs_merged):  # usa segs_rms_merged 
    seg_t = t[s:e]
    seg_y = emg[s:e]

    smoothed = savgol_filter(seg_y, window_length=11, polyorder=3)

    plt.figure(figsize=(6, 2))
    plt.plot(seg_t, smoothed, color='darkorange')
    plt.title(f'Contracción {i+1} — {seg_t[0]:.3f}s a {seg_t[-1]:.3f}s')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud (V)')
    plt.tight_layout()
    plt.show()
```

<p align="center"><b>Gráfica contracción 1</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/dac33dff-fa9c-4b58-aa43-10dd120375ab" />
</p>

<p align="center"><b>Gráfica contracción 2</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/871f57d6-f8ea-4d18-8b68-2f606575971e" />
</p>

<p align="center"><b>Gráfica contracción 3</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/9f9b9526-1096-44a7-bfe3-4b231f101365" />
</p>

<p align="center"><b>Gráfica contracción 4</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/48fd7d98-52ce-40f8-9113-d928d661e993" />
</p>

<p align="center"><b>Gráfica contracción 5</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/b73bfa91-e807-4c17-8217-13cb096ce28c" />
</p>

<p align="center"><b>Gráfica contracción 6</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/00572a87-00c8-4b15-8a45-e6c7bcafece9" />
</p>

<p align="center"><b>Gráfica contracción 7</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/a4c057ed-b3ca-4848-aae6-5f55bbaca522" />
</p>

<p align="center"><b>Gráfica contracción 8</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/7d2e3748-67ba-4162-9059-8550611abdeb" />
</p>

<p align="center"><b>Gráfica contracción 9</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/67b2f085-d597-40b5-b127-66845d377fbc" />
</p>

<p align="center"><b>Gráfica contracción 10</b></p>

<p align="center">
<img width="590" height="190" alt="image" src="https://github.com/user-attachments/assets/7f105f3d-0338-42d5-9dad-d076541b8e09" />
</p>

## **5. Transformada de fourier: Frecuencia Media y Frecuencia Mediana**

> ### **5.1 Inicialización de lista para los resultados**

El código comienza creando dos listas vacías llamadas ``f_mean_list`` y ``f_median_list``, en las cuales se guardaran los valores de frecuencia media y frecuencia mediana de cada contracción muscular detectada. Estas listas permiten conservar los resultados obtenidos, para poder realizar la comparación entre contracciones.

```python
f_mean_list = []
f_median_list = []
```

> ### **5.2 Toma de contracciones detectada**

Se inicia un ciclo ``for`` que recorre las contracciones identificadas de la señal dentro del arreglo ``segs_merged``. En cada acción, se extrae cada contracción usando los índices ``s`` (inicio) y ``e`` (fin). Se define ``N`` como la cantidad total de muestras de ese segmento y ``T`` como el periodo de muestreo. De esta forma, el código analiza la información de cada contracción muscular por separado.

```python
for i, (s, e) in enumerate(segs_merged):
    seg_y = emg[s:e]
    N = len(seg_y)
    T = 1 / FS
```

> ### **5.3 Vectores e inicio transformada de Fourier**

En esta parte se crean los vectores k y n, que representan los índices de frecuencia y de tiempo necesarios para aplicar la definición matemática de la Transformada Discreta de Fourier.

```python
    # --- Transformada de Fourier (definición discreta) ---
    k = np.arange(N)
    n = np.arange(N)
    X = np.zeros(N, dtype=complex)
```

> ### **5.4 Cálculo de la transformada de Fourier**

Se realiza el cálculo directo de la transformada de Fourier mediante un bucle ``for``. Para cada frecuencia ``kk``, multiplicando los valores de la señal por un término exponencial complejo ``e^(-j2πkn/N)`` y sumando los resultados. Se toma únicamente la mitad positiva del espectro ya que el EMG, tiene simetría en frecuencia. Con np.linspace() se crea el eje de frecuencias f, y se calcula la densidad espectral.

```python
    # cálculo directo de la transformada de Fourier discreta
    for kk in range(N):
        X[kk] = np.sum(seg_y * np.exp(-1j * 2 * np.pi * kk * n / N))

    # Solo tomamos la mitad positiva (simetría para señales)
    f = np.linspace(0, FS/2, N//2)
    Pxx = np.abs(X[:N//2])**2 / N
```

> ### **5.5 Gráfica de transformada (espectro)**

El código grafica el espectro de potencia en función de la frecuencia para cada contracción muscular. En el gráfico, el eje horizontal representa las frecuencias en hertz, mientras que el eje vertical muestra la potencia relativa de la señal. Esta representación permite visualizar los picos dominantes en el espectro, verificar el rango de frecuencias activas durante la contracción y observar la distribución energética de la señal EMG, complementando el análisis numérico realizado anteriormente.

```python
    # --- Graficar espectro ---
    plt.figure(figsize=(7,3))
    plt.plot(f, Pxx, color='teal')
    plt.title(f'Transformada de Fourier — Contracción {i+1}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia relativa')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```

<p align="center"><b>Gráfica espectro 1</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/28b7b5ad-1b7b-4c68-acf9-0f9624a5a795" />
</p>

<p align="center"><b>Gráfica espectro 2</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/7b42aad1-d180-4180-9bd1-0a8b6bf20870" />
</p>

<p align="center"><b>Gráfica espectro 3</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/f5fefb25-78a0-458f-93ac-8660a9a43bac" />
</p>

<p align="center"><b>Gráfica espectro 4</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/bfb485a3-5886-4a2b-922b-f1351964daa6" />
</p>

<p align="center"><b>Gráfica espectro 5</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/cfd68349-8142-44fd-aebf-15aa151e8547" />
</p>

<p align="center"><b>Gráfica espectro 6</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/5c775a5e-5cce-484d-a749-f986958b7fe6" />
</p>

<p align="center"><b>Gráfica espectro 7</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/7485a640-784c-4ae5-8804-5cc56345cd0d" />
</p>

<p align="center"><b>Gráfica espectro 8</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/ba0a8885-883a-45d3-87fc-1375e2b44cb3" />
</p>

<p align="center"><b>Gráfica espectro 9</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/51033e44-00d6-42b8-bcc4-4ee7d7ffdd6d" />
</p>

<p align="center"><b>Gráfica espectro 10</b></p>

<p align="center">
<img width="690" height="290" alt="image" src="https://github.com/user-attachments/assets/c0c035bb-007c-423b-b4a0-d463bf881add" />
</p>

> ### **5.6 Cálculo de la frecuencia media y mediana**

La frecuencia media ``(f_mean)`` se obtiene como el promedio ponderado de las frecuencias, lo que indica el centro de la transformada. La frecuencia mediana ``(f_median)`` se calcula como el punto donde la energía acumulada alcanza el 50% del total. Estos indicadores permiten evaluar el contenido frecuencial de la señal EMG.

```python
    # --- Calcular frecuencia media y mediana ---
    f_mean = np.sum(f * Pxx) / np.sum(Pxx)
    cumsum = np.cumsum(Pxx)
    f_median = f[np.where(cumsum >= cumsum[-1]/2)[0][0]]

    f_mean_list.append(f_mean)
    f_median_list.append(f_median)
```
<h3 align="center">Resultados finales (frecuencia media y mediana)</h3>

<div align="center">
 
| Contracción | Frecuencia media (Hz) | Frecuencia mediana (Hz) |
|:------------:|:--------------------:|:-----------------------:|
| 1 | 35.71 | 3.86 |
| 2 | 38.66 | 3.89 |
| 3 | 38.32 | 3.89 |
| 4 | 39.38 | 3.89 |
| 5 | 39.34 | 3.89 |
| 6 | 40.27 | 3.89 |
| 7 | 39.04 | 3.89 |
| 8 | 38.58 | 3.89 |
| 9 | 38.36 | 3.89 |
| 10 | 39.27 | 3.89 |

</div>

> ### **5.7 Gráfica evolución de las frecuencias**

Se genera una gráfica que muestra cómo evolucionan las frecuencias media y mediana de la señal EMG a lo largo de las contracciones musculares obtenidas mediante la Transformada de Fourier. Se grafican dos curvas: una de color naranja que representa la frecuencia media y otra de color rojo que muestra la frecuencia mediana, ambas en función del número de contracción. Esto permite observar visualmente si hay variaciones o estabilidad en las frecuencias durante las contracciones.

```python
plt.figure(figsize=(8,4))
plt.plot(range(1, len(f_mean_list)+1), f_mean_list, 'o-', color='orange', label='Frecuencia media')
plt.plot(range(1, len(f_median_list)+1), f_median_list, 's-', color='red', label='Frecuencia mediana')
plt.title('Evolución de las frecuencias (Transformada de Fourier)')
plt.xlabel('Contracción')
plt.ylabel('Frecuencia (Hz)')
plt.ylim(0, 45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

<p align="center"><b>Gráfica</b></p>

<p align="center">
<img width="790" height="390" alt="image" src="https://github.com/user-attachments/assets/06cc20a6-83bb-4059-895e-a610ddaf0ad8" />
</p>

## **6. Análisis**

A lo largo de las diez contracciones simuladas, se observa que la frecuencia media presenta valores entre 35,7 Hz y 40,3 Hz, mostrando un ligero incremento en las primeras contracciones y luego una tendencia a estabilizarse alrededor de los 39 Hz. Este comportamiento indica que, conforme avanza la serie de contracciones, el contenido espectral de la señal se mantiene dentro de un rango relativamente constante, lo que sugiere una actividad muscular simulada sin cambios significativos en la intensidad ni en la velocidad de las contracciones.

Por otro lado, la frecuencia mediana permanece prácticamente constante en 3,89 Hz durante casi todas las contracciones, con una ligera variación inicial. Esta estabilidad refleja que la distribución de energía de la señal no experimenta desplazamientos notables hacia frecuencias más altas o más bajas. En conjunto, la constancia tanto de la frecuencia media como de la mediana sugiere que las contracciones simuladas son uniformes, sin signos de fatiga o variaciones en la respuesta del generador biológico utilizado para emular la señal EMG

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

Posteriormente, se validaron las contracciones mediante tres criterios de aceptación: 
1. duración mínima suficiente para descartar artefactos transitorios
2. Amplitud pico a pico significativa que refleje activación muscular genuina
3. energía RMS superior al nivel de ruido basal.

Únicamente las contracciones que satisfacen simultáneamente estos requisitos se conservan como eventos válidos, garantizando la exclusión de falsas detecciones. El conjunto ``filtered_segs`` reúne así los intervalos correspondientes a activaciones musculares reales, proporcionando una base confiable para el análisis cuantitativo y temporal posterior.

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

En las primeras cinco contracciones (``60–66 s`` aproximadamente), la señal EMG exhibe **mayor estabilidad temporal** y **menor variabilidad en amplitud**. Las formas de onda presentan morfología definida con transiciones suaves, reflejando una **actividad muscular controlada** en ausencia de fatiga significativa. Las amplitudes pico a pico son moderadas y relativamente simétricas respecto al eje de referencia, mientras que la densidad de oscilaciones es reducida, sugiriendo contracciones eficientes con bajo contenido de ruido fisiológico.

Por el contrario, las últimas cinco contracciones (``266–270 s`` aproximadamente) muestran marcada **irregularidad morfológica**. Se evidencian picos más abruptos, fluctuaciones asimétricas y un incremento considerable en la amplitud de la señal, particularmente en las contracciones 112 y 115. Este patrón es consistente con los cambios electromiográficos característicos de la fatiga muscular, donde el aumento de amplitud y la pérdida de consistencia temporal se asocian con la desincronización progresiva de las unidades motoras y el reclutamiento de fibras musculares adicionales para compensar la pérdida de fuerza.

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
El siguiente fragmento de código implementa el cálculo explícito de la *Transformada Discreta de Fourier (DFT)* de cada segmento de señal EMG. A diferencia de la función np.fft —que utiliza algoritmos rápidos (FFT)—, en este caso se construye manualmente la matriz exponencial compleja para aplicar la definición directa de la DFT. Este procedimiento permite ilustrar el fundamento matemático del análisis espectral, mostrando cómo se descompone la señal en sus componentes sinusoidales elementales.

```python
# --- Transformada de Fourier "normal" (DFT manual) ---
k = np.arange(N)
n = np.arange(N)
exp_matrix = np.exp(-2j * np.pi * np.outer(k, n) / N)
Y = np.dot(exp_matrix, seg_y)
```
En la primera parte, se generan los vectores k y n que representan los índices de frecuencia y de tiempo, respectivamente. La matriz exponencial ``exp_matrix`` contiene los términos complejos, que definen las bases armónicas de la transformada. Al multiplicar esta matriz por el vector de la señal ``seg_y``, se obtiene Y, el espectro complejo del segmento, cuyas magnitudes cuadráticas corresponden a la densidad espectral de potencia (Pxx).

```python
# --- Graficar espectro individual ---
plt.figure(figsize=(6, 2))
plt.plot(f, Pxx, color='darkblue')
plt.title(f'Espectro de Potencia — Contracción {i+1}\n')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Potencia')
plt.tight_layout()
plt.show();
```
Posteriormente, se grafica el espectro individual de cada contracción, mostrando cómo se distribuye la energía de la señal en el dominio de la frecuencia. Esta representación permite identificar el comportamiento espectral característico de cada activación muscular, donde el máximo de potencia suele concentrarse en las bandas de baja a media frecuencia (entre 50 y 150 Hz), típicas de la actividad electromiográfica voluntaria.

El análisis de estos espectros es fundamental para estudiar la fatiga muscular, teniendo en cuenta que el desplazamiento progresivo del contenido energético hacia frecuencias más bajas refleja una disminución en la velocidad de conducción de las fibras musculares, un fenómeno directamente asociado a la fatiga fisiológica.

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

En las gráficas se observa que, en las primeras contracciones, la energía de la señal se distribuye de manera más amplia y con mayor intensidad, lo que refleja una activación muscular normal y eficiente. A medida que avanzan las contracciones, el espectro de potencia muestra una disminución general en la amplitud y un desplazamiento progresivo hacia frecuencias más bajas. Este cambio indica la presencia de fatiga muscular, debido a que el músculo pierde capacidad de respuesta rápida y la actividad eléctrica se concentra en componentes de menor frecuencia, evidenciando una reducción en la velocidad de conducción de las fibras musculares.

>### Análisis

A partir del cálculo de la frecuencia media y mediana para las 116 contracciones detectadas, se observa una **variabilidad significativa en la distribución de energía espectral** entre los distintos eventos musculares. Las primeras contracciones muestran valores más elevados de frecuencia media (en torno a ``120–140 Hz``) y mediana (entre ``80–100 Hz``), lo cual refleja una mayor actividad de fibras rápidas y una **eficiente conducción de los potenciales de acción musculares**.

Sin embargo, conforme avanza el registro, ambas frecuencias tienden a **disminuir progresivamente**, alcanzando valores medios más bajos (entre ``80–100 Hz`` para la frecuencia media y 40–60 Hz para la mediana). Este descenso es característico de la aparición de fatiga muscular, fenómeno asociado a la reducción en la velocidad de conducción de las fibras, al aumento de la sincronización de unidades motoras y al predominio de fibras lentas de tipo I durante los periodos prolongados de esfuerzo.

La diferencia entre las frecuencias media y mediana también se amplía en algunos tramos, lo que indica una redistribución asimétrica de la energía espectral hacia componentes de menor frecuencia, evidenciando un desplazamiento del contenido espectral típico de la fatiga. Así, la **disminución sostenida de la frecuencia media y mediana** constituye un marcador objetivo del proceso de fatiga, coherente con los patrones esperados en señales electromiográficas durante ejercicios de contracción repetitiva.

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
## 1. Transformada Rápida de Fourier (FFT) a cada contracción

Para estudiar la distribución de la energía de la señal EMG capturada del voluntario, es necesario aplicar la FFT a cada contracción detectada por el anterior algoritmo. De esta manera, es posible identificar las componentes frecuenciales predominantes de cada evento y evaluar su cambio a medida que se acerca a la fatiga muscular. Para ello, es necesario plantear un algoritmo que recorra una por una las contracciones detectadas y almacenadas en ``filtered_segs``:

```python
# Lista para guardar las frecuencias pico
peak_freqs = []

for i, (s, e) in enumerate(filtered_segs):
    seg_y = emg[s:e]
    N = len(seg_y)
    if N < 4:
        continue
```

En este fragmento `seg_y` etrae el fragmento de la señal que contiene una contracción y si el segmento es demasiado corto (`N < 4`) se omite para no generar un análisis poco confiable.

> [!TIP]
> Se puede crear una lista antes de entrar en el bucle que almacene las frecuencias pico. En este código, `peak_freqs = []` se inicializa como un vector vacío para cumplir con este papel.

```python
 # Eliminar componente DC
    seg_y = seg_y - np.mean(seg_y)

    # FFT
    Y = np.fft.rfft(seg_y)
    Pxx = np.abs(Y)**2 / N
    f = np.fft.rfftfreq(N, 1 / FS)

    # Guardar pico espectral (frecuencia con mayor potencia)
    peak_freq = f[np.argmax(Pxx)]
    peak_freqs.append(peak_freq)
```
Posteriormente, se elimina el offset de la señal (valor medio) para evitar que el espectro detecte los $0$ $Hz$ como mayor amplitud, restando la media de la señal segmentada en `seg_y`. A su vez, se calcula la FFT con `np.fft.rfft()`, se obtiene el espectro de potencia con su ecuación característica y se almacena en `Pxx`. El vector de las frecuencias se obtiene aplicando `np.fft.rfftfreq()` y se almacena en `f`. Para encontrar el pico espectral que resalta la frecuencia dominante de la contracción, `np.argmax(Pxx)` encuentra el índice correspondiente cuando `Pxx` es máximo, y `f[...]` selecciona la frecuencia correspondiende a ese índice. Estos picos se van añadiendo secuencialmente a `peak_freqs = []` para construir el vector que contenga todas las frecuencias dominantes de cada contracción.

```python
    # Graficar espectro
    plt.figure(figsize=(7, 3))
    plt.plot(f, Pxx, color='steelblue')
    plt.title(f'FFT — Contracción {i+1}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Potencia relativa')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, 200)
    plt.show()
```
Por último, se grafica el espectro de cada contracción empleando la frecuencia `f` y la potencia `Pxx` (eje y), hasta $200$ $Hz$ para mejorar visualización e identificación de los picos en las frecuencias esperadas.

> [!IMPORTANT]
> No olvidar que todo el anterior código esta anidado en el bucle `if`. 

## 2. Gráficas del Espectro de Amplitud

A continuación se presentarán las cinco primeras y últimas gráficas del espectro de amplitud en función de la frecuencia, para realizar su respectivo análisis y comparación.

>### Primeras 5 Contracciones

<img width="694" height="290" alt="image" src="https://github.com/user-attachments/assets/5b6b6aa2-f90b-4f18-ac38-20856d29fada" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/f10ad12f-73de-4de2-a45e-4a7917b88875" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/9b141cb2-5e77-464c-884b-3a3ca16f01b4" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/cfc2a020-70e5-48a2-848c-cda3fcded728" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/665c152f-a9c5-441f-881b-e891b92f76a6" />

>### Últimas 5 Contracciones

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/94feb606-3565-4804-a836-8fd0041dfeaf" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/afbf83c8-c846-4ed0-ad5b-66f132b6bfb7" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/4b187aca-7e38-40dd-a167-2dbe6c8e3c5d" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/33f90696-f353-48cd-8337-a4f09de8d7e2" />

<img width="703" height="290" alt="image" src="https://github.com/user-attachments/assets/217ca30c-5644-4942-bc60-f3733c4ff5ab" />

>### Análisis

Al comparar los espectros de amplitud de las primeras y las últimas contracciones, es posible observar la **reducción progresiva** de las componentes de **alta frecuencia** en la señal. En las primeras gráficas es posible apreciar una distribución de la potencia de la señal **intermedia**, con picos notables alrededor de $20$ y $175$ $Hz$. Esto sugiere que, al inicio de la actividad, la señal EMG posee componentes de frecuencia elevadas asociadas a una mayor activación y reclutamiento de unidades motoras rápidas. En contraparte, el espectro de las últimas contracciones revela un principal pico dominante en torno a los $25$ $Hz$ y una mayor distribución de la energía en las frecuencias bajas. Esto indica que, a medida que se acerca a la **fatiga muscular**, la señal EMG **pierde componentes de alta frecuencia**; disminución en la velocidad de conducción en las fibras musculares y reclutamiento mayor de fibras lentas.

## 3. Cálculo del Desplazamiento del Pico Espectral

El desplazamiento del pico espectral hace referencia a el cambio en la frecuencia donde se concentra la mayor energía del espectro de la señal. Para la señal EMG, este desplazamiento refleja la disminución de las componentes de alta frecuencia de la señal a medida que se acerca a la fatiga muscular, donde la frecuencia pico inicial tiende a desplazarse hacia frecuencias más bajas. En el aplicativo de `Python`, se implementó el siguiente código para calcularlo:

```python
if len(peak_freqs) >= 2:
    spectral_shift = np.abs(peak_freqs[-1] - peak_freqs[0])
    print(f"Desplazamiento del pico espectral: {spectral_shift:.2f} Hz")
else:
    print("No hay suficientes contracciones para calcular el desplazamiento.")
```
El compilador verifica que existan más de dos picos para calcular el desplazamiento, luego `spectral_shift` almacena la diferencia entre la frecuencia pico de la última contracción `peak_freqs[-1]` y la primera `peak_freqs[0]`. Para el caso puntal estudiado, se obtuvo el siguiente resultado:

<img width="513" height="48" alt="image" src="https://github.com/user-attachments/assets/e07fc0ee-71ae-43e5-8c66-8040c35bb048" />

Un desplazamiento de $10.03$ $Hz$ sugiere que las principales componentes de frecuencia de la señal EMG disminuyeron en esa magnitud. Aunque el cambio no es muy grande, se observa una tendencia a la reducción de las componentes de alta frecuencia conforme el músculo entra en fatiga. Esto se debe a que, a medida que el músculo se somete a contracciones constantes y sin intervalos de descanso, las señales asociadas al reclutamiento de unidades motoras rápidas tienden a disminuir, reflejando una disminución en la velocidad de conducción muscular.

## 4. Conclusiones

El análisis espectral de la señal EMG permite identificar la fatiga muscular mediante la localización de las componentes frecuenciales dominantes en cada contracción, lo que facilita la evaluación de aspectos como la velocidad de conducción, el reclutamiento de fibras musculares rápidas y los cambios en el tipo de unidades motoras que intervienen durante el esfuerzo. Además, constituye una técnica no invasiva y cuantitativa de gran utilidad en contextos clínicos, académicos y deportivos, al posibilitar el estudio de la función neuromuscular de forma relativamente sencilla y de bajo costo. Finalmente, aunque presenta limitaciones asociadas a artefactos de movimiento, calidad de contacto de los electrodos o ruido ambiental, la aplicación adecuada de herramientas como los filtros digitales permite mitigar dichos efectos y aprovechar la técnica en diversos escenarios experimentales.

# Material Complementario

En esta sección se encuentran las señales extraídas y trabajadas a lo largo de la práctica en formato `.txt`, así como el link al Notebook de *Google Colab*.

## Señales

**EMG del Generador de Señales Biológicas:** [GeneradorEMG_5000.txt](https://github.com/user-attachments/files/23380802/GeneradorEMG_5000.txt)

**EMG hasta fatiga de Voluntario:** [PacienteEMG_5000.txt](https://github.com/user-attachments/files/23380814/PacienteEMG_5000.txt)

> [!NOTE]
> Ambas señales tienen una frecuencia de muestreo $f_s=5k$ $Hz$.

## Notebook

**Link:** [Práctica 4 - EMG](https://colab.research.google.com/drive/1pLhhQaEHhZLiDvcKC2RjFPZOt01cU5rc?usp=sharing)
