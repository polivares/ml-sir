# `src/sir/`: módulos reutilizables del benchmark SIR

Esta carpeta contiene el **código activo** del benchmark para inferir parámetros del modelo **SIR**
desde series temporales, principalmente usando como entrada la trayectoria **I(t)** y como salida
los parámetros **(beta, gamma)**.

En el repositorio, los experimentos reproducibles se ejecutan desde `scripts/` (Exp0/Exp1/Exp2),
pero esos scripts dependen directamente de estos módulos para:
- Simular SIR (`simulate.py`)
- Cargar/preparar datasets (`datasets.py`)
- Modelar observación ruidosa y preprocesamiento (`noise.py`)
- Ajustar baselines clásicos (`baseline.py`)
- Entrenar/evaluar modelos ML (`ml.py`)
- Calcular métricas y tiempos (`metrics.py`)
- Guardar artefactos de corrida (`io.py`)
- Reusar splits/arrays vía caché (`cache.py`)
- Configurar logs (`logging_utils.py`)
- Guardar predicciones por curva (`predictions.py`)
- Actualizar bitacora de experimentos (`experiment_log.py`)

Visualización:
- Las funciones de plots viven en `src/visualization/visualize.py`.
- Los scripts pueden generar figuras con `--save-plots` (ver README principal).
- Con `--save-plot-data`, los scripts también guardan `plot_data.npz` y `plot_data.json` para
  reproducir las figuras sin recalcular.
- `visualize.py` también se puede ejecutar como script para generar figuras desde `predictions.npz`
  y/o `metrics.csv` (ver docstring al inicio del archivo).
  - Incluye un plot `architectures.png` que lista qué baseline(s) y arquitecturas ML se usaron en la corrida.
  - También puedes generarlo manualmente con:
    `python -m src.visualization.visualize --predictions runs/<run>/predictions.npz --plots architectures`

---

## Convenciones generales (inputs/outputs)

### Representación de una simulación en `sir.pkl`

Los scripts/notebooks asumen que `sir.pkl` contiene una lista de tuplas:

- `outputs`: `np.ndarray` de shape `(T, 3)` con columnas `[S, I, R]`
- `times`: `np.ndarray` de shape `(T,)` con la grilla temporal
- `params`: típicamente `[beta, gamma]` (o estructura equivalente)

### Representación del dataset supervisado

- `X`: `np.ndarray` de shape `(n_samples, T)` donde cada fila es una serie temporal `I(t)`
- `y`: `np.ndarray` de shape `(n_samples, 2)` con columnas `[beta, gamma]`

### Reproducibilidad

- Los scripts llaman `set_global_seed(seed)` y usan `np.random.default_rng(seed)`.
- Varias funciones aceptan `rng` (NumPy Generator) para controlar aleatoriedad.

---

## `config.py` — defaults y seeds

**Objetivo**
- Centralizar configuraciones por defecto para scripts/experimentos.
- Proveer una función única para fijar semillas en librerías relevantes.

**Componentes principales**

- `Defaults` (`@dataclass(frozen=True)`):
  - Contiene rangos y parámetros globales del benchmark:
    - Grilla temporal: `t0`, `t1`, `dt`
    - Condiciones iniciales: `s0`, `i0`, `r0`
    - Rangos de búsqueda: `beta_range`, `gamma_range`
    - Split: `test_size`, `val_size`
    - Observación: `rho`, `k` (NegBin)
    - Paths: `data_path`, `runs_dir`
- `DEFAULTS`: instancia global de `Defaults`.
- `set_global_seed(seed: int) -> None`:
  - **Entrada**: `seed` (int)
  - **Salida**: ninguna (efecto lateral)
  - **Qué hace**: setea semilla en `random`, `numpy` y en `tensorflow` (si está instalado) y habilita *GPU memory growth* cuando hay GPU, para evitar que TensorFlow reserve toda la VRAM al inicio.
  - **Nota**: si estás en notebooks y ya importaste TensorFlow antes de llamar `set_global_seed`, reinicia el kernel para asegurar que la configuración de GPU se aplique antes de inicializar la GPU.

**Ejemplo**
```python
from src.sir.config import DEFAULTS, set_global_seed

set_global_seed(42)
print(DEFAULTS.dt, DEFAULTS.beta_range)
```

---

## `simulate.py` — simulación SIR con `summer`

**Objetivo**
- Proveer una única función de simulación SIR usada por los baselines clásicos (que re-simulan
  por cada curva durante la optimización).

**Función principal**

- `simulate_sir(beta, gamma, s0=..., i0=..., r0=..., t0=..., t1=..., dt=..., return_full=False)`
  - **Entradas**
    - `beta: float`: tasa de transmisión (en el flujo de infección)
    - `gamma: float`: tasa de recuperación
    - `s0, i0, r0: float`: estado inicial
    - `t0, t1: float`: inicio/fin de la simulación
    - `dt: float`: timestep (debe ser > 0)
    - `return_full: bool`: si `True` retorna trayectorias completas
  - **Salidas**
    - Si `return_full=False`: `I` (`np.ndarray` shape `(T,)`)
    - Si `return_full=True`: `(times, outputs)` donde:
      - `times`: `np.ndarray` shape `(T,)`
      - `outputs`: `np.ndarray` shape `(T, 3)` con columnas `[S, I, R]`
  - **Dependencias**: `from summer import CompartmentalModel` (paquete `summerepi`)

**Ejemplo**
```python
import numpy as np
from src.sir.simulate import simulate_sir

I = simulate_sir(beta=0.5, gamma=0.2, t0=0.0, t1=30.0, dt=0.1)
assert isinstance(I, np.ndarray)

times, outputs = simulate_sir(0.5, 0.2, return_full=True)
S, I2, R = outputs.T
```

**Notas**
- La simulación es determinista dado `(beta, gamma, s0, i0, r0, t0, t1, dt)`.
- En Exp2 (downsampling) los scripts ajustan `dt` y `t1` efectivos para mantener coherencia con la
  longitud de la serie transformada.

---

## `datasets.py` — carga, armado de X/y, splits y normalización

**Objetivo**
- Cargar `sir.pkl`.
- Convertir la lista de simulaciones en arreglos `X`/`y`.
- Crear splits reproducibles.
- Estandarizar normalizaciones para ML.

### `load_sir_pkl`

- `load_sir_pkl(path=DEFAULTS.data_path, limit=None, rng=None) -> list`
  - **Entrada**
    - `path`: ruta al `sir.pkl`
    - `limit`: si no es `None`, submuestrea `limit` simulaciones (sin reemplazo)
    - `rng`: `np.random.Generator` (si no, usa `DEFAULTS.seed`)
  - **Salida**
    - Lista de simulaciones: `[(outputs, times, params), ...]`
  - **Nota importante (memoria)**
    - Actualmente hace `pickle.load` del archivo completo y **luego** submuestrea. Si `sir.pkl` es
      muy grande, el primer load puede consumir mucha RAM. El caché de `scripts/` ayuda a evitar
      re-cargar el pickle en corridas repetidas, pero el primer run aún lo carga completo.

### `build_Xy_I_only`

- `build_Xy_I_only(data, normalize=None, dtype=np.float32, return_times=False)`
  - **Entrada**
    - `data`: iterable de tuplas `(outputs, times, params)`
    - `normalize`: `None | "population" | "max"` (aplicada sobre `I(t)` en este paso)
    - `dtype`: tipo numérico para `X` y `y`
    - `return_times`: si `True`, devuelve un `times_ref` para plotting
  - **Salida**
    - `X`: shape `(n_samples, T)` con `I(t)`
    - `y`: shape `(n_samples, 2)` con `[beta, gamma]`
    - opcional `times_ref`: shape `(T,)`

### `train_val_test_split`

- `train_val_test_split(X, y, test_size=..., val_size=..., rng=None, shuffle=True, return_indices=False)`
  - **Entrada**
    - `X`, `y` como arriba
    - `test_size`, `val_size`: fracciones
    - `rng`: controla `shuffle`
    - `return_indices`: si `True`, devuelve `idx_train/idx_val/idx_test`
  - **Salida**
    - Diccionario con arrays del split:
      - `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`
      - y opcionalmente `idx_train`, `idx_val`, `idx_test`

### `normalize_series`

- `normalize_series(X, method=None, population=None) -> np.ndarray`
  - **Entrada**
    - `X`: series (batch) shape `(n_samples, T)` o compatible
    - `method`: `None | "max" | "population"`
    - `population`: requerido si `method="population"`; shape `(n_samples,)`
  - **Salida**
    - `X_norm` con la misma shape que `X`

**Ejemplo**
```python
import numpy as np
from src.sir.datasets import load_sir_pkl, build_Xy_I_only, train_val_test_split, normalize_series

data = load_sir_pkl("data/raw/simulated/SIR/sir.pkl", limit=5000, rng=np.random.default_rng(42))
X, y = build_Xy_I_only(data, normalize=None)
splits = train_val_test_split(X, y, test_size=0.1, val_size=0.1, rng=np.random.default_rng(42))

X_train = normalize_series(splits["X_train"], method="max")
```

---

## `noise.py` — modelos de observación y preprocesamiento

**Objetivo**
- Transformar series latentes `I(t)` en observaciones más realistas.
- Proveer transformaciones de “disponibilidad” de datos (ventanas parciales, downsampling, missingness).

**Nota importante (para evitar confusiones)**
- Las funciones `observe_*` **no** generan “`I(t) + ruido`” (ruido aditivo).
- En su lugar modelan un **proceso de observación**: a partir de la trayectoria latente `I(t)` generan una serie observada
  `Y(t)` por muestreo (conteos enteros) usando una distribución (Poisson o NegBin).
- En los scripts del benchmark, típicamente `Y(t)` se usa como “la observación” que alimenta al baseline clásico por MLE
  o a los modelos ML, aunque conceptualmente `Y(t)` y `I(t)` no son lo mismo.

### Observación por conteos

- `observe_poisson(I, rho, rng=None) -> np.ndarray`
  - **Entrada**
    - `I`: array (1D o batch) con valores no-negativos (puede ser float)
    - `rho`: reporting rate (escala la media)
    - `rng`: `np.random.Generator`
  - **Salida**
    - `Y`: `np.ndarray` de enteros (`int64`) con conteos Poisson, misma shape que `I`
  - **Modelo**
    - `Y_t ~ Poisson(lambda_t)` con `lambda_t = rho * I_t`
  - **Interpretación**
    - `rho` actúa como reporting rate (subreporte si `rho < 1`).
    - La salida es una realización aleatoria; misma shape que `I`, pero en enteros.

- `observe_negbin(I, rho, k, rng=None) -> np.ndarray`
  - **Entrada**
    - `I`: como arriba
    - `rho`: reporting rate
    - `k`: parámetro de dispersión (>0)
  - **Salida**
    - `Y`: `np.ndarray` (`int64`) con conteos NegBin, misma shape que `I`
  - **Parametrización**
    - media `mu = rho * I`
    - varianza `mu + mu^2 / k`
    - internamente usa `p = k / (k + mu)` para muestreo estable
  - **Interpretación**
    - NegBin permite **overdispersion** (varianza > media), frecuente en datos epidemiológicos.
    - Menor `k` implica mayor varianza para un mismo `mu`.

### Transformaciones temporales

- `apply_downsample(x, step) -> np.ndarray`
  - **Entrada**: `x` con eje temporal al final; `step` entero > 0
  - **Salida**: `x[..., ::step]`

- `apply_window(x, T, dt) -> np.ndarray`
  - **Entrada**: `T` (días a conservar), `dt` (timestep original)
  - **Salida**: recorta el eje temporal al número de puntos `n = round(T/dt)+1`

### Missingness e imputación simple

- `apply_missing(x, p, rng=None, method="ffill", return_mask=False)`
  - **Entrada**
    - `x`: series (1D o batch)
    - `p`: probabilidad de missing por entrada (`0 <= p < 1`)
    - `method`: `None | "ffill" | "interp"`
    - `return_mask`: si `True`, retorna también el mask booleano
  - **Salida**
    - `x_out`: `np.ndarray` float (imputado o con `NaN` si `method=None`)
    - opcional `mask`: `np.ndarray` booleano (`True` = missing)
  - **Notas**
    - `"ffill"` hace forward-fill y backfill del inicio si es necesario.
    - `"interp"` interpola linealmente sobre el tiempo.
    - Este paso modela datos faltantes (no ruido de medición).

**Ejemplo**
```python
import numpy as np
from src.sir.noise import observe_poisson, observe_negbin, apply_window, apply_downsample, apply_missing

rng = np.random.default_rng(0)
I = np.linspace(0, 50, 1001)

Y_pois = observe_poisson(I, rho=0.5, rng=rng)
Y_nb = observe_negbin(I, rho=0.5, k=10.0, rng=rng)

I_30d = apply_window(I, T=30.0, dt=0.1)
I_weekly = apply_downsample(I, step=10)

I_imp, mask = apply_missing(I, p=0.1, rng=rng, method="interp", return_mask=True)
```

---

## `metrics.py` — métricas de error y resumen de tiempos

**Objetivo**
- Estandarizar el reporte de métricas por parámetro y de tiempos (latencia).

**Funciones principales**

- `mae(y_true, y_pred) -> float`
- `rmse(y_true, y_pred) -> float`
- `r2(y_true, y_pred) -> float`
  - Implementa `R^2 = 1 - SS_res/SS_tot` y protege caso `SS_tot=0`.

- `per_param_metrics(y_true, y_pred) -> Dict[str, float]`
  - **Entrada**
    - `y_true`, `y_pred`: arrays shape `(n_samples, 2)` con columnas `[beta, gamma]`
  - **Qué hace exactamente**
    - Calcula **3 métricas por parámetro** (beta y gamma) usando las funciones `mae`, `rmse` y `r2`.
    - Internamente recorre las columnas `0 -> beta` y `1 -> gamma`, y devuelve un diccionario “plano” (no anidado).
  - **Salida**: dict con 6 llaves:
    - `mae_beta`, `rmse_beta`, `r2_beta`, `mae_gamma`, `rmse_gamma`, `r2_gamma`
  - **Qué NO hace**
    - No calcula métricas de tiempo (para eso está `timing_summary`).
    - No calcula incertidumbre/cobertura (si se agrega a futuro, será un módulo/función aparte).

- `timing_summary(times) -> Dict[str, float]`
  - **Entrada**: `times` (array-like) con tiempos en segundos
  - **Salida**: `{"time_p50": ..., "time_p90": ...}`

**Ejemplo**
```python
import numpy as np
from src.sir.metrics import per_param_metrics, timing_summary

y_true = np.array([[0.5, 0.2], [1.0, 0.3]])
y_pred = np.array([[0.6, 0.1], [0.9, 0.35]])

print(per_param_metrics(y_true, y_pred))
print(timing_summary(np.array([0.01, 0.02, 0.05])))
```

---

## `baseline.py` — baselines clásicos (optimización / MLE)

**Objetivo**
- Implementar métodos clásicos comparables con ML:
  - Ajuste por MSE en I(t) limpio.
  - Ajuste por MLE bajo modelos observacionales (Poisson / NegBin).
- Incluir multi-start y registrar tiempos para comparar eficiencia.

### Estructura de salida: `FitResult`

`FitResult` (`@dataclass`) contiene:
- `params: np.ndarray`: mejor vector de parámetros encontrado
  - Para `fit_mse`: `[beta, gamma]`
  - Para `fit_poisson_mle`/`fit_negbin_mle`:
    - `[beta, gamma]` si `estimate_rho=False`
    - `[beta, gamma, rho]` si `estimate_rho=True`
- `loss: float`: valor de la función objetivo (MSE o NLL)
- `times: List[float]`: tiempos por reinicio (cada llamada a `scipy.optimize.minimize`)

### `fit_mse`

- `fit_mse(I_obs, beta_bounds=..., gamma_bounds=..., n_starts=5, rng=None, t0=..., t1=..., dt=..., s0=..., i0=..., r0=...)`
  - **Entrada**
    - `I_obs`: `np.ndarray` shape `(T,)` (serie observada)
    - bounds: rangos de búsqueda (por defecto `DEFAULTS.beta_range` / `DEFAULTS.gamma_range`)
    - `n_starts`: cantidad de reinicios
    - `rng`: controla inicializaciones aleatorias
    - parámetros de simulación: definen la grilla y condiciones iniciales usadas en el fit
  - **Salida**: `FitResult`
  - **Qué optimiza**
    - Minimiza `mean((I_sim(beta,gamma) - I_obs)^2)`

### `fit_poisson_mle`

- `fit_poisson_mle(Y_obs, rho=DEFAULTS.rho, estimate_rho=False, ... , n_starts=5, ...)`
  - **Entrada**
    - `Y_obs`: `np.ndarray` shape `(T,)` con conteos observados
    - `rho`: reporting rate fijo (si `estimate_rho=False`)
    - `estimate_rho`: si `True`, agrega `rho` al vector de parámetros y optimiza también
  - **Salida**: `FitResult`
  - **Qué optimiza**
    - NLL Poisson (sin constante factorial por defecto):
      - `lambda_t = rho * I_sim(t)`
      - `NLL = sum(lambda_t - Y_t * log(lambda_t))`

### `fit_negbin_mle`

- `fit_negbin_mle(Y_obs, rho=DEFAULTS.rho, k=DEFAULTS.k, estimate_rho=False, ...)`
  - **Entrada**
    - `Y_obs`: conteos observados
    - `k`: dispersión (fija)
    - `rho`, `estimate_rho`: análogo a Poisson
  - **Salida**: `FitResult`
  - **Qué optimiza**
    - NLL de NegBin con tamaño `k` y probabilidad `p = k/(k+mu)` para `mu = rho*I_sim`.

**Ejemplo**
```python
import numpy as np
from src.sir.simulate import simulate_sir
from src.sir.baseline import fit_mse, fit_poisson_mle

# Serie "observada" (aquí sin ruido, solo ejemplo)
I_obs = simulate_sir(beta=0.8, gamma=0.2, t0=0.0, t1=30.0, dt=0.1)

fit = fit_mse(I_obs, n_starts=5)
print("beta,gamma:", fit.params, "mse:", fit.loss)

# Observación Poisson de esa serie
rng = np.random.default_rng(0)
Y_obs = rng.poisson(0.5 * I_obs).astype(np.int64)
fit_p = fit_poisson_mle(Y_obs, rho=0.5, n_starts=5)
print("beta,gamma:", fit_p.params, "nll:", fit_p.loss)
```

**Notas**
- El método de optimización es `L-BFGS-B` (`scipy.optimize.minimize`) con bounds.
- El costo total del baseline clásico crece con:
  - número de curvas (`max-test` en scripts)
  - `n_starts`
  - longitud temporal `T` (por costo de simulación)

---

## `ml.py` — modelos ML/DL (TensorFlow/Keras)

**Objetivo**
- Definir arquitecturas estándar (MLP/CNN/RNN/Transformer) para mapear `I(t) -> (beta, gamma)`.
- Incluir variantes distribucionales (heteroscedástico, MDN) cuando se necesita incertidumbre.
- Proveer entrenamiento con early stopping y medición de tiempo de entrenamiento/inferencia.

**Dependencia**
- Requiere `tensorflow` instalado. Si no lo está, `_require_tf()` lanza `RuntimeError`.

### Modelos

- `build_linear(input_dim=1001) -> keras.Model`
  - Regresión lineal (Dense(2)) con L2 opcional.
  - Útil como baseline ML muy simple.

- `build_mlp(input_dim=1001) -> keras.Model`
  - MLP secuencial con capas densas y activación `swish`.
  - Salida: 2 unidades lineales `[beta, gamma]`.
  - Compila con `AdamW`, `Huber(delta=1e-4)` y `R2Score`.

- `build_mlp_branched(input_dim=1001) -> keras.Model`
  - Tronco compartido + 2 cabezas (beta y gamma) concatenadas.
  - Mismo loss/métricas para comparabilidad.

- `build_resmlp(input_dim=1001, width=128, depth=4, dropout=0.1) -> keras.Model`
  - MLP con bloques residuales + LayerNorm.

- `build_cnn1d(input_len=1001) -> keras.Model`
  - CNN liviana: reshape a `(T, 1)` + Conv1D + GAP + dense.

- `build_tcn(input_len=1001, ...) -> keras.Model`
  - TCN con Conv1D dilatadas y residuales.

- `build_inception(input_len=1001, ...) -> keras.Model`
  - CNN estilo Inception (múltiples kernels por bloque).

- `build_attn_cnn(input_len=1001, ...) -> keras.Model`
  - CNN con pooling por atención.

- `build_gru(input_len=1001, ...) -> keras.Model`
  - Encoder GRU (opcional bidireccional).

- `build_lstm(input_len=1001, ...) -> keras.Model`
  - Encoder LSTM (opcional bidireccional).

- `build_conv_gru(input_len=1001, ...) -> keras.Model`
  - Downsampling Conv1D + GRU.

- `build_transformer(input_len=1001, ...) -> keras.Model`
  - Encoder Transformer pequeño con pooling global.

- `build_mlp_heteroscedastic(input_dim=1001, ...) -> keras.Model`
  - Predice media + log-varianza por parámetro.
  - Loss: NLL gaussiana (heteroscedástico).

- `build_mlp_mdn(input_dim=1001, n_components=3, ...) -> keras.Model`
  - MDN (mezcla Gaussiana) sobre `(beta, gamma)`.
  - Loss: NLL de mezcla.

### Entrenamiento y timing

- `train_model(model, X_train, y_train, X_val, y_val, epochs=100, patience=15, batch_size=32) -> TrainResult`
  - **Entradas**
    - `X_*`: shape `(n_samples, T)` (float)
    - `y_*`: shape `(n_samples, 2)`
  - **Salida**: `TrainResult` con:
    - `history`: objeto Keras History
    - `eval_metrics`: dict con métricas en validation (según `model.metrics_names`)
    - `train_time_sec`: tiempo total de entrenamiento (segundos)
  - Usa `EarlyStopping` con `monitor=model.sir_monitor` si existe; si no, usa `val_r2_score`.
    - Los modelos heteroscedásticos/MDN fijan `sir_monitor="val_loss"` por defecto.

- `predict_time_per_sample(model, X, n_samples=100) -> np.ndarray`
  - **Entrada**: `X` shape `(n_samples_total, T)`
  - **Salida**: array de tiempos por predicción (segundos), estimado muestreando predicciones de a 1.

- `predict_params(model, X) -> np.ndarray`
  - Devuelve `(beta, gamma)` incluso si el modelo tiene cabeza distribucional:
    - heteroscedástico: retorna la media (primeras 2 columnas).
    - MDN: retorna la media ponderada por las mezclas.

- `save_model_artifacts(model, name, out_dir) -> dict`
  - Guarda **pesos** (checkpoints TF), **diagrama** (`architecture.png`), `architecture.json`
    y `summary.txt` en `out_dir/<name>/`.
  - Es usado por los scripts de experimento para dejar trazabilidad de la arquitectura entrenada.
  - El diagrama usa `tf.keras.utils.plot_model` y requiere Graphviz + `pydot`
    (si no están instalados, se emite un warning y el resto se guarda igual).

**Ejemplo**
```python
import numpy as np
from src.sir.ml import build_mlp, train_model

# X_train: (n, T), y_train: (n, 2)
X_train = np.random.rand(1000, 1001).astype(np.float32)
y_train = np.random.rand(1000, 2).astype(np.float32)
X_val = np.random.rand(200, 1001).astype(np.float32)
y_val = np.random.rand(200, 2).astype(np.float32)

model = build_mlp(input_dim=X_train.shape[1])
res = train_model(model, X_train, y_train, X_val, y_val, epochs=50, patience=5, batch_size=32)
print(res.train_time_sec, res.eval_metrics)
```

---

## `io.py` — escritura de artefactos (config y métricas)

**Objetivo**
- Mantener consistente la forma en que los scripts crean carpetas y guardan resultados.

**Funciones**

- `ensure_dir(path) -> Path`
  - Crea carpeta (con `parents=True`) y retorna `Path`.

- `save_json(path, payload) -> None`
  - Guarda JSON con `indent=2` y `sort_keys=True` (differences estables).

- `save_csv(path, rows) -> None`
  - `rows`: iterable de dicts.
  - Si `rows` está vacío, no crea el archivo.
  - Usa las llaves de la primera fila como `fieldnames`.

**Ejemplo**
```python
from src.sir.io import ensure_dir, save_json, save_csv

out = ensure_dir("runs/debug_example")
save_json(out / "config.json", {"seed": 42})
save_csv(out / "metrics.csv", [{"method": "baseline", "mae_beta": 0.1}])
```

---

## `logging_utils.py` — logging consistente en scripts

**Objetivo**
- Proveer un setup de logging uniforme (consola + archivo) para los scripts de benchmark.

**Función principal**

- `setup_logging(level="INFO", log_file=None, console=True) -> logging.Logger`
  - **Entrada**:
    - `level`: nivel (`INFO`, `WARNING`, `DEBUG`).
    - `log_file`: path opcional del archivo de log.
    - `console`: si `True`, emite logs a consola.
  - **Salida**: logger configurado (root).
  - **Qué hace**: limpia handlers previos y configura formato con timestamps.

**Ejemplo**
```python
from src.sir.logging_utils import setup_logging

setup_logging(level="INFO", log_file="runs/exp0_debug/run.log")
```

---

## `predictions.py` — guardar predicciones por curva

**Objetivo**
- Guardar, por corrida, las series `I(t)` (test), el tiempo `t` y las predicciones `(beta, gamma)`
  por método para auditoría y re-plotting posterior.

**Función principal**

- `save_predictions(out_dir, times, i_true, y_true, y_pred_by_method, ...)`
  - **Entradas**:
    - `times`: grilla temporal (después de window/downsample si aplica).
    - `i_true`: serie I(t) del test (shape `n_test x T`).
    - `y_true`: `(beta, gamma)` verdadero (shape `n_test x 2`).
    - `y_pred_by_method`: dict con predicciones por método.
    - `i_obs`: opcional (solo Exp1) para observaciones ruidosas.
  - **Salidas**:
    - `predictions.npz`: arrays con inputs + predicciones.
    - `predictions.json`: metadata (exp, seed, dt, etc.).

**Ejemplo**
```python
from src.sir.predictions import save_predictions

save_predictions(
    "runs/exp0_debug",
    times=times,
    i_true=X_test,
    y_true=y_test,
    y_pred_by_method={"baseline_mse": y_pred},
    metadata={"exp": "exp0"},
)
```

---

## `experiment_log.py` — bitacora automatica de corridas

**Objetivo**
- Mantener `EXPERIMENTS.md` actualizado con:
  - el ultimo run por experimento,
  - una lista con checkboxes para seleccionar corridas finales,
  - un historial de corridas con metricas resumidas.

**Funciones principales**

- `update_experiment_log(...)`
  - Inserta un nuevo entry en el historial y actualiza las secciones de
    "Last run" y "Final selection".
- `summarize_args(...)`
  - Convierte un dict de args a un resumen compacto para el log.

---

## `cache.py` — caché de arrays/splits (hash-based)

**Objetivo**
- Evitar recomputar:
  - `X`, `y`, `pop`
  - índices `idx_train/idx_val/idx_test`
cuando se repiten corridas con el mismo `data_path/limit/seed/sizes`.

**Funciones**

- `hash_config(payload, length=12) -> str`
  - **Entrada**: dict con configuración relevante.
  - **Salida**: hash corto (sha256 truncado) usado como nombre de carpeta.

- `cache_paths(base_dir, key) -> (dir, arrays_path, config_path)`
  - Construye paths:
    - `<base_dir>/<key>/arrays.npz`
    - `<base_dir>/<key>/config.json`

- `save_cache(base_dir, key, arrays, config) -> None`
  - `arrays`: dict de `np.ndarray` que se guarda como `np.savez_compressed`.
  - `config`: dict guardado como JSON.

- `load_cache(base_dir, key) -> (arrays_dict, config_dict)`
  - Devuelve arrays materializados como `Dict[str, np.ndarray]`.

- `cache_exists(base_dir, key) -> bool`

**Ejemplo**
```python
import numpy as np
from src.sir.cache import hash_config, save_cache, load_cache, cache_exists

cfg = {"data_path": "data/raw/simulated/SIR/sir.pkl", "limit": 5000, "seed": 42, "test_size": 0.1, "val_size": 0.1}
key = hash_config(cfg)

if not cache_exists("data/processed/sir", key):
    save_cache("data/processed/sir", key, arrays={"X": np.zeros((2, 3))}, config=cfg)

arrays, cfg2 = load_cache("data/processed/sir", key)
```

**Nota**
- El caché se usa en `scripts/exp0_run.py`, `scripts/exp1_noise.py` y `scripts/exp2_window_downsample.py`.

---

## `__init__.py` — re-exports para imports simples

**Objetivo**
- Evitar imports largos y entregar una “API” mínima del paquete:
  - `from src.sir import simulate_sir, load_sir_pkl, fit_mse, ...`

**Qué exporta**
- `DEFAULTS`, `set_global_seed`
- `simulate_sir`
- `load_sir_pkl`, `build_Xy_I_only`, `train_val_test_split`
- `per_param_metrics`, `timing_summary`
- `observe_poisson`, `observe_negbin`, `apply_downsample`, `apply_window`, `apply_missing`
- `fit_mse`, `fit_poisson_mle`, `fit_negbin_mle`
- `hash_config`, `cache_exists`, `load_cache`, `save_cache`

**Ejemplo**
```python
from src.sir import simulate_sir, fit_mse, observe_poisson
```
