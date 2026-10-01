# Informe — Laboratorio de Minería de Datos

## Objetivo

Aplicar las fases de CRISP-DM y entrenar tres regresiones lineales múltiples sobre los archivos CSV suministrados. Se documentan la preparación, la validación, la interpretación de coeficientes, el desempeño en prueba y la exportación de los modelos.

## CRISP-DM

1. **Comprensión del negocio:** estimar el precio del dólar, el nivel de glucosa y el consumo eléctrico con las variables incluidas en cada dataset.
2. **Comprensión de los datos:** los archivos contienen 500, 2.000 y 10.000 filas, respectivamente. La validación encontró cero celdas faltantes y cero filas duplicadas en los tres archivos.
3. **Preparación de los datos:** se conservaron las filas y columnas originales. En energía, Hora (1–24) y Dia_Semana (1–7) se codifican con pares seno/coseno para representar correctamente su periodicidad.
4. **Modelado:** regresión lineal por mínimos cuadrados. El modelo de dólar se evalúa con corte temporal (primer 80 % para entrenamiento); glucosa y energía usan partición aleatoria reproducible 80/20 con semilla 42.
5. **Evaluación:** MSE, RMSE y R² se calculan sobre datos reservados que no participaron en el ajuste evaluado. Después se reajusta un modelo final con todas las filas y se exporta a joblib para reutilizarlo.
6. **Despliegue:** una interfaz estática ejecuta la misma ecuación en el navegador; no necesita servidor Python para hacer predicciones. El código Python permite reproducir el entrenamiento y cargar los artefactos.

## Resultados e interpretación

Las magnitudes estandarizadas permiten comparar predictores con unidades distintas. Para los pares seno/coseno de hora y día se agrupa la magnitud de los dos coeficientes mediante raíz de suma de cuadrados. Los resultados son asociativos, no causales.

### Precio del dólar

- Dataset: `dolar_data.csv`; 500 observaciones; 0 faltantes; 0 duplicados.
- Variable objetivo: Precio del dólar (unidad monetaria del dataset (no especificada)).
- Validación: Temporal: primeras 400 filas para entrenamiento y últimas 100 para prueba.
- Prueba: MSE = 2905.7132; RMSE = 53.9047; R² = 0.8788.
- Mayor impacto estandarizado: **Día** (0.998).
- Intercepto del modelo final: 3978.984617.
- Predictores:
  - **Día**: impacto estandarizado 0.9978; coeficiente = 4.999060. Correlación bivariada r = 0.9976. Día: Con las demás variables fijas, +1 día se asocia con 4.999 unidades más del objetivo. Es la variable con mayor impacto estandarizado en este modelo. Asociación del modelo, no evidencia causal.
  - **Inflación**: impacto estandarizado 0.0023; coeficiente = -338.059761. Correlación bivariada r = 0.0196. Inflación: Con las demás variables fijas, +0,01 en inflación (un punto porcentual si la tasa se ingresa como fracción) se asocia con 3.381 unidades menos del objetivo. Impacto estandarizado comparativo: 0.002. Asociación del modelo, no evidencia causal.
  - **Tasa de interés**: impacto estandarizado 0.0017; coeficiente = -2.532823. Correlación bivariada r = 0.0743. Tasa de interés: Con las demás variables fijas, +1 unidad original se asocia con 2.533 unidades menos del objetivo. Impacto estandarizado comparativo: 0.002. Asociación del modelo, no evidencia causal.
- Ecuación: `ŷ = 3978.984617 +4.999060·Día -338.059761·Inflación -2.532823·Tasa de interés`.

### Nivel de glucosa

- Dataset: `glucosa_data.csv`; 2000 observaciones; 0 faltantes; 0 duplicados.
- Variable objetivo: Nivel de glucosa (mg/dL).
- Validación: Aleatorio reproducible: 80 % entrenamiento, 20 % prueba, random_state=42.
- Prueba: MSE = 233.6930; RMSE = 15.2870; R² = 0.6814.
- Mayor impacto estandarizado: **Edad** (0.792).
- Intercepto del modelo final: 66.314991.
- Predictores:
  - **Edad**: impacto estandarizado 0.7916; coeficiente = 1.233681. Correlación bivariada r = 0.7898. Edad: Con las demás variables fijas, +1 años se asocia con 1.234 unidades más del objetivo. Es la variable con mayor impacto estandarizado en este modelo. Asociación del modelo, no evidencia causal.
  - **IMC**: impacto estandarizado 0.1287; coeficiente = 0.883236. Correlación bivariada r = 0.1383. IMC: Con las demás variables fijas, +1 kg/m² se asocia con 0.883 unidades más del objetivo. Impacto estandarizado comparativo: 0.129. Asociación del modelo, no evidencia causal.
  - **Actividad física**: impacto estandarizado 0.2141; coeficiente = -2.010386. Correlación bivariada r = -0.1984. Actividad física: Con las demás variables fijas, +1 horas/semana se asocia con 2.010 unidades menos del objetivo. Impacto estandarizado comparativo: 0.214. Asociación del modelo, no evidencia causal.
- Ecuación: `ŷ = 66.314991 +1.233681·Edad +0.883236·IMC -2.010386·Actividad física`.

### Consumo de energía

- Dataset: `energia_data.csv`; 10000 observaciones; 0 faltantes; 0 duplicados.
- Variable objetivo: Consumo de energía (kWh).
- Validación: Aleatorio reproducible: 80 % entrenamiento, 20 % prueba, random_state=42.
- Prueba: MSE = 907.8891; RMSE = 30.1312; R² = 0.7819.
- Mayor impacto estandarizado: **Temperatura** (0.776).
- Intercepto del modelo final: 153.506388.
- Predictores:
  - **Temperatura**: impacto estandarizado 0.7756; coeficiente = 9.892068. Correlación bivariada r = 0.7719. Temperatura: Con las demás variables fijas, +1 °C se asocia con 9.892 unidades más del objetivo. Es la variable con mayor impacto estandarizado en este modelo. Asociación del modelo, no evidencia causal.
  - **Hora del día**: impacto estandarizado 0.4247; efecto representado por términos cíclicos. Correlación bivariada r = 0.5290. Hora del día: Su efecto se modela con seno y coseno para respetar el ciclo; no existe un coeficiente lineal constante por cada unidad. Impacto estandarizado comparativo: 0.425. Asociación del modelo, no evidencia causal.
  - **Día de la semana**: impacto estandarizado 0.0792; efecto representado por términos cíclicos. Correlación bivariada r = -0.1022. Día de la semana: Su efecto se modela con seno y coseno para respetar el ciclo; no existe un coeficiente lineal constante por cada unidad. Impacto estandarizado comparativo: 0.079. Asociación del modelo, no evidencia causal.
- Ecuación: `ŷ = 153.506388 +9.892068·Temperatura -38.066195·seno de la hora -4.582719·coseno de la hora +6.461092·seno del día +3.052427·coseno del día`.

## Conclusiones

El modelo de dólar obtiene el R² más alto, pero su validación temporal es más estricta que una partición aleatoria y aun así deja error en los días futuros. En glucosa, la edad presenta el mayor impacto estandarizado; el R² indica que hay variación no explicada por las tres variables disponibles. En energía, la temperatura es el predictor de mayor impacto; la codificación cíclica evita suponer que las horas 1 y 24 o los días 1 y 7 están en extremos opuestos.

MSE y RMSE dependen de las unidades de cada objetivo, por lo que no se deben comparar directamente entre los tres problemas. R² compara la proporción de variabilidad explicada dentro de cada conjunto de prueba. Ninguna de estas métricas demuestra causalidad ni garantiza desempeño fuera del rango de los datos observados.

## Reproducibilidad y archivos

Desde la raíz del proyecto: `python -m pip install -r analysis/requirements.txt` y luego `python -m analysis.train_models`. Esto regenera las métricas, los coeficientes estáticos usados por la interfaz, los tres archivos `.joblib`, este informe y el PDF. Los modelos `.joblib` se entrenan con todas las filas y se guardan solo después de evaluar el corte de prueba.

El precio del dólar no declara unidad monetaria en el CSV; el informe no asume COP. Inflación se registra como fracción decimal (por ejemplo, 0,02 equivale a 2 %).
