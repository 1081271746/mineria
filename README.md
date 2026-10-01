# Laboratorio de Minería de Datos

Aplicación educativa en español con los tres ejercicios del laboratorio CRISP-DM:
predicción del precio del dólar, glucosa y consumo de energía.

## Aplicación web

- Interfaz React + Vite en `artifacts/laboratorio-mineria`.
- Entrenamiento en vivo ejecutado en el navegador con los CSV originales; la
  validación conserva las particiones del análisis Python y el ajuste final usa
  todas las filas.
- Predicciones ejecutadas localmente con los coeficientes ajustados desde Python
  o con el último reentrenamiento del navegador.
- Gráficas, métricas, interpretación, informe, datasets y modelos descargables.
- No requiere base de datos, secretos ni una API para hacer predicciones.

## Reproducir el análisis

Desde la raíz del repositorio:

```bash
python -m pip install -r analysis/requirements.txt
python -m analysis.train_models
```

El script valida los CSV, reserva datos de prueba antes de medir el desempeño,
entrena los modelos finales con todos los datos y regenera:

- `analysis/results/model-results.json`
- `analysis/INFORME.md`
- `artifacts/laboratorio-mineria/src/data/model-data.json`
- `artifacts/laboratorio-mineria/public/downloads/modelos/*.joblib`
- `artifacts/laboratorio-mineria/public/downloads/informe-laboratorio.pdf`
- `artifacts/laboratorio-mineria/public/downloads/codigo-entrenamiento.zip`

La aplicación también permite descargar cada CSV y modelo por separado. El botón
«Entrenamiento en vivo» vuelve a ajustar el ejercicio seleccionado en el navegador
y actualiza los coeficientes que se usan en la predicción interactiva.

Los datos de dólar se dividen cronológicamente: los días iniciales se usan para
entrenamiento y los últimos 100 para prueba. Glucosa y energía usan una partición
aleatoria reproducible 80/20 (`random_state=42`). Hora y día de la semana se
codifican con seno/coseno en el modelo de energía, para representar sus ciclos.

Para predecir desde Python con el modelo exportado:

```bash
python -m analysis.predict_models dolar '{"Dia": 240, "Inflacion": 0.02, "Tasa_interes": 5.0}'
python -m analysis.predict_models glucosa '{"Edad": 48, "IMC": 25, "Actividad_Fisica": 4}'
python -m analysis.predict_models energia '{"Temperatura": 25, "Hora": 18, "Dia_Semana": 3}'
```

Los archivos `.joblib` se deben cargar únicamente desde una fuente confiable y con
una versión compatible de scikit-learn; pickle/joblib no es un formato seguro para
abrir modelos de terceros.

## Ejecutar en Replit

El artefacto web cuenta con un workflow propio. Para comprobar tipos:

```bash
pnpm --filter @workspace/laboratorio-mineria run typecheck
```

## Publicar en Vercel

El proyecto incluye `vercel.json` en la raíz. En Vercel, importa el repositorio
manteniendo como directorio raíz la raíz del monorepo. Vercel instalará el
workspace pnpm, ejecutará el build del artefacto y servirá el contenido estático
de `artifacts/laboratorio-mineria/dist/public`. No se necesita configurar Python
ni variables de entorno para producción.