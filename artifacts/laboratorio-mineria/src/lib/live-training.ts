import type { ModelFitSnapshot, ModelId, RegressionModel } from '@/data/models';

type ProgressUpdate = {
  progress: number;
  message: string;
};

export type LiveFit = ModelFitSnapshot & {
  modelId: ModelId;
  trainingRows: number;
  testRows: number;
  elapsedMs: number;
};

type ParsedDataset = {
  rows: number[][];
  targets: number[];
};

const yieldToBrowser = () =>
  new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));

function parseDataset(text: string, model: RegressionModel): ParsedDataset {
  const lines = text.trim().replace(/^\uFEFF/, '').split(/\r?\n/);
  const headers = lines.shift()?.split(',').map((header) => header.trim()) ?? [];
  const required = [...model.predictors.map((predictor) => predictor.id), model.target.id];
  const indexes = required.map((name) => headers.indexOf(name));
  const missing = required.filter((_, index) => indexes[index] < 0);

  if (missing.length) {
    throw new Error(`El CSV no contiene las columnas: ${missing.join(', ')}.`);
  }

  const rows: number[][] = [];
  const targets: number[] = [];
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex].trim();
    if (!line) continue;
    const cells = line.split(',');
    if (cells.length !== headers.length) {
      throw new Error(`La fila ${lineIndex + 2} no tiene ${headers.length} columnas.`);
    }
    const values = indexes.map((index) => Number(cells[index]));
    if (!values.every(Number.isFinite)) {
      throw new Error(`La fila ${lineIndex + 2} contiene un valor no numérico.`);
    }
    rows.push(values.slice(0, -1));
    targets.push(values[values.length - 1]);
  }

  if (rows.length < 10) {
    throw new Error('El CSV debe contener al menos 10 filas válidas.');
  }
  return { rows, targets };
}

function transformRow(modelId: ModelId, raw: number[]): number[] {
  if (modelId !== 'energia') return raw;

  const [temperature, hour, weekday] = raw;
  if (!Number.isInteger(hour) || hour < 1 || hour > 24) {
    throw new Error('Hora debe ser un entero entre 1 y 24.');
  }
  if (!Number.isInteger(weekday) || weekday < 1 || weekday > 7) {
    throw new Error('Día de la semana debe ser un entero entre 1 y 7.');
  }

  const hourAngle = (2 * Math.PI * (hour - 1)) / 24;
  const weekdayAngle = (2 * Math.PI * (weekday - 1)) / 7;
  return [
    temperature,
    Math.sin(hourAngle),
    Math.cos(hourAngle),
    Math.sin(weekdayAngle),
    Math.cos(weekdayAngle),
  ];
}

class NumpyCompatibleRandom {
  private readonly state = new Uint32Array(624);
  private index = 624;

  constructor(seed: number) {
    this.state[0] = seed >>> 0;
    for (let index = 1; index < this.state.length; index += 1) {
      const previous = this.state[index - 1];
      this.state[index] =
        (Math.imul(1812433253, previous ^ (previous >>> 30)) + index) >>> 0;
    }
  }

  private nextUint32(): number {
    if (this.index >= this.state.length) this.twist();

    let value = this.state[this.index];
    this.index += 1;
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c5680;
    value ^= (value << 15) & 0xefc60000;
    value ^= value >>> 18;
    return value >>> 0;
  }

  private twist() {
    for (let index = 0; index < this.state.length; index += 1) {
      const next = (index + 1) % this.state.length;
      const source = (index + 397) % this.state.length;
      const value =
        (this.state[index] & 0x80000000) | (this.state[next] & 0x7fffffff);
      this.state[index] =
        (this.state[source] ^
          (value >>> 1) ^
          (value & 1 ? 0x9908b0df : 0)) >>>
        0;
    }
    this.index = 0;
  }

  private integerInclusive(maximum: number): number {
    let mask = maximum >>> 0;
    mask |= mask >>> 1;
    mask |= mask >>> 2;
    mask |= mask >>> 4;
    mask |= mask >>> 8;
    mask |= mask >>> 16;
    let value = this.nextUint32() & mask;
    while (value > maximum) value = this.nextUint32() & mask;
    return value;
  }

  permutation(size: number): number[] {
    const values = Array.from({ length: size }, (_, index) => index);
    for (let index = size - 1; index > 0; index -= 1) {
      const other = this.integerInclusive(index);
      [values[index], values[other]] = [values[other], values[index]];
    }
    return values;
  }
}

function splitRows(size: number, modelId: ModelId) {
  if (modelId === 'dolar') {
    const cutoff = Math.floor(size * 0.8);
    return {
      train: Array.from({ length: cutoff }, (_, index) => index),
      test: Array.from({ length: size - cutoff }, (_, index) => index + cutoff),
    };
  }

  const permutation = new NumpyCompatibleRandom(42).permutation(size);
  const testSize = Math.ceil(size * 0.2);
  return {
    test: permutation.slice(0, testSize),
    train: permutation.slice(testSize),
  };
}

function fitLinearRegression(rows: number[][], targets: number[]) {
  const rowCount = rows.length;
  const featureCount = rows[0]?.length ?? 0;
  const means = Array.from({ length: featureCount }, (_, column) =>
    rows.reduce((sum, row) => sum + row[column], 0) / rowCount,
  );
  const scales = Array.from({ length: featureCount }, (_, column) => {
    const variance =
      rows.reduce((sum, row) => sum + (row[column] - means[column]) ** 2, 0) /
      rowCount;
    const standardDeviation = Math.sqrt(variance);
    return standardDeviation > 1e-12 ? standardDeviation : 1;
  });
  const dimension = featureCount + 1;
  const gram = Array.from({ length: dimension }, () =>
    Array<number>(dimension).fill(0),
  );
  const targetProducts = Array<number>(dimension).fill(0);

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const vector = [
      1,
      ...rows[rowIndex].map(
        (value, column) => (value - means[column]) / scales[column],
      ),
    ];
    const target = targets[rowIndex];
    for (let left = 0; left < dimension; left += 1) {
      targetProducts[left] += vector[left] * target;
      for (let right = 0; right < dimension; right += 1) {
        gram[left][right] += vector[left] * vector[right];
      }
    }
  }

  const augmented = gram.map((row, index) => [...row, targetProducts[index]]);
  for (let column = 0; column < dimension; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < dimension; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) {
        pivot = row;
      }
    }
    if (Math.abs(augmented[pivot][column]) < 1e-12) {
      throw new Error('El diseño de predictores no permite resolver la regresión.');
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];

    const divisor = augmented[column][column];
    for (let entry = column; entry <= dimension; entry += 1) {
      augmented[column][entry] /= divisor;
    }
    for (let row = 0; row < dimension; row += 1) {
      if (row === column) continue;
      const multiplier = augmented[row][column];
      for (let entry = column; entry <= dimension; entry += 1) {
        augmented[row][entry] -= multiplier * augmented[column][entry];
      }
    }
  }

  const standardizedCoefficients = augmented.map((row) => row[dimension]);
  const coefficients = standardizedCoefficients
    .slice(1)
    .map((value, column) => value / scales[column]);
  const intercept =
    standardizedCoefficients[0] -
    coefficients.reduce((sum, value, column) => sum + value * means[column], 0);
  return { intercept, coefficients };
}

function predictLinear(fit: ReturnType<typeof fitLinearRegression>, row: number[]) {
  return fit.intercept + row.reduce((sum, value, index) => sum + value * fit.coefficients[index], 0);
}

function evaluateFit(
  fit: ReturnType<typeof fitLinearRegression>,
  rows: number[][],
  targets: number[],
) {
  const predictions = rows.map((row) => predictLinear(fit, row));
  const mse =
    predictions.reduce(
      (sum, prediction, index) => sum + (targets[index] - prediction) ** 2,
      0,
    ) / targets.length;
  const mean = targets.reduce((sum, value) => sum + value, 0) / targets.length;
  const totalSumSquares = targets.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  const residualSumSquares = mse * targets.length;

  return {
    mse,
    rmse: Math.sqrt(mse),
    r2: totalSumSquares === 0 ? 0 : 1 - residualSumSquares / totalSumSquares,
  };
}

export async function trainModelInBrowser(
  model: RegressionModel,
  onProgress: (update: ProgressUpdate) => void,
): Promise<LiveFit> {
  const startedAt = performance.now();
  onProgress({ progress: 8, message: 'Leyendo el CSV original del ejercicio…' });
  const response = await fetch(model.deliverables.dataset);
  if (!response.ok) {
    throw new Error(`No se pudo cargar el dataset (${response.status}).`);
  }
  const dataset = parseDataset(await response.text(), model);

  onProgress({ progress: 28, message: `Validando ${dataset.rows.length.toLocaleString('es-CO')} filas…` });
  await yieldToBrowser();

  const designRows = dataset.rows.map((row) => transformRow(model.id, row));
  const split = splitRows(designRows.length, model.id);
  const trainingRows = split.train.map((index) => designRows[index]);
  const trainingTargets = split.train.map((index) => dataset.targets[index]);
  const holdoutRows = split.test.map((index) => designRows[index]);
  const holdoutTargets = split.test.map((index) => dataset.targets[index]);

  onProgress({
    progress: 52,
    message: `Ajustando regresión con ${trainingRows.length.toLocaleString('es-CO')} filas de entrenamiento…`,
  });
  await yieldToBrowser();
  const holdoutFit = fitLinearRegression(trainingRows, trainingTargets);

  onProgress({ progress: 75, message: 'Evaluando sobre filas reservadas…' });
  await yieldToBrowser();
  const holdout = evaluateFit(holdoutFit, holdoutRows, holdoutTargets);

  onProgress({ progress: 88, message: 'Reajustando el modelo final con todos los datos…' });
  await yieldToBrowser();
  const fullFit = fitLinearRegression(designRows, dataset.targets);

  onProgress({ progress: 100, message: 'Modelo listo; ya se usa para las predicciones.' });
  return {
    modelId: model.id,
    observations: dataset.rows.length,
    trainingRows: trainingRows.length,
    testRows: holdoutRows.length,
    intercept: fullFit.intercept,
    coefficients: fullFit.coefficients,
    holdout,
    elapsedMs: performance.now() - startedAt,
  };
}