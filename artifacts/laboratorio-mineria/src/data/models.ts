import modelData from './model-data.json';

export type ModelId = 'dolar' | 'glucosa' | 'energia';
export type ScatterPoint = { x: number; y: number };
export type ScatterSeries = {
  predictorId: string;
  correlation: number | null;
  points: ScatterPoint[];
  linePoints: ScatterPoint[];
};
export type ModelTerm = {
  basis: 'linear' | 'sin' | 'cos';
  coefficient: number | null;
};
export type Predictor = {
  id: string;
  label: string;
  unit: string;
  coefficient: number | null;
  standardizedCoefficient: number | null;
  standardizedImpact: number;
  importance: number | null;
  interpretation: string;
  min: number;
  max: number;
  defaultValue: number;
  encoding?: { kind: 'cyclic'; period: number };
  /** Para variables cíclicas: términos seno/coseno del ajuste. */
  terms?: ModelTerm[];
};
export type EquationTerm = {
  name: string;
  predictorId: string;
  basis: 'linear' | 'sin' | 'cos';
  coefficient: number;
};
export type RegressionModel = {
  id: ModelId;
  label: string;
  target: { id: string; label: string; unit: string };
  observationCount: number | null;
  holdout: { mse: number | null; rmse: number | null; r2: number | null };
  intercept: number | null;
  predictors: Predictor[];
  scatter: ScatterSeries[];
  equationTerms: EquationTerm[];
  splitDescription: string;
  largestImpactId: string;
  deliverables: {
    report: string;
    reportMarkdown: string;
    source: string;
    dataset: string;
    exportedModel: string;
  };
};
export type ModelFitSnapshot = {
  observations: number;
  intercept: number;
  coefficients: number[];
  holdout: { mse: number; rmse: number; r2: number };
};

const publicBase = import.meta.env?.BASE_URL ?? '/';
const resource = (path: string) => `${publicBase}${path}`;

export const models: RegressionModel[] = modelData.models.map((source) => {
  const maxImpact = Math.max(
    ...source.predictors.map((predictor) => predictor.standardizedImpact),
    Number.EPSILON,
  );
  const predictors: Predictor[] = source.predictors.map((predictor) => {
    const cyclicPeriod =
      predictor.id === 'Hora' ? 24 : predictor.id === 'Dia_Semana' ? 7 : null;
    return {
      id: predictor.id,
      label: predictor.label,
      unit: predictor.unit,
      coefficient: predictor.coefficient,
      standardizedCoefficient: predictor.standardizedCoefficient,
      standardizedImpact: predictor.standardizedImpact,
      importance: predictor.standardizedImpact / maxImpact,
      interpretation: predictor.interpretation,
      min: predictor.min,
      max: predictor.max,
      defaultValue: predictor.defaultValue,
      ...(cyclicPeriod ? { encoding: { kind: 'cyclic' as const, period: cyclicPeriod } } : {}),
      ...(cyclicPeriod
        ? {
            terms: predictor.terms.map((term) => ({
              basis: term.basis.endsWith('Sin') ? ('sin' as const) : ('cos' as const),
              coefficient: term.coefficient,
            })),
          }
        : {}),
    };
  });

  return {
    id: source.id as ModelId,
    label: source.title,
    target: {
      id: source.target,
      label: source.targetLabel,
      unit: source.targetUnit,
    },
    observationCount: source.observations,
    holdout: source.metrics,
    intercept: source.equation.intercept,
    predictors,
    scatter: source.predictors.map((predictor) => ({
      predictorId: predictor.id,
      correlation: predictor.correlation,
      points: predictor.scatterPoints.map(([x, y]) => ({ x, y })),
      linePoints: predictor.linePoints.map(([x, y]) => ({ x, y })),
    })),
    equationTerms: source.equation.terms.map((term) => ({
      name: term.name,
      predictorId: term.predictorId,
      basis: term.basis === 'raw' ? 'linear' : term.basis.endsWith('Sin') ? 'sin' : 'cos',
      coefficient: term.coefficient,
    })),
    splitDescription: source.split.description,
    largestImpactId: source.largestImpactId,
    deliverables: {
      report: resource('downloads/informe-laboratorio.pdf'),
      reportMarkdown: resource('downloads/informe-laboratorio.md'),
      source: resource('downloads/codigo-entrenamiento.zip'),
      dataset: resource(`downloads/datasets/${source.dataset}`),
      exportedModel: resource(`downloads/modelos/${source.id}.joblib`),
    },
  };
});

export const modelById = (id: ModelId) => models.find((model) => model.id === id)!;

export function applyFitToModel(model: RegressionModel, fit: ModelFitSnapshot): RegressionModel {
  const equationTerms = model.equationTerms.map((term, index) => ({
    ...term,
    coefficient: fit.coefficients[index] ?? term.coefficient,
  }));
  const predictors = model.predictors.map((predictor) => {
    const terms = equationTerms.filter((term) => term.predictorId === predictor.id);
    if (predictor.encoding) {
      return {
        ...predictor,
        coefficient: null,
        terms: terms.map((term) => ({ basis: term.basis, coefficient: term.coefficient })),
      };
    }
    return { ...predictor, coefficient: terms[0]?.coefficient ?? predictor.coefficient };
  });
  return {
    ...model,
    intercept: fit.intercept,
    holdout: fit.holdout,
    observationCount: fit.observations,
    predictors,
    equationTerms,
  };
}

export function isModelReady(model: RegressionModel): boolean {
  return model.intercept !== null &&
    model.predictors.length > 0 &&
    model.predictors.every((predictor) => {
      if (predictor.encoding?.kind === 'cyclic') {
        return predictor.terms?.length === 2 && predictor.terms.every((term) => term.coefficient !== null);
      }
      return predictor.coefficient !== null;
    });
}

/** Evalúa las columnas originales del CSV; para hora/día expande el ciclo en sin y cos. */
export function predict(model: RegressionModel, rawValues: Record<string, number>): number | null {
  if (!isModelReady(model) || model.intercept === null) return null;
  let value = model.intercept;
  for (const predictor of model.predictors) {
    const raw = rawValues[predictor.id];
    if (!Number.isFinite(raw)) return null;
    if (predictor.encoding?.kind === 'cyclic') {
      if (!Number.isInteger(raw) || raw < 1 || raw > predictor.encoding.period) return null;
      // Los valores de entrada son ordinales desde 1 (hora 1 / día 1).
      const angle = (2 * Math.PI * (raw - 1)) / predictor.encoding.period;
      for (const term of predictor.terms ?? []) {
        if (term.coefficient === null) return null;
        value += term.coefficient * (term.basis === 'sin' ? Math.sin(angle) : term.basis === 'cos' ? Math.cos(angle) : raw);
      }
    } else if (predictor.coefficient !== null) value += predictor.coefficient * raw;
    else return null;
  }
  return Number.isFinite(value) ? value : null;
}