import { useMemo, useState } from 'react';
import { Route, Switch } from 'wouter';
import {
  Activity, ArrowDownToLine, ArrowRight, BarChart3, BookOpen, BrainCircuit,
  Check, ChevronRight, CircleHelp, Database, FileCode2, FileText, FlaskConical,
  Info, LoaderCircle, Menu, Play, RotateCcw, ShieldCheck, Sparkles, X,
} from 'lucide-react';
import { applyFitToModel, isModelReady, models, predict, type ModelId, type Predictor, type RegressionModel } from '@/data/models';
import { trainModelInBrowser, type LiveFit } from '@/lib/live-training';

const sections = [
  { id: 'ejercicios', label: 'Ejercicios', icon: FlaskConical },
  { id: 'analisis', label: 'Análisis del modelo', icon: BarChart3 },
  { id: 'entrenamiento', label: 'Entrenamiento en vivo', icon: Activity },
  { id: 'prediccion', label: 'Probar predicción', icon: Activity },
  { id: 'proceso', label: 'Proceso CRISP-DM', icon: BookOpen },
  { id: 'recursos', label: 'Entregables', icon: ArrowDownToLine },
];
const crispSteps = [
  ['01', 'Comprensión del negocio', 'Definir qué pregunta útil responde cada estimación.'],
  ['02', 'Comprensión de los datos', 'Revisar origen, columnas, calidad y cobertura.'],
  ['03', 'Preparación', 'Limpiar y transformar las columnas de entrada.'],
  ['04', 'Modelado', 'Ajustar regresión y conservar sus coeficientes.'],
  ['05', 'Evaluación', 'Medir el error con datos reservados.'],
  ['06', 'Despliegue', 'Reproducir la predicción y documentar el resultado.'],
];
const displayNumber = (value: number | null | undefined, digits = 4) =>
  value === null || value === undefined || !Number.isFinite(value) ? 'Sin publicar' : value.toLocaleString('es-ES', { maximumFractionDigits: digits });

function ScatterPlot({ model, predictor }: { model: RegressionModel; predictor: Predictor | undefined }) {
  const series = model.scatter.find((item) => item.predictorId === predictor?.id);
  const points = series?.points ?? [];
  if (!points.length) return <div className="chart-empty"><div><div className="empty-mark"><BarChart3 size={16} /></div><p className="empty-copy">Al incorporar la serie observada de este predictor, aquí aparecerán los puntos reales y su relación con el objetivo.</p></div></div>;
  const linePoints = series?.linePoints ?? [];
  const xs = [...points, ...linePoints].map((point) => point.x);
  const ys = [...points, ...linePoints].map((point) => point.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const xPad = (maxX - minX || 1) * .08, yPad = (maxY - minY || 1) * .08;
  const x = (value: number) => 42 + ((value - minX + xPad) / (maxX - minX + 2 * xPad)) * 350;
  const y = (value: number) => 158 - ((value - minY + yPad) / (maxY - minY + 2 * yPad)) * 130;
  const linePath = linePoints.map((point) => `${x(point.x)},${y(point.y)}`).join(' ');
  return <div className="plot-wrap">
    <svg viewBox="0 0 410 190" role="img" aria-label={`Dispersión observada: ${predictor?.label ?? 'predictor'} y ${model.target.label}`}>
      {[0, 1, 2, 3].map((tick) => <g key={tick}><line x1="42" x2="392" y1={28 + tick * 43} y2={28 + tick * 43} stroke="#e8ebe3" /><text x="33" y={31 + tick * 43} textAnchor="end" fill="#9ba49c" fontSize="8">{displayNumber(maxY - (maxY - minY) * tick / 3, 2)}</text></g>)}
      <line x1="42" x2="392" y1="158" y2="158" stroke="#ccd4ca" /><line x1="42" x2="42" y1="28" y2="158" stroke="#ccd4ca" />
      {linePoints.length > 1 && <polyline points={linePath} fill="none" stroke="#c36f4f" strokeWidth="2.5" strokeLinejoin="round" />}
      {points.map((point, index) => <circle key={`${point.x}-${point.y}-${index}`} cx={x(point.x)} cy={y(point.y)} r="3.2" fill="#4d907a" fillOpacity=".73" />)}
      <text x="217" y="184" textAnchor="middle" fill="#89958b" fontSize="9">{predictor?.label} · {predictor?.unit}</text>
      <text x="11" y="95" textAnchor="middle" fill="#89958b" fontSize="9" transform="rotate(-90 11 95)">{model.target.label}</text>
    </svg>
    <div className="plot-caption"><span>{points.length.toLocaleString('es-ES')} puntos · {displayNumber(model.observationCount, 0)} filas</span><span>Correlación r = {displayNumber(series?.correlation, 3)}</span></div>
  </div>;
}

function RegressionDiagram({ model }: { model: RegressionModel }) {
  return <div className="regression-diagram" role="img" aria-label={`Diagrama del modelo: predictores, regresión lineal y ${model.target.label}`}>
    <div className="diagram-inputs"><span className="diagram-label">ENTRADAS</span><div className="diagram-chips">{model.predictors.map((item) => <span key={item.id}>{item.label}</span>)}</div></div>
    <ArrowRight className="diagram-arrow" aria-hidden="true" />
    <div className="diagram-equation"><span className="diagram-label">MODELO AJUSTADO</span><strong>ŷ = β₀ + Σ βᵢxᵢ</strong><small>{model.predictors.some((item) => item.encoding) ? 'con términos cíclicos seno / coseno' : 'regresión lineal múltiple'}</small></div>
    <ArrowRight className="diagram-arrow" aria-hidden="true" />
    <div className="diagram-output"><span className="diagram-label">ESTIMACIÓN</span><strong>{model.target.label}</strong><small>{model.target.unit}</small></div>
  </div>;
}

function ModelAnalysis({ model }: { model: RegressionModel }) {
  const [selectedPredictor, setSelectedPredictor] = useState(model.predictors[0]?.id ?? '');
  const activePredictor = model.predictors.find((item) => item.id === selectedPredictor) ?? model.predictors[0];
  const metrics = [
    ['MSE · prueba', model.holdout.mse],
    ['RMSE · prueba', model.holdout.rmse],
    ['R² · prueba', model.holdout.r2],
    ['Intercepto · ajuste total', model.intercept],
  ] as const;
  return <div className="analysis-card">
    <div className="analysis-head"><div><h3>{model.label}: lectura del ajuste</h3><p>Desempeño en holdout y parámetros del modelo ajustado sobre el conjunto completo.</p></div><span className="data-badge">{isModelReady(model) ? 'MODELO CARGADO' : 'MODELO NO DISPONIBLE'}</span></div>
    <div className="analysis-body">
      <div className="metric-grid">{metrics.map(([label, value]) => <div className="metric" key={label}><div className="metric-label">{label}</div><div className={`metric-value ${value === null ? 'unfilled' : ''}`}>{displayNumber(value)}</div></div>)}</div>
      <RegressionDiagram model={model} />
      <div className="analysis-columns">
        <div className="inner-panel">
          <div className="panel-title"><span className="panel-title-main">Observado · objetivo</span><span>{activePredictor ? `X: ${activePredictor.label}` : 'SERIE REAL'}</span></div>
          {model.predictors.length > 1 && <div className="plot-selectors">{model.predictors.map((item) => <button key={item.id} className={`plot-chip ${item.id === activePredictor?.id ? 'on' : ''}`} onClick={() => setSelectedPredictor(item.id)} data-testid={`button-scatter-${model.id}-${item.id}`}>{item.label}</button>)}</div>}
          <ScatterPlot model={model} predictor={activePredictor} />
        </div>
        <div className="inner-panel">
          <div className="panel-title"><span className="panel-title-main">Coeficientes y lectura</span><span>AJUSTE TOTAL</span></div>
          <div className="coef-list">{model.predictors.map((item) => <div className="coef-row" key={item.id}><div><div className="coef-name" title={item.label}>{item.label}</div><div className="coef-interpret">{item.interpretation}</div></div><div className="coef-values"><span>{item.coefficient === null ? 'Cíclico' : displayNumber(item.coefficient, 3)}</span><small>{item.encoding ? `|β*| ${displayNumber(item.standardizedImpact, 3)}` : `β* ${displayNumber(item.standardizedCoefficient, 3)}`}</small></div><div className="coef-bar"><i style={{ width: `${Math.max(0, Math.min(100, (item.importance ?? 0) * 100))}%` }} /></div></div>)}</div>
          <div className="analysis-foot"><ShieldCheck size={13} /><span>Coeficientes completos en el `.joblib`; el modelo exportado no incluye filas de entrenamiento.</span></div>
        </div>
      </div>
      <div className="analysis-foot"><Info size={13} /><span>{model.splitDescription} R² describe la varianza explicada en prueba; RMSE conserva la unidad del objetivo y MSE queda expresado al cuadrado.</span></div>
    </div>
  </div>;
}

type TrainingState = {
  modelId: ModelId;
  status: 'running' | 'complete' | 'error';
  progress: number;
  message: string;
};

function TrainingDiagram({ progress }: { progress: number }) {
  const stages = [
    ['01', 'Dataset CSV', 'Lectura del archivo original'],
    ['02', 'Preparación', 'Columnas y codificación cíclica'],
    ['03', 'Regresión OLS', 'Ajuste y estimación de coeficientes'],
    ['04', 'Holdout', 'MSE, RMSE y R² en datos reservados'],
  ];
  return <div className="training-flow" role="img" aria-label="Flujo de entrenamiento desde el CSV hasta la evaluación holdout">
    {stages.map(([number, title, description], index) => <div className="training-flow-part" key={number}>
      <div className={`training-node ${progress >= [20, 50, 78, 100][index] ? 'done' : ''}`}>
        <span className="training-node-index">{progress >= [20, 50, 78, 100][index] ? <Check size={13} /> : number}</span>
        <strong>{title}</strong><small>{description}</small>
      </div>
      {index < stages.length - 1 && <ArrowRight className="training-arrow" aria-hidden="true" />}
    </div>)}
  </div>;
}

function Home() {
  const [activeModel, setActiveModel] = useState<ModelId>('dolar');
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [liveFits, setLiveFits] = useState<Partial<Record<ModelId, LiveFit>>>({});
  const [training, setTraining] = useState<TrainingState | null>(null);
  const baseModel = models.find((item) => item.id === activeModel)!;
  const currentFit = liveFits[activeModel];
  const model = currentFit ? applyFitToModel(baseModel, currentFit) : baseModel;
  const ready = isModelReady(model);
  const chosenPredictor = useMemo(() => model.predictors, [model]);
  const trainingForModel = training?.modelId === activeModel ? training : null;
  const isTraining = training?.status === 'running';
  const inputError = chosenPredictor.some((item) => {
    const value = formValues[`${activeModel}.${item.id}`];
    return value === undefined || value.trim() === '' || !Number.isFinite(Number(value)) ||
      (item.encoding?.kind === 'cyclic' && (
        !Number.isInteger(Number(value)) ||
        Number(value) < 1 ||
        Number(value) > item.encoding.period
      ));
  });
  const runPrediction = () => {
    const raw: Record<string, number> = {};
    model.predictors.forEach((item) => { raw[item.id] = Number(formValues[`${activeModel}.${item.id}`]); });
    setPrediction(predict(model, raw));
  };
  const resetForm = () => {
    setFormValues((current) => Object.fromEntries(Object.entries(current).filter(([key]) => !key.startsWith(`${activeModel}.`))));
    setPrediction(null);
  };
  const onModelChange = (id: ModelId) => { setActiveModel(id); setPrediction(null); };
  const runLiveTraining = async () => {
    setPrediction(null);
    setTraining({
      modelId: activeModel,
      status: 'running',
      progress: 0,
      message: 'Iniciando entrenamiento…',
    });
    try {
      const fit = await trainModelInBrowser(baseModel, (update) => {
        setTraining({ modelId: activeModel, status: 'running', ...update });
      });
      setLiveFits((current) => ({ ...current, [activeModel]: fit }));
      setTraining({
        modelId: activeModel,
        status: 'complete',
        progress: 100,
        message: `Entrenado en ${(fit.elapsedMs / 1000).toFixed(2)} s. El modelo final ya alimenta la predicción.`,
      });
    } catch (error) {
      setTraining({
        modelId: activeModel,
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'No se pudo entrenar este modelo.',
      });
    }
  };
  return <div className="app-shell">
    <aside className={`sidebar ${menuOpen ? 'open' : ''}`}>
      <div className="brand"><div className="brand-mark"><BrainCircuit size={19} /></div><div><div className="brand-name">Laboratorio<br />de minería</div><div className="brand-caption">Cuaderno CRISP-DM</div></div></div>
      <div className="nav-label">CUADERNO</div>
      <nav aria-label="Navegación principal">{sections.map(({ id, label, icon: Icon }, index) => <a key={id} href={`#${id}`} onClick={() => setMenuOpen(false)} className={`side-link ${index === 0 ? 'active' : ''}`} data-testid={`link-section-${id}`}><Icon /><span>{label}</span></a>)}</nav>
      <div className="sidebar-spacer" />
      <div className="side-note"><strong>Una predicción trazable.</strong>Cada resultado se reproduce desde columnas originales y coeficientes exportados, sin llamadas a un servicio externo.</div>
      <div className="side-user"><div className="user-dot">LAB</div><div className="user-meta">Entorno de aprendizaje<span>Regresión · ejecución local</span></div></div>
    </aside>
    <main className="main-area">
      <div className="topbar">
        <div className="mobile-brand"><span className="brand-mark"><BrainCircuit size={16} /></span> Cuaderno CRISP-DM</div>
        <div className="crumb"><span>Laboratorio</span><ChevronRight size={12} /><b>Resumen del proyecto</b></div>
        <div className="top-actions"><span className="status-pill"><i className="status-light" /> PROCESAMIENTO LOCAL</span><button className="icon-button menu-toggle" aria-label={menuOpen ? 'Cerrar menú' : 'Abrir menú'} onClick={() => setMenuOpen(!menuOpen)} data-testid="button-menu">{menuOpen ? <X size={16} /> : <Menu size={16} />}</button><button className="icon-button" aria-label="Ir a recursos" onClick={() => document.getElementById('recursos')?.scrollIntoView({ behavior: 'smooth' })} data-testid="button-go-resources"><ArrowDownToLine size={15} /></button></div>
      </div>
      <div className="content fade-in">
        <section className="hero-row" aria-labelledby="main-heading"><div><div className="eyebrow">CUADERNO DE ANÁLISIS · 03 EJERCICIOS</div><h1 id="main-heading" className="page-title">De los datos a la<br />predicción explicable.</h1><p className="lead">Tres ejercicios reales de regresión, recorridos con CRISP-DM. Explora cómo se entrenó cada modelo, qué error comete y cómo cada columna se convierte en una predicción.</p></div><div className="hero-index"><strong>01</strong><span />03 <span>LAB</span></div></section>
        <div className="overview-strip" aria-label="Resumen del laboratorio">
          <div className="overview-cell"><div className="overview-value">03</div><div className="overview-label">casos de regresión</div></div>
          <div className="overview-cell"><div className="overview-value">CRISP-DM</div><div className="overview-label">proceso de trabajo</div></div>
          <div className="overview-cell"><div className="overview-value">Local</div><div className="overview-label">predicción sin API</div></div>
          <div className="overview-cell"><div className="overview-value">Trazable</div><div className="overview-label">datos, ajuste y métricas</div></div>
        </div>
        <section id="ejercicios" className="section-block">
          <div className="section-heading"><div><h2 className="section-title">Tres preguntas, tres modelos</h2><div className="section-sub">Elige un caso para inspeccionar sus datos y parámetros.</div></div><a className="link-small" href="#analisis">Abrir análisis <ArrowRight size={13} /></a></div>
          <div className="model-grid">{models.map((item, index) => <article className="model-card" key={item.id}>
            <div className="model-card-top"><span className="model-symbol">{index === 0 ? <Activity size={17} /> : index === 1 ? <FlaskConical size={17} /> : <BarChart3 size={17} />}</span><span className="model-id">REG · 0{index + 1}</span></div>
            <h3 className="model-title">{item.label}</h3><p className="model-desc">Predicción por regresión lineal múltiple</p>
            <div className="model-card-bottom"><span className="model-target">Objetivo · {item.target.unit}</span><button className="model-open" onClick={() => { onModelChange(item.id); document.getElementById('analisis')?.scrollIntoView({ behavior: 'smooth' }); }} data-testid={`button-open-model-${item.id}`}>Ver ejercicio <ArrowRight size={12} /></button></div>
          </article>)}</div>
        </section>
        <section id="analisis" className="section-block">
          <div className="section-heading"><div><h2 className="section-title">Anatomía del modelo</h2><div className="section-sub">Una lectura auditable: validación, coeficientes y puntos observados.</div></div><span className="data-badge">HOLDOUT + AJUSTE COMPLETO</span></div>
          <div className="analysis-layout">
            <div className="model-tabs" role="tablist" aria-label="Seleccionar ejercicio">{models.map((item, index) => <button role="tab" aria-selected={activeModel === item.id} key={item.id} className={`model-tab ${activeModel === item.id ? 'selected' : ''}`} onClick={() => onModelChange(item.id)} data-testid={`tab-model-${item.id}`}><span className="tab-num">0{index + 1}</span><span className="tab-name">{item.label}</span></button>)}</div>
            <ModelAnalysis key={model.id} model={model} />
          </div>
        </section>
        <section id="entrenamiento" className="section-block">
          <div className="section-heading"><div><h2 className="section-title">Entrenamiento en vivo</h2><div className="section-sub">Carga el CSV real, ajusta una regresión y valida con el mismo holdout reproducible.</div></div><span className="data-badge">EN ESTE NAVEGADOR</span></div>
          <div className="training-panel">
            <div className="training-panel-head">
              <div><h3>Reentrenar · {model.label}</h3><p>El ajuste se ejecuta aquí mismo, sin API. Los resultados actualizan los coeficientes usados por el formulario de predicción.</p></div>
              <button className="primary-button training-button" onClick={runLiveTraining} disabled={isTraining} data-testid="button-run-training">
                {isTraining ? <LoaderCircle size={14} className="spin" /> : currentFit ? <RotateCcw size={14} /> : <Play size={14} />}
                {isTraining ? 'Entrenando…' : currentFit ? 'Reentrenar modelo' : 'Entrenar modelo'}
              </button>
            </div>
            <TrainingDiagram progress={trainingForModel?.progress ?? 0} />
            <div className="training-progress" aria-hidden="true"><span style={{ width: `${trainingForModel?.progress ?? 0}%` }} /></div>
            <div className={`training-status ${trainingForModel?.status ?? ''}`} role="status" aria-live="polite" data-testid="training-status">
              {trainingForModel?.status === 'complete' ? <Check size={14} /> : trainingForModel?.status === 'running' ? <LoaderCircle size={14} className="spin" /> : <Info size={14} />}
              <span>{trainingForModel?.message ?? `Listo para ajustar ${model.observationCount?.toLocaleString('es-CO')} observaciones del CSV original.`}</span>
            </div>
            <div className="notice training-notice"><Info size={14} /><span>Las métricas MSE, RMSE y R² se calculan sobre el holdout. Los coeficientes finales se reajustan con todas las filas; esas mismas columnas y coeficientes alimentan la predicción interactiva.</span></div>
          </div>
        </section>
        <section id="prediccion" className="section-block">
          <div className="section-heading"><div><h2 className="section-title">Prueba una predicción</h2><div className="section-sub">Escribe los valores de las columnas originales. El cálculo ocurre en este navegador.</div></div><span className="data-badge">SIN API</span></div>
          <div className="predict-layout">
            <div className="predict-form"><h3>Entradas del modelo</h3><p className="form-intro">Ejercicio seleccionado: <strong>{model.label}</strong>. Las columnas y unidades se leen del modelo exportado.</p>
              {model.predictors.length ? <>
                <div className="field-grid">{model.predictors.map((item) => <div className="field" key={item.id}><label htmlFor={`input-${item.id}`}>{item.label}<span>{item.unit}</span></label><input id={`input-${item.id}`} type="number" step={item.encoding?.kind === 'cyclic' ? 1 : 'any'} min={item.encoding?.kind === 'cyclic' ? 1 : undefined} max={item.encoding?.kind === 'cyclic' ? item.encoding.period : undefined} placeholder={item.encoding?.kind === 'cyclic' ? `1–${item.encoding.period}` : `${displayNumber(item.defaultValue, 2)} · mediana`} value={formValues[`${activeModel}.${item.id}`] ?? ''} onChange={(event) => { setFormValues((current) => ({ ...current, [`${activeModel}.${item.id}`]: event.target.value })); setPrediction(null); }} data-testid={`input-predict-${activeModel}-${item.id}`} /></div>)}</div>
                {model.id === 'energia' && <div className="notice"><Info size={14} /><span>Hora (1–24) y día (1–7) se conservan como columnas de entrada; se transforman en componentes seno y coseno al calcular, para respetar su naturaleza cíclica.</span></div>}
              </> : <div className="form-empty"><Database size={18} /><div><strong>Columnas de entrada no disponibles</strong><span>No se puede calcular una predicción hasta cargar los predictores del modelo.</span></div></div>}
              <div className="form-actions"><button className="primary-button" onClick={runPrediction} disabled={!ready || inputError} data-testid="button-run-prediction"><Sparkles size={14} /> Calcular estimación</button><button className="quiet-button" onClick={resetForm} data-testid="button-reset-prediction"><RotateCcw size={13} /> Limpiar valores</button></div>
              {!ready && <div className="input-note"><CircleHelp size={13} /> La predicción se activará cuando estén disponibles el intercepto y todos los coeficientes del modelo.</div>}
            </div>
            <div className="result-panel"><div><div className="result-top"><h3>Resultado estimado</h3><span className="result-stamp">{prediction !== null ? 'CÁLCULO LOCAL' : 'ESPERANDO ENTRADAS'}</span></div>
              {prediction !== null ? <><div className="result-value" data-testid="text-prediction-result">{displayNumber(prediction, 5)}</div><div className="result-unit">{model.target.unit}</div></> : <p className="result-placeholder">Ingresa los valores de entrada para ver la estimación calculada con los coeficientes actuales.</p>}</div>
              <div className="result-meta">{prediction !== null ? `Objetivo: ${model.target.label}. La estimación no sustituye la evaluación sobre datos reservados.` : `Modelo: ${currentFit ? 'reentrenado en el navegador' : 'ajuste completo exportado'}.`}</div>
            </div>
          </div>
        </section>
        <section id="proceso" className="section-block">
          <div className="section-heading"><div><h2 className="section-title">El recorrido CRISP-DM</h2><div className="section-sub">Un ciclo de trabajo: del problema a una entrega reproducible.</div></div><BookOpen size={17} color="#628b7a" /></div>
          <div className="crisp-process-diagram" role="img" aria-label="Diagrama de las seis etapas del proceso CRISP-DM">
            {crispSteps.map(([number, title], index) => <div className="crisp-process-part" key={number}>
              <div className="crisp-process-node"><span>{number}</span><strong>{title}</strong></div>
              {index < crispSteps.length - 1 && <ArrowRight className="crisp-process-arrow" aria-hidden="true" />}
            </div>)}
          </div>
          <div className="crisp-grid">{crispSteps.map(([number, title, description]) => <article className="crisp-step" key={number}><div className="step-index">{number} / 06</div><div className="step-name">{title}</div><p className="step-copy">{description}</p></article>)}</div>
        </section>
        <section id="recursos" className="section-block">
          <div className="section-heading"><div><h2 className="section-title">Entregables del laboratorio</h2><div className="section-sub">Descarga el informe, el código fuente y los artefactos de cada ejercicio.</div></div></div>
          <div className="deliverable-grid">
            <a className="deliverable" href={model.deliverables.report} download="informe-laboratorio.pdf" data-testid="link-download-report-pdf"><span className="deliverable-icon"><FileText size={16} /></span><span className="deliverable-copy"><strong>Informe completo</strong><span>PDF · resultados y metodología</span></span><ArrowDownToLine size={13} className="deliverable-arrow" /></a>
            <a className="deliverable" href={model.deliverables.reportMarkdown} download="informe-laboratorio.md" data-testid="link-download-report-md"><span className="deliverable-icon"><FileText size={16} /></span><span className="deliverable-copy"><strong>Informe editable</strong><span>Markdown · mismo contenido del PDF</span></span><ArrowDownToLine size={13} className="deliverable-arrow" /></a>
            <a className="deliverable" href={model.deliverables.source} download="codigo-entrenamiento.zip" data-testid="link-download-source"><span className="deliverable-icon"><FileCode2 size={16} /></span><span className="deliverable-copy"><strong>Código reproducible</strong><span>ZIP · Python y dependencias</span></span><ArrowDownToLine size={13} className="deliverable-arrow" /></a>
            {models.map((item) => <div className="deliverable-pair" key={item.id}>
              <a className="deliverable" href={item.deliverables.dataset} download data-testid={`link-download-dataset-${item.id}`}><span className="deliverable-icon"><Database size={16} /></span><span className="deliverable-copy"><strong>Datos · {item.label}</strong><span>CSV original · {displayNumber(item.observationCount, 0)} filas</span></span><ArrowDownToLine size={13} className="deliverable-arrow" /></a>
              <a className="deliverable" href={item.deliverables.exportedModel} download data-testid={`link-download-model-${item.id}`}><span className="deliverable-icon"><ArrowDownToLine size={16} /></span><span className="deliverable-copy"><strong>Modelo · {item.label}</strong><span>scikit-learn · .joblib</span></span><ArrowDownToLine size={13} className="deliverable-arrow" /></a>
            </div>)}
          </div>
          <div className="notice resource-notice"><Info size={14} /><span>El archivo `.joblib` contiene el ajuste de scikit-learn; el ZIP incluye el código de entrenamiento y la transformación cíclica para reproducirlo.</span></div>
        </section>
        <footer className="footnote"><span>Cuaderno de minería de datos · Regresión supervisada · <a href="#proceso">CRISP-DM</a></span><span>Los coeficientes y métricas publicados se calcularon desde los CSV suministrados.</span></footer>
      </div>
    </main>
  </div>;
}

function NotFound() {
  return <main className="not-found"><div className="eyebrow">CUADERNO CRISP-DM</div><h1 className="page-title">No encontramos esta página.</h1><a href="/">Volver al laboratorio <ArrowRight size={14} /></a></main>;
}

function App() {
  return <Switch><Route path="/" component={Home} /><Route component={NotFound} /></Switch>;
}
export default App;