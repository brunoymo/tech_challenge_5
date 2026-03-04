import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Cell, ReferenceLine,
} from 'recharts';
import { CheckCircle2, Loader2, Target, Layers, TrendingUp } from 'lucide-react';

const API_URL = 'https://passos-magicos-api-chcj.onrender.com';

// ── Tipos ──────────────────────────────────────────────────────────────────
interface Metrics {
  accuracy: number;
  roc_auc: number;
  f1: number;
  precision: number;
  recall: number;
  confusion_matrix: number[][];
  roc_curve: { fpr: number[]; tpr: number[] };
  n_amostras: number;
  n_treino: number;
  n_teste: number;
}
interface FeatImportance { feature: string; importance: number }

// ── Hook ───────────────────────────────────────────────────────────────────
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    fetch(url)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [url]);
  return { data, loading };
}

// ── Sub-componentes ────────────────────────────────────────────────────────
function Card({ title, sub, children, className = '' }: {
  title: string; sub?: string; children: React.ReactNode; className?: string;
}) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-100 shadow-sm p-6 ${className}`}>
      <div className="mb-5">
        <h3 className="text-sm font-bold text-slate-700 uppercase tracking-wide">{title}</h3>
        {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
      </div>
      {children}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="flex items-center justify-center h-52">
      <Loader2 className="w-7 h-7 animate-spin text-indigo-300" />
    </div>
  );
}

function Offline() {
  return (
    <div className="flex flex-col items-center justify-center h-52 text-slate-300 space-y-2">
      <Layers className="w-10 h-10" />
      <p className="text-sm font-medium">Dados não disponíveis</p>
      <p className="text-xs">Execute <code className="bg-slate-100 px-1 rounded">train.py</code> e reinicie a API.</p>
    </div>
  );
}

// ── Componente principal ───────────────────────────────────────────────────
export default function ModelMetrics() {
  const { data: metrics, loading: lMetrics } = useFetch<Metrics>(`${API_URL}/metrics`);
  const { data: fi, loading: lFi } = useFetch<FeatImportance[]>(`${API_URL}/feature-importance`);

  // Curva ROC como array de pontos
  const rocData = metrics
    ? metrics.roc_curve.fpr.map((fpr, i) => ({ fpr, tpr: metrics.roc_curve.tpr[i] }))
    : [];

  // Feature importance — ordena crescente para bar horizontal parecer melhor
  const fiSorted = fi ? [...fi].sort((a, b) => a.importance - b.importance) : [];

  // Métricas principais
  const METRIC_ROWS = metrics
    ? [
      { label: 'ROC-AUC', value: metrics.roc_auc, color: 'bg-indigo-500', text: 'text-indigo-700' },
      { label: 'Accuracy', value: metrics.accuracy, color: 'bg-emerald-500', text: 'text-emerald-700' },
      { label: 'F1-Score', value: metrics.f1, color: 'bg-amber-500', text: 'text-amber-700' },
      { label: 'Precision', value: metrics.precision, color: 'bg-blue-500', text: 'text-blue-700' },
      { label: 'Recall', value: metrics.recall, color: 'bg-rose-500', text: 'text-rose-700' },
    ]
    : [];

  return (
    <div className="space-y-6">

      {/* ── KPIs resumo ── */}
      {metrics && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
          {METRIC_ROWS.map(m => (
            <div key={m.label} className="bg-white rounded-2xl border border-slate-100 shadow-sm p-4 text-center hover:shadow-md transition-shadow">
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide">{m.label}</p>
              <p className={`text-3xl font-black mt-1 ${m.text}`}>{(m.value * 100).toFixed(1)}%</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Info do dataset ── */}
      {metrics && (
        <div className="bg-indigo-50 border border-indigo-100 rounded-2xl px-6 py-4 flex flex-wrap gap-6">
          <div>
            <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide">Dataset total</p>
            <p className="text-lg font-black text-indigo-700">{metrics.n_amostras.toLocaleString('pt-BR')} amostras</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide">Treino (80%)</p>
            <p className="text-lg font-black text-indigo-700">{metrics.n_treino.toLocaleString('pt-BR')}</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide">Teste (20%)</p>
            <p className="text-lg font-black text-indigo-700">{metrics.n_teste.toLocaleString('pt-BR')}</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide">Algoritmo</p>
            <p className="text-lg font-black text-indigo-700">Random Forest</p>
          </div>
          <div>
            <p className="text-xs font-semibold text-indigo-400 uppercase tracking-wide">Estimadores</p>
            <p className="text-lg font-black text-indigo-700">200 árvores</p>
          </div>
        </div>
      )}

      {/* ── Linha 1: Métricas bars + Matriz de confusão ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <Card title="Métricas de Desempenho" sub="Conjunto de teste — 20% dos dados">
          {lMetrics ? <Skeleton /> : !metrics ? <Offline /> : (
            <div className="space-y-4">
              {METRIC_ROWS.map(m => (
                <div key={m.label}>
                  <div className="flex justify-between items-center mb-1.5">
                    <span className="text-sm font-semibold text-slate-600">{m.label}</span>
                    <span className={`text-sm font-black ${m.text}`}>{(m.value * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${m.color}`}
                      style={{ width: `${m.value * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card title="Matriz de Confusão" sub="Previsão vs. realidade no conjunto de teste">
          {lMetrics ? <Skeleton /> : !metrics ? <Offline /> : (
            <div className="flex flex-col items-center justify-center h-full space-y-3 py-2">
              {/* Labels eixo X */}
              <div className="flex w-64 justify-center space-x-2 text-xs font-semibold text-slate-400 pl-14">
                <span className="w-24 text-center">Previsto: 0</span>
                <span className="w-24 text-center">Previsto: 1</span>
              </div>
              {(['Sem Risco', 'Em Risco'] as const).map((rowLabel, ri) => (
                <div key={ri} className="flex items-center space-x-2">
                  <span className="w-12 text-right text-xs font-semibold text-slate-400 shrink-0">{rowLabel}</span>
                  {metrics.confusion_matrix[ri].map((val, ci) => {
                    const isCorrect = ri === ci;
                    const total = metrics.n_teste;
                    const pct = ((val / total) * 100).toFixed(1);
                    return (
                      <div
                        key={ci}
                        className={`w-24 h-16 rounded-xl flex flex-col items-center justify-center border-2 ${isCorrect
                          ? 'bg-emerald-50 border-emerald-200'
                          : val === 0 ? 'bg-slate-50 border-slate-100' : 'bg-red-50 border-red-200'
                          }`}
                      >
                        <span className={`text-2xl font-black ${isCorrect ? 'text-emerald-700' : val === 0 ? 'text-slate-300' : 'text-red-600'}`}>
                          {val}
                        </span>
                        <span className={`text-[10px] font-medium ${isCorrect ? 'text-emerald-500' : val === 0 ? 'text-slate-300' : 'text-red-400'}`}>
                          {pct}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              ))}
              <div className="flex justify-center space-x-5 text-xs text-slate-400 pt-1">
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-200 inline-block" />Correto</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-red-200 inline-block" />Erro</span>
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* ── Linha 2: Curva ROC + Feature Importance ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <Card
          title="Curva ROC"
          sub={metrics ? `AUC = ${metrics.roc_auc.toFixed(4)}` : ''}
        >
          {lMetrics ? <Skeleton /> : !metrics ? <Offline /> : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rocData} margin={{ top: 4, right: 16, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} axisLine={false} tickLine={false}
                    tick={{ fontSize: 11 }} label={{ value: 'FPR', position: 'insideBottomRight', offset: -4, fontSize: 11, fill: '#94a3b8' }} />
                  <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tick={{ fontSize: 11 }}
                    label={{ value: 'TPR', angle: -90, position: 'insideLeft', offset: 12, fontSize: 11, fill: '#94a3b8' }} />
                  <Tooltip
                    formatter={(v: number) => v.toFixed(4)}
                    labelFormatter={v => `FPR: ${Number(v).toFixed(4)}`}
                    contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 20px rgb(0 0 0 / .08)', fontSize: 11 }}
                  />
                  {/* Linha diagonal de referência (classificador aleatório) */}
                  <Line dataKey="tpr" stroke="#6366f1" strokeWidth={2.5} dot={false} />
                  <ReferenceLine stroke="#cbd5e1" strokeDasharray="5 4" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Importância das Features" sub="Contribuição de cada indicador para o modelo">
          {lFi ? <Skeleton /> : !fi ? <Offline /> : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fiSorted} layout="vertical" margin={{ top: 4, right: 40, left: 10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 11 }}
                    tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false}
                    tick={{ fontSize: 12, fontWeight: 600, fill: '#475569' }} width={40} />
                  <Tooltip
                    formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, 'Importância']}
                    contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 20px rgb(0 0 0 / .08)', fontSize: 12 }}
                  />
                  <Bar dataKey="importance" radius={[0, 6, 6, 0]} label={{ position: 'right', formatter: (v: number) => `${(v * 100).toFixed(1)}%`, fontSize: 10, fill: '#94a3b8' }}>
                    {fiSorted.map((_, i) => {
                      const colors = ['#4f46e5', '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe', '#e0e7ff', '#f0f4ff', '#f8fafc', '#f1f5f9'];
                      return <Cell key={i} fill={colors[Math.min(i, colors.length - 1)]} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>
      </div>

      {/* ── Nota sobre IAN ── */}
      {fi && fi.length > 0 && fi[0].feature === 'IAN' && (
        <div className="bg-amber-50 border border-amber-100 rounded-2xl px-6 py-4 flex items-start space-x-3">
          <Target className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-bold text-amber-800">Sobre a dominância do IAN</p>
            <p className="text-sm text-amber-700 mt-1">
              O <strong>IAN</strong> (Índice de Adequação ao Nível) é calculado a partir da relação entre a fase atual e a fase ideal do aluno, sendo matematicamente correlacionado com a defasagem.
              Isso explica sua importância de {(fi[0].importance * 100).toFixed(1)}% no modelo. Em um cenário prospectivo (prever antes do cálculo do IAN),
              os demais indicadores como INDE, Fase e IDA ganham mais relevância.
            </p>
          </div>
        </div>
      )}

    </div>
  );
}
