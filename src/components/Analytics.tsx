import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, Cell, PieChart, Pie,
} from 'recharts';
import { Users, TrendingUp, Award, AlertCircle, Loader2 } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'https://passos-magicos-api-chcj.onrender.com';

// ── Tipos ─────────────────────────────────────────────────────────────────
interface Stats {
  total_alunos: number;
  em_risco: number;
  pct_risco: number;
  inde_medio: number;
  inde_por_ano: { ano: number; INDE: number }[];
  dist_pedras: { pedra: string; quantidade: number }[];
}
interface EvolRow { ano: number; INDE?: number; IAA?: number; IEG?: number; IDA?: number; IPV?: number; IAN?: number }
interface RiscoFase { fase: number; total: number; em_risco: number; pct_risco: number }

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface Metrics {
  accuracy: number;
  roc_auc: number;
  f1: number;
  precision: number;
  recall: number;
}

// ── Cores ──────────────────────────────────────────────────────────────────
const PEDRA_COLORS: Record<string, string> = {
  'Quartzo': '#94a3b8', 'Ágata': '#38bdf8', 'Ametista': '#a855f7', 'Topázio': '#fbbf24',
};
const LINES = [
  { key: 'INDE', color: '#6366f1', label: 'INDE' },
  { key: 'IAA', color: '#10b981', label: 'IAA' },
  { key: 'IEG', color: '#f59e0b', label: 'IEG' },
  { key: 'IDA', color: '#ef4444', label: 'IDA' },
  { key: 'IPV', color: '#3b82f6', label: 'IPV' },
];

// ── Hook reutilizável ──────────────────────────────────────────────────────
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
function KpiCard({ title, value, sub, icon, accent }: {
  title: string; value: string; sub?: string; icon: React.ReactNode; accent: string;
}) {
  return (
    <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5 flex items-start space-x-4 hover:shadow-md transition-shadow">
      <div className={`p-3 rounded-xl shrink-0 ${accent}`}>{icon}</div>
      <div className="min-w-0">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide truncate">{title}</p>
        <p className="text-2xl font-black text-slate-800 mt-0.5">{value}</p>
        {sub && <p className="text-xs text-slate-400 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

function Card({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white rounded-2xl border border-slate-100 shadow-sm p-6 ${className}`}>
      <h3 className="text-sm font-bold text-slate-700 uppercase tracking-wide mb-5">{title}</h3>
      {children}
    </div>
  );
}

function Skeleton() {
  return (
    <div className="flex items-center justify-center h-64 border border-slate-50 bg-slate-50/50 rounded-xl">
      <Loader2 className="w-7 h-7 animate-spin text-indigo-300" />
    </div>
  );
}

const fmtTooltip = (v: number) => v.toFixed(2);

// ── Componente principal ───────────────────────────────────────────────────
export default function Analytics() {
  const { data: stats, loading: lStats } = useFetch<Stats>(`${API_URL}/analytics/stats`);
  const { data: evolucao, loading: lEvolucao } = useFetch<EvolRow[]>(`${API_URL}/analytics/evolucao`);
  const { data: riscoFase, loading: lRisco } = useFetch<RiscoFase[]>(`${API_URL}/analytics/risco-por-fase`);

  // Model endpoints
  const { data: fi, loading: lFi } = useFetch<FeatureImportance[]>(`${API_URL}/feature-importance`);
  const { data: metrics, loading: lMetrics } = useFetch<Metrics>(`${API_URL}/metrics`);

  const dataPedras = (stats?.dist_pedras ?? []).map(p => ({
    name: p.pedra, value: p.quantidade, color: PEDRA_COLORS[p.pedra] ?? '#6366f1',
  }));

  const topazioQtd = dataPedras.find(p => p.name === 'Topázio')?.value;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">

      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Total de Alunos"
          value={stats ? stats.total_alunos.toLocaleString('pt-BR') : '—'}
          icon={<Users className="w-5 h-5 text-blue-600" />}
          accent="bg-blue-50"
        />
        <KpiCard
          title="INDE Médio Geral"
          value={stats ? stats.inde_medio.toFixed(2) : '—'}
          sub="Índice de Desenvolvimento"
          icon={<TrendingUp className="w-5 h-5 text-emerald-600" />}
          accent="bg-emerald-50"
        />
        <KpiCard
          title="Alunos Topázio"
          value={topazioQtd != null ? topazioQtd.toLocaleString('pt-BR') : '—'}
          sub="Melhor classificação"
          icon={<Award className="w-5 h-5 text-amber-600" />}
          accent="bg-amber-50"
        />
        <KpiCard
          title="Em Risco"
          value={stats ? stats.em_risco.toLocaleString('pt-BR') : '—'}
          sub={stats ? `${stats.pct_risco}% do total` : undefined}
          icon={<AlertCircle className="w-5 h-5 text-red-600" />}
          accent="bg-red-50"
        />
      </div>

      {/* ── Linha 1: Evolução + Pedras ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <Card title="Evolução dos Indicadores por Ano">
          {lEvolucao ? <Skeleton /> : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={evolucao ?? []} margin={{ top: 4, right: 16, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="ano" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                  <YAxis domain={[4, 10]} axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip
                    formatter={fmtTooltip}
                    contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 20px rgb(0 0 0 / .08)', fontSize: 12 }}
                  />
                  <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
                  {LINES.map(l => (
                    <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color}
                      strokeWidth={2.5} dot={{ r: 3, fill: l.color }} activeDot={{ r: 5 }} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Distribuição por Classificação (Pedras)">
          {lStats ? <Skeleton /> : (
            <div className="h-64 flex items-center">
              <div className="w-1/2">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie data={dataPedras} dataKey="value" cx="50%" cy="50%"
                      innerRadius={50} outerRadius={80} paddingAngle={3}>
                      {dataPedras.map((e, i) => <Cell key={i} fill={e.color} />)}
                    </Pie>
                    <Tooltip
                      formatter={(v: number) => [v.toLocaleString('pt-BR'), 'Alunos']}
                      contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 20px rgb(0 0 0 / .08)', fontSize: 12 }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="w-1/2 space-y-3 pr-2">
                {dataPedras.map(p => (
                  <div key={p.name} className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="w-3 h-3 rounded-full shrink-0" style={{ background: p.color }} />
                      <span className="text-sm text-slate-600 font-medium">{p.name}</span>
                    </div>
                    <span className="text-sm font-bold text-slate-800">{p.value.toLocaleString('pt-BR')}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* ── Linha 2: INDE por Ano + Risco por Fase ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <Card title="INDE Médio por Ano">
          {lStats ? <Skeleton /> : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats?.inde_por_ano ?? []} margin={{ top: 4, right: 16, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                  <XAxis dataKey="ano" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 10]} axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <Tooltip
                    formatter={(v: number) => [v.toFixed(3), 'INDE médio']}
                    contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 20px rgb(0 0 0 / .08)', fontSize: 12 }}
                  />
                  <Bar dataKey="INDE" radius={[6, 6, 0, 0]}>
                    {(stats?.inde_por_ano ?? []).map((_, i) => (
                      <Cell key={i} fill={['#818cf8', '#6366f1', '#4f46e5'][i % 3]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Risco de Defasagem por Fase">
          {lRisco ? <Skeleton /> : (
            <div className="h-64 flex flex-col">
              <div className="flex-1">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={riscoFase ?? []} margin={{ top: 4, right: 16, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="fase" axisLine={false} tickLine={false} tick={{ fontSize: 11 }}
                      tickFormatter={v => v === 0 ? 'ALFA' : `F${v}`} />
                    <YAxis unit="%" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                    <Tooltip
                      formatter={(v: number) => [`${v.toFixed(1)}%`, '% em risco']}
                      labelFormatter={v => v === 0 ? 'Fase ALFA' : `Fase ${v}`}
                      contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 20px rgb(0 0 0 / .08)', fontSize: 12 }}
                    />
                    <Bar dataKey="pct_risco" radius={[6, 6, 0, 0]}>
                      {(riscoFase ?? []).map((r, i) => (
                        <Cell key={i} fill={r.pct_risco > 70 ? '#ef4444' : r.pct_risco > 40 ? '#f59e0b' : '#10b981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 flex items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" />≤ 40%</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-amber-500 inline-block" />40–70%</span>
                <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-red-500 inline-block" />&gt; 70%</span>
              </div>
            </div>
          )}
        </Card>

      </div>

      {/* ── Linha 3: ML Insights ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        <Card title="Importância das Features (Modelo ML)">
          {lFi ? <Skeleton /> : !fi ? (
            <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
              Modelo ainda não treinado.
            </div>
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={fi} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                  <XAxis type="number" domain={[0, 'auto']} axisLine={false} tickLine={false} />
                  <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} width={50} />
                  <Tooltip formatter={(v: number) => (v * 100).toFixed(1) + '%'} contentStyle={{ borderRadius: '8px', border: 'none' }} />
                  <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card title="Métricas do Modelo Preditivo">
          {lMetrics ? <Skeleton /> : !metrics ? (
            <div className="h-64 flex items-center justify-center text-slate-400 text-sm">
              Modelo ainda não treinado.
            </div>
          ) : (
            <div className="space-y-4 mt-2">
              {[
                { label: 'ROC-AUC', value: metrics.roc_auc, color: 'bg-indigo-500' },
                { label: 'Accuracy', value: metrics.accuracy, color: 'bg-emerald-500' },
                { label: 'F1-Score', value: metrics.f1, color: 'bg-amber-500' },
                { label: 'Precision', value: metrics.precision, color: 'bg-blue-500' },
                { label: 'Recall', value: metrics.recall, color: 'bg-rose-500' },
              ].map(m => (
                <div key={m.label}>
                  <div className="flex justify-between text-sm mb-1.5">
                    <span className="font-medium text-slate-700">{m.label}</span>
                    <span className="font-bold text-slate-800">{(m.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden">
                    <div className={`h-full ${m.color} rounded-full`} style={{ width: `${m.value * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

      </div>

      {/* ── Linha 4: Tabela risco por fase ── */}
      {riscoFase && riscoFase.length > 0 && (
        <Card title="Detalhamento — Risco por Fase">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b border-slate-100">
                  <th className="pb-3 font-semibold text-slate-500 text-xs uppercase tracking-wide">Fase</th>
                  <th className="pb-3 font-semibold text-slate-500 text-xs uppercase tracking-wide text-right">Total</th>
                  <th className="pb-3 font-semibold text-slate-500 text-xs uppercase tracking-wide text-right">Em Risco</th>
                  <th className="pb-3 font-semibold text-slate-500 text-xs uppercase tracking-wide text-right">% Risco</th>
                  <th className="pb-3 pl-4"></th>
                </tr>
              </thead>
              <tbody>
                {riscoFase.map(r => (
                  <tr key={r.fase} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                    <td className="py-2.5 font-semibold text-slate-700">
                      {r.fase === 0 ? 'ALFA' : `Fase ${r.fase}`}
                    </td>
                    <td className="py-2.5 text-right text-slate-600">{r.total.toLocaleString('pt-BR')}</td>
                    <td className="py-2.5 text-right text-slate-600">{r.em_risco.toLocaleString('pt-BR')}</td>
                    <td className={`py-2.5 text-right font-bold ${r.pct_risco > 70 ? 'text-red-600' : r.pct_risco > 40 ? 'text-amber-600' : 'text-emerald-600'
                      }`}>{r.pct_risco.toFixed(1)}%</td>
                    <td className="py-2.5 pl-4 w-32">
                      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${r.pct_risco > 70 ? 'bg-red-500' : r.pct_risco > 40 ? 'bg-amber-400' : 'bg-emerald-500'}`}
                          style={{ width: `${Math.min(r.pct_risco, 100)}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

    </div>
  );
}
