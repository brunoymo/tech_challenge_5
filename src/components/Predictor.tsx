import React, { useState } from 'react';
import { BrainCircuit, AlertTriangle, CheckCircle2, Loader2, RotateCcw, Info } from 'lucide-react';

const API_URL = 'https://passos-magicos-api-chcj.onrender.com';

// ── Configuração dos campos ────────────────────────────────────────────────
const FIELDS = [
  { name: 'Idade', label: 'Idade', step: '1', min: 6, max: 30, desc: 'Idade do aluno em anos' },
  { name: 'Fase', label: 'Fase', step: '1', min: 0, max: 8, desc: '0 = ALFA, 1–8 = fases escolares' },
  { name: 'IAA', label: 'IAA · Autoavaliação', step: '0.1', min: 0, max: 10, desc: 'Índice de Autoavaliação do Aluno' },
  { name: 'IEG', label: 'IEG · Engajamento', step: '0.1', min: 0, max: 10, desc: 'Índice de Engajamento' },
  { name: 'IPS', label: 'IPS · Psicossocial', step: '0.1', min: 0, max: 10, desc: 'Índice Psicossocial' },
  { name: 'IDA', label: 'IDA · Aprendizagem', step: '0.1', min: 0, max: 10, desc: 'Índice de Desenvolvimento do Aprendizado' },
  { name: 'IPV', label: 'IPV · Ponto de Virada', step: '0.1', min: 0, max: 10, desc: 'Índice do Ponto de Virada' },
  { name: 'IAN', label: 'IAN · Adequação', step: '0.1', min: 0, max: 10, desc: 'Índice de Adequação ao Nível' },
  { name: 'INDE', label: 'INDE · Índice Geral', step: '0.1', min: 0, max: 10, desc: 'Índice de Desenvolvimento Educacional', highlight: true as const },
];

type FieldName = (typeof FIELDS)[number]['name'];
type FormData = Record<FieldName, number>;

const DEFAULT: FormData = {
  Idade: 14, Fase: 3, IAA: 8.5, IEG: 7.0, IPS: 6.5,
  IDA: 7.5, IPV: 8.0, IAN: 7.0, INDE: 7.4,
};

// ── Componente principal ───────────────────────────────────────────────────
export default function Predictor() {
  const [form, setForm] = useState<FormData>(DEFAULT);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ risco: number; prob: number } | null>(null);
  const [tooltip, setTooltip] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm(prev => ({ ...prev, [e.target.name]: parseFloat(e.target.value) || 0 }));
    setResult(null);
  };

  const handleReset = () => { setForm(DEFAULT); setResult(null); };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(form),
    })
      .then(async r => {
        if (!r.ok) throw new Error('API error');
        const d = await r.json();
        setResult({ risco: d.risco_defasagem, prob: d.probabilidade });
      })
      .catch(() => {
        // Fallback client-side quando a API está offline
        let prob = 0.1;
        if (form.INDE < 5.5) prob += 0.5;
        else if (form.INDE < 6.5) prob += 0.3;
        if (form.IDA < 5.0) prob += 0.2;
        if (form.IEG < 5.0) prob += 0.1;
        if (form.IAN < 5.0) prob += 0.3;
        prob = Math.min(0.98, prob);
        setResult({ risco: prob > 0.5 ? 1 : 0, prob });
      })
      .finally(() => setLoading(false));
  };

  const probPct = result ? (result.prob * 100).toFixed(1) : null;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* ── Formulário ── */}
        <div className="lg:col-span-3 bg-white rounded-2xl border border-slate-100 shadow-sm overflow-hidden">
          <div className="bg-gradient-to-r from-indigo-600 to-indigo-700 px-6 py-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="bg-white/10 p-2 rounded-xl">
                  <BrainCircuit className="w-5 h-5 text-indigo-200" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white">Indicadores do Aluno</h2>
                  <p className="text-indigo-200 text-xs mt-0.5">Preencha todos os campos abaixo</p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="flex items-center gap-1.5 text-xs text-indigo-200 hover:text-white transition-colors bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded-lg"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Resetar
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="p-6 space-y-5">
            {/* Campos em grid */}
            <div className="grid grid-cols-2 gap-x-5 gap-y-4">
              {FIELDS.map(f => (
                <div key={f.name} className={f.highlight ? 'col-span-2' : ''}>
                  <div className="flex items-center gap-1 mb-1.5">
                    <label className={`text-xs font-bold ${f.highlight ? 'text-indigo-600' : 'text-slate-600'}`}>
                      {f.label}
                    </label>
                    <button
                      type="button"
                      onMouseEnter={() => setTooltip(f.name)}
                      onMouseLeave={() => setTooltip(null)}
                      className="text-slate-300 hover:text-slate-500 transition-colors relative"
                    >
                      <Info className="w-3 h-3" />
                      {tooltip === f.name && (
                        <span className="absolute left-4 -top-1 z-10 bg-slate-800 text-white text-[11px] px-2 py-1 rounded-lg shadow-lg whitespace-nowrap">
                          {f.desc}
                        </span>
                      )}
                    </button>
                  </div>
                  <div className="relative">
                    <input
                      type="number"
                      name={f.name}
                      value={form[f.name]}
                      onChange={handleChange}
                      step={f.step}
                      min={f.min}
                      max={f.max}
                      className={`w-full px-3 py-2.5 rounded-xl border text-sm font-semibold transition-all focus:outline-none focus:ring-2 focus:border-transparent ${f.highlight
                        ? 'border-indigo-200 bg-indigo-50/40 text-indigo-700 focus:ring-indigo-400'
                        : 'border-slate-200 bg-white text-slate-700 focus:ring-indigo-400'
                        }`}
                    />
                    {/* Mini barra de progresso dentro do campo */}
                    <div className="absolute bottom-0 left-3 right-3 h-0.5 bg-slate-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-300 rounded-full ${f.highlight ? 'bg-indigo-400' : 'bg-slate-300'}`}
                        style={{ width: `${Math.min((form[f.name] / f.max) * 100, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-60 text-white font-bold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm hover:shadow-md"
            >
              {loading
                ? <><Loader2 className="w-5 h-5 animate-spin" /><span>Consultando Random Forest…</span></>
                : <><BrainCircuit className="w-5 h-5" /><span>Prever Risco de Defasagem</span></>
              }
            </button>
          </form>
        </div>

        {/* ── Resultado ── */}
        <div className="lg:col-span-2 flex flex-col gap-5">

          {/* Card de resultado */}
          <div className={`flex-1 rounded-2xl border shadow-sm flex flex-col items-center justify-center p-6 min-h-64 transition-all duration-500 ${!result
            ? 'bg-white border-slate-100'
            : result.risco === 1
              ? 'bg-red-50 border-red-100'
              : 'bg-emerald-50 border-emerald-100'
            }`}>
            {!result && !loading && (
              <div className="text-center text-slate-300 space-y-3">
                <BrainCircuit className="w-14 h-14 mx-auto opacity-40" />
                <p className="text-sm font-medium">Preencha os indicadores<br />e clique em prever</p>
              </div>
            )}

            {loading && (
              <div className="text-center text-indigo-400 space-y-3">
                <Loader2 className="w-14 h-14 mx-auto animate-spin" />
                <p className="text-sm font-semibold animate-pulse">Processando…</p>
              </div>
            )}

            {result && !loading && (
              <div className="text-center w-full space-y-4">
                {result.risco === 1 ? (
                  <AlertTriangle className="w-14 h-14 mx-auto text-red-500" />
                ) : (
                  <CheckCircle2 className="w-14 h-14 mx-auto text-emerald-500" />
                )}

                <div>
                  <h3 className={`text-lg font-black ${result.risco === 1 ? 'text-red-700' : 'text-emerald-700'}`}>
                    {result.risco === 1 ? 'Alto Risco de Defasagem' : 'Baixo Risco de Defasagem'}
                  </h3>
                  <p className={`text-xs mt-1 ${result.risco === 1 ? 'text-red-500' : 'text-emerald-500'}`}>
                    {result.risco === 1
                      ? 'Padrões preocupantes identificados'
                      : 'Bom ritmo de desenvolvimento'}
                  </p>
                </div>

                {/* Gauge de probabilidade */}
                <div className="bg-white rounded-xl p-4 shadow-sm border border-white">
                  <p className="text-xs text-slate-400 font-semibold mb-2">Probabilidade de Defasagem</p>
                  <p className={`text-4xl font-black ${result.risco === 1 ? 'text-red-600' : 'text-emerald-600'}`}>
                    {probPct}%
                  </p>
                  <div className="mt-3 h-2.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${result.risco === 1 ? 'bg-red-500' : 'bg-emerald-500'}`}
                      style={{ width: `${result.prob * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Glossário dos indicadores */}
          <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-5">
            <p className="text-xs font-bold text-slate-500 uppercase tracking-wide mb-3">Glossário</p>
            <div className="space-y-1.5">
              {[
                ['INDE', 'Índice de Desenvolvimento Educacional'],
                ['IAA', 'Índice de Autoavaliação'],
                ['IEG', 'Índice de Engajamento'],
                ['IPS', 'Índice Psicossocial'],
                ['IDA', 'Índice de Aprendizagem'],
                ['IPV', 'Índice do Ponto de Virada'],
                ['IAN', 'Índice de Adequação ao Nível'],
              ].map(([abbr, full]) => (
                <div key={abbr} className="flex items-start gap-2">
                  <span className="text-[11px] font-black text-indigo-600 w-8 shrink-0 mt-0.5">{abbr}</span>
                  <span className="text-[11px] text-slate-500">{full}</span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}