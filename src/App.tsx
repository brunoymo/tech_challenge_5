import React, { useState, useEffect } from 'react';
import Analytics from './components/Analytics';
import Predictor from './components/Predictor';
import ModelMetrics from './components/ModelMetrics';
import { LayoutDashboard, BrainCircuit, GraduationCap, BarChart2, ChevronRight } from 'lucide-react';

type Tab = 'analytics' | 'predictor' | 'metrics';

const TABS: { id: Tab; label: string; sub: string; icon: React.ReactNode }[] = [
  {
    id: 'analytics',
    label: 'Visão Geral',
    sub: 'Indicadores e evolução dos alunos',
    icon: <LayoutDashboard className="w-5 h-5" />,
  },
  {
    id: 'predictor',
    label: 'Modelo Preditivo',
    sub: 'Simulador de risco individual',
    icon: <BrainCircuit className="w-5 h-5" />,
  },
  {
    id: 'metrics',
    label: 'Métricas do Modelo',
    sub: 'Avaliação e explicabilidade',
    icon: <BarChart2 className="w-5 h-5" />,
  },
];

const HEADERS: Record<Tab, { title: string; sub: string }> = {
  analytics: {
    title: 'Dashboard Educacional',
    sub: 'Acompanhe a evolução dos indicadores e a distribuição das classificações — dados reais de 2022 a 2024.',
  },
  predictor: {
    title: 'Simulador de Risco',
    sub: 'Insira os indicadores de um aluno e consulte o modelo Random Forest para estimar o risco de defasagem.',
  },
  metrics: {
    title: 'Desempenho do Modelo',
    sub: 'Métricas de avaliação, curva ROC, matriz de confusão e importância das features do modelo treinado.',
  },
};

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('analytics');
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const API_URL = import.meta.env.VITE_API_URL || 'https://passos-magicos-api-chcj.onrender.com';
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((d) => setApiStatus(d.model_loaded ? 'online' : 'offline'))
      .catch(() => setApiStatus('offline'));
  }, []);

  const header = HEADERS[activeTab];

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col md:flex-row font-sans text-slate-900">
      {/* ── Sidebar ── */}
      <aside className="w-full md:w-64 bg-white border-r border-slate-200 flex flex-col shrink-0">
        {/* Logo */}
        <div className="p-5 border-b border-slate-100">
          <div className="flex items-center space-x-3">
            <div className="bg-gradient-to-br from-indigo-500 to-indigo-700 p-2.5 rounded-xl shadow-sm">
              <GraduationCap className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-base leading-tight text-slate-800">Passos Mágicos</h1>
              <p className="text-[11px] font-semibold text-indigo-500 uppercase tracking-wider">MLOps · Datathon</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="p-3 space-y-1 flex-1">
          {TABS.map((t) => {
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`w-full flex items-center justify-between px-3 py-3 rounded-xl transition-all group ${active
                    ? 'bg-indigo-50 text-indigo-700'
                    : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800'
                  }`}
              >
                <div className="flex items-center space-x-3">
                  <span className={active ? 'text-indigo-600' : 'text-slate-400 group-hover:text-slate-600'}>
                    {t.icon}
                  </span>
                  <div className="text-left">
                    <p className={`text-sm font-semibold leading-tight ${active ? 'text-indigo-700' : ''}`}>{t.label}</p>
                    <p className={`text-[11px] leading-tight mt-0.5 ${active ? 'text-indigo-400' : 'text-slate-400'}`}>{t.sub}</p>
                  </div>
                </div>
                {active && <ChevronRight className="w-4 h-4 text-indigo-400 shrink-0" />}
              </button>
            );
          })}
        </nav>

        {/* API Status */}
        <div className="p-4 border-t border-slate-100">
          <div className={`rounded-xl px-4 py-3 border flex items-center justify-between ${apiStatus === 'online' ? 'bg-emerald-50 border-emerald-100' :
              apiStatus === 'offline' ? 'bg-red-50 border-red-100' :
                'bg-amber-50 border-amber-100'
            }`}>
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Status da API</p>
              <p className={`text-sm font-bold mt-0.5 ${apiStatus === 'online' ? 'text-emerald-700' :
                  apiStatus === 'offline' ? 'text-red-700' : 'text-amber-700'
                }`}>
                {apiStatus === 'online' ? 'Modelo Online' : apiStatus === 'offline' ? 'API Offline' : 'Verificando…'}
              </p>
            </div>
            <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${apiStatus === 'online' ? 'bg-emerald-500 animate-pulse' :
                apiStatus === 'offline' ? 'bg-red-500' : 'bg-amber-400 animate-pulse'
              }`} />
          </div>
          <p className="text-[10px] text-slate-400 text-center mt-3">
            FIAP · Machine Learning · Fase 5
          </p>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 overflow-y-auto">
        {/* Page header */}
        <div className="bg-white border-b border-slate-200 px-6 md:px-8 py-5">
          <h2 className="text-xl font-bold text-slate-800">{header.title}</h2>
          <p className="text-sm text-slate-500 mt-1 max-w-2xl">{header.sub}</p>
        </div>

        <div className="p-6 md:p-8">
          {activeTab === 'analytics' && <Analytics />}
          {activeTab === 'predictor' && <Predictor />}
          {activeTab === 'metrics' && <ModelMetrics />}
        </div>
      </main>
    </div>
  );
}
