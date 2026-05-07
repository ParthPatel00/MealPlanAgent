"use client";

import { useState, useRef, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

const SAMPLES = [
  "Plan me 3 high-protein dinners, no peanuts, under 30 min each",
  "5 vegetarian meals for the week, avoid gluten and dairy",
  "Quick keto lunches for 4 days, no shellfish or soy",
];

const MODELS = [
  { id: "ollama-llama3b", name: "Llama 3B" },
  { id: "ollama-granite2b", name: "Granite 2B" },
  { id: "groq-llama", name: "Llama 70B" },
];

const DV: Record<string, { val: number; unit: string }> = {
  "Calories (kcal)": { val: 2000, unit: "kcal" },
  "Total Fat (g)": { val: 78, unit: "g" },
  "Sugar (g)": { val: 50, unit: "g" },
  "Sodium (mg)": { val: 2300, unit: "mg" },
  "Protein (g)": { val: 50, unit: "g" },
  "Saturated Fat (g)": { val: 20, unit: "g" },
  "Carbohydrates (g)": { val: 275, unit: "g" },
};

/* eslint-disable @typescript-eslint/no-explicit-any */
type R = any;

type Message = { role: "user"; text: string } | { role: "assistant"; data: R };

export default function Page() {
  const [input, setInput] = useState("");
  const [model, setModel] = useState("ollama-llama3b");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [rec, setRec] = useState(false);
  const mr = useRef<MediaRecorder | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text }]);
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: text, model, user_id: "default_user" }),
      });
      const d = await r.json();
      if (d.error) {
        setMessages((m) => [...m, { role: "assistant", data: { error: d.error } }]);
      } else {
        setMessages((m) => [...m, { role: "assistant", data: d }]);
      }
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", data: { error: `Connection failed: ${e}` } }]);
    }
    setLoading(false);
  };

  const voice = async () => {
    if (rec) { mr.current?.stop(); setRec(false); return; }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      const m = new MediaRecorder(s);
      mr.current = m;
      const ch: Blob[] = [];
      m.ondataavailable = (e) => ch.push(e.data);
      m.onstop = async () => {
        const fd = new FormData();
        fd.append("file", new Blob(ch, { type: "audio/webm" }), "a.webm");
        try {
          const r = await (await fetch(`${API}/api/transcribe`, { method: "POST", body: fd })).json();
          if (r.text) setInput(r.text);
        } catch {}
        s.getTracks().forEach((t) => t.stop());
      };
      m.start();
      setRec(true);
    } catch { alert("Mic access denied"); }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="shrink-0 h-12 flex items-center px-5 border-b border-black/5">
        <span className="text-sm font-semibold text-gray-800">MealPlanAgent</span>
      </header>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto hide-scrollbar">
        <div className="max-w-3xl mx-auto px-5 py-6">
          {messages.length === 0 && !loading && (
            <div className="pt-[20vh] text-center">
              <h2 className="text-2xl font-semibold text-gray-800 mb-2">What should we cook this week?</h2>
              <p className="text-sm text-gray-400 mb-8">Describe your meals and I&apos;ll plan recipes, grocery lists, nutrition, and a calendar.</p>
              <div className="flex flex-wrap justify-center gap-2">
                {SAMPLES.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => { setInput(s); }}
                    className="text-sm px-4 py-2.5 bg-white border border-gray-200 text-gray-600 rounded-xl hover:border-gray-300 hover:bg-gray-50 transition"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className="mb-6">
              {msg.role === "user" ? (
                <UserBubble text={msg.text} />
              ) : (
                <AssistantResponse data={msg.data} />
              )}
            </div>
          ))}

          {loading && (
            <div className="mb-6">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <div className="flex gap-1">
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
                <span>Planning your meals...</span>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar - fixed bottom */}
      <div className="shrink-0 border-t border-black/5 bg-[#f7f5f2] px-5 py-3">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="flex items-end gap-2 px-4 py-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder="Describe what you want to eat..."
                rows={1}
                className="flex-1 text-sm bg-transparent resize-none focus:outline-none placeholder:text-gray-400 max-h-32"
                style={{ minHeight: "20px" }}
                onInput={(e) => {
                  const t = e.currentTarget;
                  t.style.height = "20px";
                  t.style.height = Math.min(t.scrollHeight, 128) + "px";
                }}
              />
              <button
                onClick={voice}
                className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center transition ${
                  rec ? "bg-red-100 text-red-600" : "text-gray-400 hover:bg-gray-100 hover:text-gray-600"
                }`}
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                </svg>
              </button>
              <button
                onClick={send}
                disabled={loading || !input.trim()}
                className="shrink-0 w-8 h-8 rounded-full bg-green-600 text-white flex items-center justify-center disabled:opacity-30 hover:bg-green-700 transition"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
            {/* Model selector row below input */}
            <div className="flex items-center gap-2 px-4 py-2 border-t border-gray-100">
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setModel(m.id)}
                  className={`px-2.5 py-1 text-[11px] rounded-md transition ${
                    model === m.id ? "bg-gray-100 text-gray-900 font-medium" : "text-gray-400 hover:text-gray-600"
                  }`}
                >
                  {m.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="bg-green-600 text-white px-4 py-2.5 rounded-2xl rounded-br-md max-w-md text-sm">
        {text}
      </div>
    </div>
  );
}

function AssistantResponse({ data }: { data: R }) {
  const [tab, setTab] = useState(0);

  if (data.error) {
    return <div className="text-sm text-red-600">{data.error}</div>;
  }

  const tabs = ["Meals", "Grocery", "Nutrition", "Schedule", "Trace"];
  const pc = data.parsed_constraints;

  return (
    <div>
      {/* Parsed understanding */}
      <div className="flex flex-wrap gap-1.5 mb-3 text-[11px]">
        {pc && (
          <>
            <span className="bg-white border border-gray-200 text-gray-600 px-2 py-0.5 rounded-md">{pc.num_meals} meals</span>
            <span className="bg-white border border-gray-200 text-gray-600 px-2 py-0.5 rounded-md">{pc.max_minutes} min max</span>
            {pc.tags?.map((t: string) => <span key={t} className="bg-green-50 border border-green-200 text-green-700 px-2 py-0.5 rounded-md">{t}</span>)}
            {pc.allergens?.map((a: string) => <span key={a} className="bg-red-50 border border-red-200 text-red-600 px-2 py-0.5 rounded-md">avoid {a}</span>)}
            {pc.dietary_notes && <span className="bg-gray-50 border border-gray-200 text-gray-500 px-2 py-0.5 rounded-md">{pc.dietary_notes}</span>}
          </>
        )}
      </div>

      {data.critic && !data.critic.valid && (
        <p className="text-xs text-amber-600 mb-3">{data.critic.issues.join(". ")}</p>
      )}

      {/* Tab navigation */}
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="flex border-b border-gray-100 px-1 pt-1">
          {tabs.map((t, i) => (
            <button
              key={t}
              onClick={() => setTab(i)}
              className={`px-3.5 py-2 text-xs font-medium rounded-t-lg transition ${
                tab === i ? "bg-gray-50 text-gray-900" : "text-gray-400 hover:text-gray-600"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="p-5">
          {tab === 0 && <MealsPanel data={data} />}
          {tab === 1 && <GroceryPanel data={data} />}
          {tab === 2 && <NutritionPanel data={data} />}
          {tab === 3 && <SchedulePanel data={data} />}
          {tab === 4 && <TracePanel data={data} />}
        </div>
      </div>

      <p className="text-[10px] text-gray-400 mt-2">{data.elapsed_seconds?.toFixed(1)}s via {data.planner_trace?.model || "unknown"}</p>
    </div>
  );
}

// ===== MEALS =====
function MealsPanel({ data }: { data: R }) {
  const [open, setOpen] = useState(0);
  const recipes: R[] = data.recipes || [];
  if (!recipes.length) return <p className="text-sm text-gray-400">No recipes found. Try a broader request.</p>;

  return (
    <div className="space-y-0 -mx-5 -my-5">
      {recipes.map((r: R, i: number) => {
        const day = r._day || ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][i % 7];
        const al = data.allergy_reports?.find((a: R) => a.recipe_name === r.name);
        const isOpen = open === i;

        return (
          <div key={i} className={i > 0 ? "border-t border-gray-100" : ""}>
            <button onClick={() => setOpen(isOpen ? -1 : i)} className="w-full text-left px-5 py-3.5 flex items-center gap-3 hover:bg-gray-50/50 transition">
              <span className="text-[10px] font-semibold text-gray-400 uppercase w-8">{day.slice(0, 3)}</span>
              <span className="text-sm text-gray-900 font-medium flex-1 truncate">{r.name}</span>
              <span className="text-[11px] text-gray-400 shrink-0">{r.minutes} min</span>
              {al && !al.safe && <span className="text-[10px] text-red-500 font-medium shrink-0">UNSAFE</span>}
              <svg className={`w-3.5 h-3.5 text-gray-300 shrink-0 transition ${isOpen ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {isOpen && (
              <div className="px-5 pb-5 pt-1">
                {r.description && <p className="text-xs text-gray-500 mb-4 leading-relaxed">{r.description}</p>}

                <div className="grid grid-cols-[1fr_180px] gap-6">
                  <div>
                    <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">Ingredients</p>
                    <p className="text-xs text-gray-600 leading-relaxed mb-4">{r.ingredients?.join(", ")}</p>

                    {r.steps?.length > 0 && (
                      <>
                        <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1.5">Method</p>
                        <div className="space-y-1.5 mb-4">
                          {r.steps.map((s: string, j: number) => (
                            <p key={j} className="text-xs text-gray-600 leading-relaxed">
                              <span className="text-gray-300 inline-block w-4">{j + 1}.</span> {s}
                            </p>
                          ))}
                        </div>
                      </>
                    )}

                    <a href={r.citation?.url} target="_blank" rel="noopener noreferrer" className="text-xs text-green-600 hover:underline">
                      View on Food.com &rarr;
                    </a>
                  </div>

                  {/* Nutrition sidebar */}
                  <div className="text-[11px]">
                    <p className="font-semibold text-gray-400 uppercase tracking-wider mb-1.5 text-[10px]">Nutrition</p>
                    {r._nutrition_abs && Object.entries(r._nutrition_abs as Record<string, number>).map(([k, v]) => {
                      const ref = DV[k];
                      const pct = ref ? Math.round((v / ref.val) * 100) : 0;
                      return (
                        <div key={k} className="flex justify-between py-0.5 border-b border-gray-50 last:border-0">
                          <span className="text-gray-500">{k.replace(/ \(.*\)/, "")}</span>
                          <span className="tabular-nums text-gray-800">{v}{ref?.unit} <span className="text-gray-400">{pct}%</span></span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ===== GROCERY =====
function GroceryPanel({ data }: { data: R }) {
  const gl = data.grocery_list || {};
  const budget = data.budget_estimate;
  if (!Object.keys(gl).length) return <p className="text-sm text-gray-400">No grocery list.</p>;
  const perItem = budget?.per_item || [];
  const perCat = budget?.per_category || {};

  return (
    <div>
      {budget?.total_estimated_cost && (
        <p className="text-lg font-semibold mb-4 tabular-nums">${budget.total_estimated_cost.toFixed(2)} <span className="text-sm font-normal text-gray-400">estimated</span></p>
      )}
      <div className="grid grid-cols-2 gap-x-8 gap-y-4">
        {Object.entries(gl as Record<string, string[]>).map(([cat, items]) => (
          <div key={cat}>
            <div className="flex justify-between mb-1">
              <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider">{cat}</span>
              {perCat[cat] && <span className="text-[10px] text-gray-400">${(perCat[cat] as number).toFixed(2)}</span>}
            </div>
            {items.map((item: string, j: number) => {
              const p = perItem.find((x: R) => x.item?.toLowerCase() === item.toLowerCase());
              return (
                <div key={j} className="flex justify-between py-0.5 text-xs">
                  <span className="text-gray-700">{item}</span>
                  {p && <span className="text-gray-400 tabular-nums">${p.estimated_price.toFixed(2)}</span>}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

// ===== NUTRITION =====
function NutritionPanel({ data }: { data: R }) {
  const recipes: R[] = data.recipes || [];
  if (!recipes.length) return <p className="text-sm text-gray-400">No data.</p>;
  const nutrients = Object.keys(DV);

  return (
    <div className="overflow-x-auto -mx-5 px-5">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-1.5 font-medium text-gray-500 pr-3">Meal</th>
            {nutrients.map((n) => <th key={n} className="text-right py-1.5 font-medium text-gray-500 px-2 whitespace-nowrap">{n.replace(/ \(.*\)/, "")}</th>)}
          </tr>
        </thead>
        <tbody>
          {recipes.map((r: R, i: number) => (
            <tr key={i} className="border-b border-gray-50">
              <td className="py-2 pr-3 text-xs text-gray-900 font-medium truncate max-w-[150px]">{r.name}</td>
              {nutrients.map((n) => {
                const v = r._nutrition_abs?.[n] || 0;
                const ref = DV[n];
                const pct = Math.round((v / ref.val) * 100);
                return (
                  <td key={n} className="text-right py-2 px-2 tabular-nums whitespace-nowrap">
                    {v}{ref.unit} <span className={pct > 80 ? "text-amber-500" : "text-gray-300"}>{pct}%</span>
                  </td>
                );
              })}
            </tr>
          ))}
          <tr className="font-medium border-t border-gray-200">
            <td className="py-2 text-xs text-gray-900">Weekly total</td>
            {nutrients.map((n) => {
              const total = recipes.reduce((s: number, r: R) => s + (r._nutrition_abs?.[n] || 0), 0);
              const ref = DV[n];
              const pctW = Math.round((total / (ref.val * 7)) * 100);
              return (
                <td key={n} className="text-right py-2 px-2 tabular-nums whitespace-nowrap">
                  {total.toFixed(0)}{ref.unit} <span className={pctW > 100 ? "text-red-500" : "text-gray-400"}>{pctW}%w</span>
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
      <p className="text-[9px] text-gray-400 mt-2">% = daily reference value. %w = % of weekly (7-day) reference. FDA 2,000 cal.</p>
    </div>
  );
}

// ===== SCHEDULE =====
function SchedulePanel({ data }: { data: R }) {
  const [hour, setHour] = useState(18);
  const [min, setMin] = useState(0);
  const [dl, setDl] = useState(false);
  const blocks = data.cooking_blocks || [];
  if (!blocks.length) return <p className="text-sm text-gray-400">No schedule.</p>;

  const download = async () => {
    setDl(true);
    try {
      const r = await fetch(`${API}/api/generate-ics`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cooking_blocks: blocks, cook_hour: hour, cook_minute: min }),
      });
      const d = await r.json();
      if (d.ics_base64) {
        const b = Uint8Array.from(atob(d.ics_base64), (c) => c.charCodeAt(0));
        const url = URL.createObjectURL(new Blob([b], { type: "text/calendar" }));
        Object.assign(document.createElement("a"), { href: url, download: "meal_plan.ics" }).click();
        URL.revokeObjectURL(url);
      }
    } catch (e) { alert(`Error: ${e}`); }
    setDl(false);
  };

  const hStr = `${hour % 12 || 12}:${min.toString().padStart(2, "0")} ${hour >= 12 ? "PM" : "AM"}`;

  return (
    <div>
      <div className="flex items-center gap-2 mb-4 text-sm">
        <span className="text-gray-500">Cook at</span>
        <select value={hour} onChange={(e) => setHour(+e.target.value)} className="text-sm border border-gray-200 rounded-md px-2 py-1 bg-white">
          {Array.from({ length: 24 }, (_, i) => <option key={i} value={i}>{i === 0 ? "12 AM" : i < 12 ? `${i} AM` : i === 12 ? "12 PM" : `${i-12} PM`}</option>)}
        </select>
        <select value={min} onChange={(e) => setMin(+e.target.value)} className="text-sm border border-gray-200 rounded-md px-2 py-1 bg-white">
          {[0, 15, 30, 45].map((m) => <option key={m} value={m}>:{m.toString().padStart(2, "0")}</option>)}
        </select>
        <button onClick={download} disabled={dl} className="ml-2 px-3 py-1 bg-green-600 text-white text-xs rounded-md font-medium hover:bg-green-700 disabled:opacity-40 transition">
          {dl ? "..." : "Download .ics"}
        </button>
      </div>
      <div className="space-y-0.5">
        {blocks.map((b: R, i: number) => (
          <div key={i} className="flex items-center py-1.5 text-xs">
            <span className="w-10 font-medium text-gray-400 uppercase">{b.day?.slice(0, 3)}</span>
            <span className="w-16 text-gray-400">{hStr}</span>
            <span className="flex-1 text-gray-800">{b.meal_name}</span>
            <span className="text-gray-400">{b.duration_minutes}m</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ===== TRACE =====
function TracePanel({ data }: { data: R }) {
  return (
    <div className="space-y-4 text-[11px]">
      <div className="text-gray-400 flex gap-3">
        <code>{data.session_id}</code>
        <span>{data.retries} retries</span>
        <span>{data.elapsed_seconds?.toFixed(1)}s</span>
      </div>

      <div>
        <p className="font-medium text-gray-700 mb-1">0. Parse</p>
        <p className="text-gray-500 italic">&quot;{data.original_input}&quot;</p>
        <Fold label="Raw output">{data.nl_parse_raw}</Fold>
      </div>

      {data.planner_trace && (
        <div>
          <p className="font-medium text-gray-700 mb-1">1. Planner <span className="font-normal text-gray-400">{data.planner_trace.model}, {data.planner_trace.latency_ms?.toFixed(0)}ms</span></p>
          <Fold label="System prompt">{data.planner_trace.system_prompt}</Fold>
          <Fold label="User prompt">{data.planner_trace.user_prompt}</Fold>
          <Fold label="Response" open>{data.planner_trace.raw_response}</Fold>
        </div>
      )}

      <div>
        <p className="font-medium text-gray-700 mb-1">2. Executor <span className="font-normal text-gray-400">{data.tool_calls?.length} calls</span></p>
        <div className="flex gap-1 mb-1.5 flex-wrap">
          {Array.from((data.tool_calls || []).reduce((m: Map<string, number>, t: R) => m.set(t.tool, (m.get(t.tool) || 0) + 1), new Map()) as Map<string, number>).map(([t, c]: [string, number]) => (
            <span key={t} className="bg-violet-50 text-violet-600 px-1.5 py-0.5 rounded">{t} x{c}</span>
          ))}
        </div>
        {(data.tool_calls || []).map((tc: R, i: number) => <Fold key={i} label={`#${i+1} ${tc.tool}`}>{JSON.stringify(tc, null, 2)}</Fold>)}
      </div>

      <div>
        <p className="font-medium text-gray-700 mb-1">3. Critic</p>
        {data.critic && <p className={data.critic.valid ? "text-green-600" : "text-red-600"}>{data.critic.valid ? "Approved" : data.critic.issues?.join(", ")}</p>}
      </div>
    </div>
  );
}

function Fold({ label, children, open = false }: { label: string; children: string; open?: boolean }) {
  return (
    <details className="mb-1" open={open}>
      <summary className="text-gray-400 cursor-pointer hover:text-gray-600 text-[11px]">{label}</summary>
      <pre className="mt-1 bg-gray-900 text-gray-300 p-2.5 rounded-md overflow-auto max-h-40 text-[10px] leading-relaxed">{children}</pre>
    </details>
  );
}
