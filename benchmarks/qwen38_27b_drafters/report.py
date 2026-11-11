"""Render report/index.html (self-contained except Plotly from cdnjs) from report/data.json."""

import argparse
import json
from pathlib import Path

DRAFTERS = {
    "mtp": dict(label="Qwen MTP (built-in)", train="unknown", kind="MTP", author="Qwen"),
    "dflash2": dict(label="DFlash2 (z-lab)", train="unknown", kind="block diffusion, block 16", author="z-lab"),
    "dspark_radixark": dict(label="DSpark (RadixArk)", train="unknown; YaRN×32 over 8192", kind="DSpark", author="RadixArk"),
    "dspark_redhat": dict(label="DSpark (Red Hat)", train="8192 (last stage, model card)", kind="DSpark, sliding window 2048", author="Red Hat"),
    "eagle4": dict(label="EAGLE-4", train="unknown", kind="EAGLE-4, chain", author="EAGLE-Corp"),
}

HTML = r"""<!doctype html>
<html lang="ru" data-theme="light"><head>
<script>try{const t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t;}catch(e){}</script><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drafter Acceptance Report</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#8a8984;--grid:#e8e7e3;--card:#ffffff;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--ar:#8a8984;--dneg:#c8402f;--dmid:#ecebe7;--dpos:#2a78d6;--tacc:rgba(27,175,122,.20);--trej:rgba(235,104,52,.38);--teos:#c8325a;--tcyc:rgba(42,120,214,.10)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#8f8e88;--grid:#2e2e2c;--card:#222221;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--ar:#8f8e88;--dneg:#e0624f;--dmid:#2c2c2a;--dpos:#4f95ea;--tacc:rgba(25,158,112,.28);--trej:rgba(217,89,38,.50);--teos:#ff6f91;--tcyc:rgba(57,135,229,.16)}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#8f8e88;--grid:#2e2e2c;--card:#222221;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--ar:#8f8e88;--dneg:#e0624f;--dmid:#2c2c2a;--dpos:#4f95ea;--tacc:rgba(25,158,112,.28);--trej:rgba(217,89,38,.50);--teos:#ff6f91;--tcyc:rgba(57,135,229,.16)}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 16px 80px;position:relative}
#theme{position:absolute;top:24px;right:16px;font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid var(--grid);background:var(--card);color:var(--fg2);cursor:pointer}
h1{font-size:26px;margin:8px 0 4px}h2{font-size:20px;margin:40px 0 6px}h3{font-size:16px;margin:24px 0 4px}
p,li{color:var(--fg2);max-width:820px}.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px;margin:12px 0}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:6px 0 4px;color:var(--fg2);font-size:14px}
select{font:inherit;padding:3px 6px;border-radius:6px;border:1px solid var(--grid);background:var(--bg);color:var(--fg)}
table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--fg2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}td.heat{text-align:center;font-variant-numeric:tabular-nums}td.heat .hn{font-size:11px;opacity:.75}
.ctitle{font-weight:600;font-size:14px;margin:2px 0 0}.plot{width:100%;height:420px}.small{height:340px}.note{font-size:13px;color:var(--muted)}
a.btn{display:inline-block;padding:8px 14px;border-radius:8px;background:var(--fg);color:var(--bg);text-decoration:none;font-weight:600}
pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--grid);border-radius:8px;padding:8px;font-size:12px;max-height:180px;overflow:auto}
</style></head><body><main>
<button id="theme" type="button"></button>
<h1>Acceptance драфтеров Qwen3.8-27B на длинных контекстах</h1>
<p>__INTRO__</p>
<div class="card"><table id="drafters"></table></div>

<div class="card"><div class="ctitle">Два режима текста</div><p style="margin:6px 0 0"><b>Свой текст</b> — каждый драфтер генерирует сам. Жадная верификация в теории даёт тот же текст, что и AR, но из-за численных расхождений (bf16, разные батчи и ядра) траектории расходятся: с AR побайтово совпадает лишь ~11% ответов, так что кривые разных драфтеров построены на разных текстах. <b>Общий текст (teacher forcing)</b> — все драфтеры проверяются на одной и той же траектории AR, побайтово. __FORCED_INTRO__</p></div>

<h2>1. Влияет ли длина входа</h2>
<p>Сравниваются документы разной длины при <b>одинаковой позиции в ответе</b>: acceptance по первым 2K токенам ответа каждого трейса, только длинные сеты. Окно фиксировано, потому что acceptance сильно растёт по ходу ответа (раздел 2): среднее по всему ответу или «до цикла» зависело бы от длины ответа, а не от входа. Ответы короче 2K (свой текст) входят целиком.</p>
<div class="card"><div class="controls">
<label>Текст <select id="in-mode"><option value="forced">общий (teacher forcing)</option><option value="natural">свой (естественная остановка)</option></select></label>
<label>Сет <select id="in-ds"></select></label></div>
<div class="ctitle">Acceptance на первых 2K токенах ответа от длины входа</div><div id="in1" class="plot"></div>
<div class="note" id="in1-note"></div></div>
<div class="card"><div class="ctitle">Наклон: изменение acceptance на каждое удвоение длины входа</div><table id="in-t"></table>
<div class="note">Внутри сетов (fixed effects по сету), 95% bootstrap по трейсам внутри сетов. Вместе с длиной меняется и сам документ, так что это не чистое вмешательство.</div></div>

<h2>2. Как acceptance меняется по ходу ответа</h2>
<p>Только общий текст: на своём тексте ответы заканчиваются в разных местах, и в дальних окнах остались бы только длинные ответы. Здесь состав трейсов в каждом окне одинаков для всех драфтеров.</p>
<div class="card"><div class="controls">
<label>Трейсы <select id="f-group"></select></label>
<label>Фаза <select id="f-phase"><option value="all">всё</option><option value="think">рассуждение (до &lt;/think&gt;)</option><option value="answer">ответ (после &lt;/think&gt;)</option></select></label>
<label>Раунды <select id="f-kind"><option value="all">все</option><option value="nc">без циклов</option></select></label></div>
<div class="ctitle">Acceptance по позиции в ответе</div><div id="f1" class="plot"></div>
<div class="note">Для каждого трейса — acceptance в окне (полуоктавы, ≥16 раундов), точка — среднее по трейсам, полоса — 95% bootstrap. Все драфтеры видят один и тот же текст, поэтому кривые сравнимы напрямую; состав трейсов в дальних окнах одинаков для всех драфтеров. Фаза определяется по первому &lt;/think&gt; в эталоне: после него target с запрещённым EOS дописывает и повторяет готовый ответ, который драфтер угадывает почти всегда, — поэтому кривая «всё» растёт по мере того, как трейсы выходят из рассуждения.</div></div>
<h2>3. Внутри одного трейса</h2>
<p>Чтобы отделить эффект длины от смены состава окон, каждый трейс сравнивается сам с собой: acceptance в окне минус acceptance того же трейса на его первых 2K сгенерированных токенов. Окна — полуоктавы; ячейка трейса учитывается, если в ней ≥16 раундов. «Без циклов» — только раунды вне обнаруженных циклов (детектор как в разделе 5), база та же. «Фаза» оставляет раунды только до или только после первого &lt;/think&gt; трейса; база — первые 2K токенов этой фазы (для ответа — от &lt;/think&gt;). Текст общий для всех драфтеров (teacher forcing).</p>
<div class="card"><div class="controls">
<select id="w-panel" hidden></select>
<label>Ось <select id="w-axis"><option value="gen">позиция в ответе</option><option value="ctx">полный контекст</option></select></label>
<label>Фаза <select id="w-phase"><option value="all">всё</option><option value="think">рассуждение (до &lt;/think&gt;)</option><option value="answer">ответ (после &lt;/think&gt;)</option></select></label>
<label>Раунды <select id="w-kind"><option value="d">все</option><option value="d_nc">без циклов</option></select></label>
<label>Трейсы <select id="w-group"></select></label></div>
<div class="ctitle">Парное изменение acceptance относительно начала трейса</div><div id="w1" class="plot"></div>
<div class="note">Точка — среднее по трейсам изменения в окне (п.п.), полоса — 95% bootstrap по трейсам. Дальние окна есть только у длинных трейсов, их число — в подсказке.</div>
<div id="w-ent"><div class="controls"><label>Значение <select id="we-kind"><option value="d">изменение относительно начала трейса</option><option value="e">энтропия</option></select></label></div>
<div class="ctitle">Энтропия target: парное изменение относительно начала трейса</div><div id="w4" class="plot small"></div>
<div class="note">Те же трейсы, окна, фаза и раунды, что на графике выше; база — первые 2K токенов трейса (в фазе — от её начала). Энтропия распределения target по top-20 токенам, нат, на траектории AR до запрета EOS. Она зависит от текста, а не от драфтера, поэтому линия одна.</div></div></div>
<div class="card"><div class="controls"><label>Драфтер <select id="w-hm"></select></label></div>
<div class="ctitle">Изменение acceptance относительно начала трейса: длина входа × сгенерировано токенов</div><div style="overflow-x:auto"><table id="w3"></table></div>
<div class="note">Раунды — из селектора выше (фаза и трейсы здесь не применяются). Первая строка — acceptance на первых 512 токенах ответа (база). Остальные клетки — среднее по трейсам разности acceptance в клетке и базы того же трейса, п.п.; мелко — число трейсов. Трейс входит в клетку, если там ≥16 его раундов; клетки с &lt;3 трейсами пустые. В подсказке — 95% bootstrap и сам acceptance. Синий — выше базы, красный — ниже; насыщенность нормирована на 10-й и 90-й перцентили клеток таблицы.</div></div>

<h2>4. Естественная генерация: acceptance по полному контексту</h2>
<p>Свой текст, естественная остановка по EOS, thinking включён, greedy, K=7 цепочкой. По оси X — полная длина контекста перед раундом проверки (вход + уже сгенерированное); каждый раунд попадает в своё окно, длинный ответ даёт точки в нескольких окнах. Это то, что видит пользователь, но здесь смешаны длина входа, позиция в ответе и состав сетов — для выводов о длине см. раздел 1. Шкала логарифмическая; точка — среднее по запросам в окне, полоса — 95% bootstrap-интервал по запросам.</p>
<div class="card"><div class="controls">
<label>Сет <select id="c1-ds"></select></label>
<label>Метрика <select id="c1-metric"><option value="acc">acceptance (принято / предложено)</option><option value="tau">принятых токенов за раунд</option></select></label>
</div><div id="c1" class="plot"></div><div class="note" id="c1-note"></div></div>


<h2>5. Что происходит, когда EOS запрещён</h2>
<p>Свой текст. __NOEOS_INTRO__</p>
<div class="card"><div class="ctitle">Acceptance по позиции генерации (EOS запрещён)</div><div class="controls"><label>Сет <select id="c4-ds"></select></label></div><div id="c4" class="plot"></div></div>
<div class="card"><div class="ctitle">Доля траекторий, находящихся в цикле</div><div class="controls"><label>Разбивка <select id="c5-by"><option value="method">по методам (свои траектории)</option><option value="dataset">по сетам (AR)</option></select></label></div>
<div id="c5" class="plot small"></div><div class="note">Цикл: последние 2048 токенов периодичны с периодом ≤128 (≥99% совпадений со сдвигом) или последние 8192 — с периодом 129–2048. Детектор как в ветке issue11. «По методам» — каждый метод на своём тексте, поэтому кривые не совпадают с AR; «по сетам» — только траектории AR.</div></div>
<div class="card"><div class="ctitle" id="c7-title">Где target впервые хочет EOS</div><div id="c7" class="plot small"></div></div>
<div class="card"><div class="ctitle">Когда начинается цикл и какой у него период</div><div style="display:flex;flex-wrap:wrap;gap:8px"><div id="c10" class="plot small" style="flex:1 1 420px"></div><div id="c11" class="plot small" style="flex:1 1 420px"></div></div><div class="note">Слева: позиция первого обнаруженного цикла (точка — траектория; в подписи — траекторий с циклом / всего). Справа: медианный период цикла в траекториях AR, логарифмическая шкала.</div></div>
<div class="card"><div class="ctitle">Acceptance до и после «желания» EOS, вне и внутри циклов</div><div id="c8" class="plot small"></div></div>
<div class="card"><div class="ctitle">Acceptance в зависимости от энтропии target (top-20, нат)</div><div id="c9" class="plot small"></div><div class="note">Энтропия target в позиции раунда по сырым top-20 logprobs AR; для драфтеров берётся текст AR, поэтому учитываются только совпадающие траектории.</div></div>
<div class="card"><div class="ctitle">Как выглядят хвосты</div><div class="note">Ссылки открывают траекторию AR в обозревателе примеров на этом участке; там же лента по всему ответу и развёртка цикла.</div><table id="examples"></table></div>

<h2>6. Примеры</h2>
<p>Тексты всех запросов и траекторий всех методов — в отдельном интерактивном обозревателе: выбор сета и примера, лента по всему ответу (acceptance раундов, циклы, EOS top-1, расхождение с AR) с переходом кликом, развёртка цикла по периоду.</p>
<p><a class="btn" href="examples.html">Открыть примеры →</a></p>

<h2>7. Данные и протокол</h2><div class="card"><table id="counts"></table></div>
<p class="note">__PROTOCOL__</p>
</main>
<script>
const D = __DATA__;
const META = __META__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const ORDER = ["dflash2","dspark_radixark","dspark_redhat","eagle4","mtp"];
const COLOR = m => m==="ar" ? css("--ar") : css("--s"+(ORDER.indexOf(m)+1));
const LABEL = m => m==="ar" ? "AR (без драфтера)" : (META[m]||{label:m}).label;
const KT = [1000,2000,4000,8000,16000,32000,64000,128000,256000];
const fmtK = v => v>=1000 ? (v/1000)+"K" : String(v);
const fmtT = v => v>=1024 ? (+(v/1024).toFixed(1))+"K" : String(v); // token positions on power-of-two edges
function layout(extra){ return Object.assign({paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",
  font:{color:css("--fg"),size:13},margin:{l:60,r:20,t:36,b:56},hovermode:"x unified",
  hoverlabel:{bgcolor:css("--card"),bordercolor:css("--grid"),font:{color:css("--fg"),size:13}},
  legend:{orientation:"h",x:0,y:1.0,yanchor:"bottom"},xaxis:{gridcolor:css("--grid"),zeroline:false},
  yaxis:{gridcolor:css("--grid"),zeroline:false}}, extra||{}); }
const cfg={displaylogo:false,responsive:true};
const GT={tickvals:[0,8192,16384,24576,32768,40960,49152],ticktext:["0","8K","16K","24K","32K","40K","48K"]};
const DASH=["solid","dot","dash","dashdot","longdash","longdashdot"];
function yr(vals,cap){ const v=vals.filter(x=>x!=null&&isFinite(x)); if(!v.length) return [0,cap||1];
  const lo=Math.max(0,Math.floor((Math.min(...v)-0.05)*10)/10), hi=Math.min(cap||1,Math.ceil((Math.max(...v)+0.05)*10)/10); return [lo,hi]; }
const SHORT=["math500","aime2026","hmmt2026","humaneval","livecodebench","bbh","ultrachat","wildchat"];
const LONG=["arxiv","govreport","multilexsum","narrativeqa","mrcr","swebench_live","longbench_v2"];
const dsOrder = arr => SHORT.concat(LONG).filter(d=>arr.includes(d)).concat(arr.filter(d=>!SHORT.includes(d)&&!LONG.includes(d)).sort());
function band(pts, m, key, lo, hi, xs){
  const c=COLOR(m), x=pts.map(p=>p.x+(xs||0));
  return [{x:x.concat(x.slice().reverse()), y:pts.map(p=>p[hi]??p[key]).concat(pts.slice().reverse().map(p=>p[lo]??p[key])),
    fill:"toself", fillcolor:c+"26", line:{width:0}, mode:"lines", hoverinfo:"skip", showlegend:false, type:"scatter"},
    {x:x, y:pts.map(p=>p[key]), name:LABEL(m), mode:"lines+markers", line:{color:c,width:2}, marker:{size:8,color:c,line:{color:css("--bg"),width:2}},
     customdata:pts.map(p=>[p.n,p.rounds||""]), hovertemplate:"%{y:.1%} · n=%{customdata[0]}<extra>"+LABEL(m)+"</extra>", type:"scatter"}];
}
function fillSelect(id, values, all){ const s=document.getElementById(id); const prev=s.value; s.innerHTML="";
  (all?[["__all","все"]]:[]).concat(values.map(v=>[v,v])).forEach(([v,t])=>{const o=document.createElement("option");o.value=v;o.textContent=t;s.appendChild(o);}); if(prev&&[...s.options].some(o=>o.value===prev)) s.value=prev; return s; }
const methodsIn = arr => ORDER.concat(["ar"]).filter(m=>arr.some(p=>p.method===m));

function renderAll(){
// drafter table
(function(){ const t=document.getElementById("drafters");
  let h="<tr><th>Драфтер</th><th>Тип</th><th>Длина обучения</th></tr>";
  ORDER.forEach(m=>{ const x=META[m]; const mr=D.natural&&D.natural.match_ar[m];
    h+=`<tr><td><span style="color:${COLOR(m)}">■</span> ${x.label}</td><td>${x.kind}</td><td>${x.train}</td></tr>`;});
  t.innerHTML=h; })();

// 1. acceptance vs context
if (D.natural){
  const N=D.natural, dsList=[...new Set(N.curves_ds.map(p=>p.dataset))].sort();
  const s1=fillSelect("c1-ds", dsList, true), met=document.getElementById("c1-metric");
  function draw1(){ const ds=s1.value, key=met.value, inp=false;
    const src = ds==="__all" ? N.curves : N.curves_ds.filter(p=>p.dataset===ds);
    let tr=[]; methodsIn(src).filter(m=>m!=="ar").forEach(m=>{ const pts=src.filter(p=>p.method===m && p.n>=3).sort((a,b)=>a.x-b.x);
      tr=tr.concat(band(pts.map(p=>Object.assign({},p,{x:p.x+ (p.x===0?500:p.x/2)})), m, key, key==="acc"?"lo":"tau_lo", key==="acc"?"hi":"tau_hi")); });
    const ys=tr.filter(t=>t.showlegend!==false).flatMap(t=>t.y);
    Plotly.react("c1", tr, layout({xaxis:{type:"log",range:[Math.log10(700),Math.log10(300000)],title:inp?"длина входа, токенов":"полный контекст перед раундом (вход + сгенерированное), токенов",tickvals:KT,ticktext:KT.map(fmtK),gridcolor:css("--grid")},
      yaxis:{title:key==="acc"?(inp?"acceptance на всём ответе":"acceptance"):"принято за раунд",tickformat:key==="acc"?".0%":".1f",range:key==="acc"?yr(ys):[0,7],gridcolor:css("--grid")}}), cfg);
    document.getElementById("c1-note").textContent="Окно [K, 2K) по полному контексту перед раундом. Точки с менее чем 3 запросами скрыты. Состав окон разный: справа остаются только длинные входы (а 128K+ — в основном MRCR и LongBench v2), поэтому «все сеты» смешивают эффект длины и состава; для честного сравнения выберите сет.";
  }
  s1.onchange=draw1; met.onchange=draw1; draw1();

  // counts table
  const ct=document.getElementById("counts"); const cds=[...new Set(N.counts.map(c=>c.dataset))];
  const DSDESC={
    aime2026:"AIME 2026: олимпиадные задачи по математике (MathArena)",
    hmmt2026:"HMMT февраль 2026: олимпиадная математика (MathArena)",
    math500:"MATH-500: школьная/олимпиадная математика",
    humaneval:"HumanEval: дописать функцию на Python по описанию",
    livecodebench:"LiveCodeBench: задачи спортивного программирования",
    bbh:"BIG-Bench Hard: короткие задачи на логику и рассуждение",
    ultrachat:"UltraChat: синтетические диалоги с ассистентом",
    wildchat:"WildChat: реальные бытовые вопросы пользователей (отобраны вручную)",
    arxiv:"длинный вход: научная статья arXiv, нужно пересказать",
    govreport:"длинный вход: отчёт госагентства США, нужно пересказать",
    multilexsum:"длинный вход: материалы судебного дела, нужно резюме",
    narrativeqa:"длинный вход: книга или сценарий, вопрос по сюжету",
    mrcr:"длинный вход: длинная переписка (OpenAI MRCR), нужно найти и дословно повторить один из ответов",
    swebench_live:"длинный вход: issue + файлы репозитория (SWE-bench-Live), нужен патч",
    longbench_v2:"длинный вход: документ + вопрос с вариантами ответа (LongBench v2)"};
  const medIn=d=>{ const v=N.per_request.filter(r=>r.dataset===d&&r.method===N.per_request[0].method).map(r=>r.prompt_tokens).sort((a,b)=>a-b); return v.length?v[Math.floor(v.length/2)]:null; };
  const allSame=cds.every(d=>new Set(methodsIn(N.counts).map(m=>(N.counts.find(x=>x.dataset===d&&x.method===m)||{}).n)).size===1);
  let h="<tr><th>Сет</th><th>Что это</th><th class=\"num\">вход, медиана</th>"+(allSame?"<th class=\"num\">запросов</th>":methodsIn(N.counts).map(m=>`<th>${LABEL(m)}</th>`).join(""))+"</tr>";
  dsOrder(cds).forEach(d=>{const mi=medIn(d); h+=`<tr><td>${d}</td><td>${DSDESC[d]||""}</td><td class="num">${mi==null?"—":fmtK(Math.round(mi/100)*100)}</td>`+(allSame?`<td class="num">${N.counts.find(x=>x.dataset===d).n}</td>`:methodsIn(N.counts).map(m=>{const c=N.counts.find(x=>x.dataset===d&&x.method===m);return `<td class="num">${c?c.n:"—"}</td>`;}).join(""))+"</tr>";});
  ct.innerHTML=h;
}

// 2. noeos
if (D.noeos){
  const E=D.noeos, dsList=[...new Set(E.curves_ds.map(p=>p.dataset))].sort();
  const s4=fillSelect("c4-ds", dsList, true);
  function draw4(){ const ds=s4.value; const src= ds==="__all"?E.curves:E.curves_ds.filter(p=>p.dataset===ds);
    let tr=[]; methodsIn(src).filter(m=>m!=="ar").forEach(m=>{ tr=tr.concat(band(src.filter(p=>p.method===m).sort((a,b)=>a.x-b.x).map(p=>Object.assign({},p,{x:p.x+1024})), m,"acc","lo","hi")); });
    Plotly.react("c4", tr, layout({xaxis:Object.assign({title:"сгенерировано токенов",gridcolor:css("--grid")},GT),
      yaxis:{title:"acceptance",tickformat:".0%",range:yr(tr.filter(t=>t.showlegend!==false).flatMap(t=>t.y)),gridcolor:css("--grid")}}), cfg); }
  s4.onchange=draw4; draw4();
  const s5=document.getElementById("c5-by");
  function draw5(){ const by=s5.value; let tr=[];
    if(by==="method"){ methodsIn(E.cycle_prevalence).forEach(m=>{const pts=E.cycle_prevalence.filter(p=>p.method===m).sort((a,b)=>a.x-b.x);
      tr.push({x:pts.map(p=>p.x+1024),y:pts.map(p=>p.cyc),name:LABEL(m),mode:"lines",line:{color:COLOR(m),width:2,dash:m==="ar"?"dot":"solid"},hovertemplate:"%{y:.0%}<extra>"+LABEL(m)+"</extra>"});}); }
    else { [...new Set(E.cycle_prevalence_ds.map(p=>p.dataset))].forEach((d,i)=>{const pts=E.cycle_prevalence_ds.filter(p=>p.dataset===d).sort((a,b)=>a.x-b.x);
      tr.push({x:pts.map(p=>p.x+1024),y:pts.map(p=>p.cyc),name:d,mode:"lines",line:{color:css("--fg2"),width:2,dash:DASH[i%DASH.length]},hovertemplate:"%{y:.0%}<extra>"+d+"</extra>"});}); }
    Plotly.react("c5", tr, layout({xaxis:Object.assign({title:"сгенерировано токенов",gridcolor:css("--grid")},GT),
      yaxis:{tickformat:".0%",range:[0,1],gridcolor:css("--grid")}}), cfg); }
  s5.onchange=draw5; draw5();
  if (E.eos_wanted){
    const fe=E.first_eos.filter(r=>r.first!=null);
    document.getElementById("c7-title").textContent=`Где target впервые хочет EOS (${fe.length} из ${E.first_eos.length} траекторий AR)`;
    Plotly.react("c7",[{x:fe.map(r=>r.first),type:"histogram",xbins:{size:512},marker:{color:css("--s1"),line:{color:css("--bg"),width:2}},hovertemplate:"%{x}: %{y}<extra></extra>"}],
      layout({hovermode:"closest",xaxis:{title:"позиция, токенов",gridcolor:css("--grid")},yaxis:{title:"траекторий",gridcolor:css("--grid")}}), cfg);
    const cats=[["до первого EOS",false,E.acc_by_eos,"after"],["после первого EOS",true,E.acc_by_eos,"after"],["вне цикла",false,E.acc_by_cycle,"cyc"],["в цикле",true,E.acc_by_cycle,"cyc"]];
    let tr=[]; ORDER.forEach((m,i)=>{ const ys=cats.map(([n,v,arr,k])=>{const p=(arr||[]).find(q=>q.method===m&&q[k]===v);return p||{};});
      if(!ys.some(p=>p.acc!=null))return;
      tr.push({x:cats.map(c=>c[0]),y:ys.map(p=>p.acc),error_y:{type:"data",symmetric:false,array:ys.map(p=>p.hi!=null?p.hi-p.acc:0),arrayminus:ys.map(p=>p.lo!=null?p.acc-p.lo:0),color:COLOR(m)},
        type:"scatter",mode:"markers",marker:{size:10,color:COLOR(m),line:{color:css("--bg"),width:2}},name:LABEL(m),offsetgroup:m,hovertemplate:"%{y:.1%}<extra>"+LABEL(m)+"</extra>"}); });
    Plotly.react("c8",tr,layout({hovermode:"closest",scattermode:"group",yaxis:{tickformat:".0%",range:yr(tr.flatMap(t=>t.y)),gridcolor:css("--grid")}}),cfg);
    if (E.acc_by_entropy){ const bins=["[0.0, 0.01)","[0.01, 0.1)","[0.1, 0.3)","[0.3, 1.0)","[1.0, 2.0)","[2.0, 10.0)"]; let tr2=[];
      ORDER.forEach(m=>{const pts=bins.map(b=>(E.acc_by_entropy.find(p=>p.method===m&&p.ent_bin===b)||{}).acc??null); if(pts.every(v=>v==null))return;
        tr2.push({x:bins,y:pts,name:LABEL(m),mode:"lines+markers",line:{color:COLOR(m),width:2},marker:{size:8},hovertemplate:"%{y:.1%}<extra>"+LABEL(m)+"</extra>"});});
      Plotly.react("c9",tr2,layout({xaxis:{title:"энтропия target в позиции раунда",gridcolor:css("--grid")},yaxis:{tickformat:".0%",range:[0,1],gridcolor:css("--grid")}}),cfg); }
  }
  if (E.cycle_onset){ const ms=methodsIn(E.cycle_onset); const tot=m=>(E.n_traj||{})[m]??"?";
    Plotly.react("c10", ms.slice().reverse().map(m=>{const q=E.cycle_onset.filter(p=>p.method===m); return {type:"box",orientation:"h",x:q.map(p=>p.t),name:LABEL(m)+` · ${q.length}/${tot(m)}`,boxpoints:"all",jitter:0.5,pointpos:0,
      marker:{color:COLOR(m),size:7,line:{color:css("--bg"),width:1}},line:{color:COLOR(m),width:1.5},fillcolor:"rgba(0,0,0,0)",text:q.map(p=>p.dataset+" · "+p.id),hovertemplate:"%{text}<br>%{x} ток.<extra></extra>"};}),
      layout({showlegend:false,hovermode:"closest",margin:{l:190,r:10,t:10,b:50},xaxis:Object.assign({title:"начало цикла, токенов",gridcolor:css("--grid"),range:[0,50000]},GT),yaxis:{gridcolor:css("--grid")}}), cfg);
    const per=E.cycle_period||[];
    Plotly.react("c11",[{type:"histogram",x:per.map(v=>Math.log2(v)),xbins:{size:0.5},marker:{color:css("--ar"),line:{color:css("--bg"),width:2}},hovertemplate:"период ≈ 2^%{x}: %{y}<extra></extra>"}],
      layout({hovermode:"closest",margin:{l:50,r:10,t:10,b:50},xaxis:{title:"период цикла, токенов",range:[-0.3,11.3],tickvals:[0,2,4,6,8,10,11],ticktext:["1","4","16","64","256","1024","2048"],gridcolor:css("--grid")},yaxis:{title:"траекторий AR",gridcolor:css("--grid")}}), cfg); }
  const ex=document.getElementById("examples"); let eh="<tr><th>Сет</th><th>Пример</th><th class=\"num\">первый EOS top-1</th><th>Участки</th></tr>";
  (E.examples||[]).forEach(e=>{ const u=at=>"examples.html#"+["noeos",e.id,"ar","",at,1024].map(encodeURIComponent).join("/");
    eh+=`<tr><td>${e.dataset}</td><td><a href="${u(e.first_eos??0)}">${e.id}</a></td><td class="num">${e.first_eos??"нет"}</td><td>${e.tails.map(t=>`<a href="${u(t.at+Math.floor(t.len/2))}">${t.at}–${t.at+t.len}</a>`).join(" · ")}</td></tr>`; });
  ex.innerHTML=eh;
}

// 3. teacher forcing: same text for every drafter
if (D.forced && D.forced.position) (function(){
  const F=D.forced.position, xc=(x,x1)=>Math.sqrt(Math.max(x,128)*Math.min(x1,1<<20));
  const opt=(id,items)=>{ const s=document.getElementById(id), prev=s.value; s.innerHTML=items.map(([v,t])=>`<option value="${v}">${t}</option>`).join(""); if(items.some(([v])=>v===prev)) s.value=prev; return s; };
  const g=[...new Set(F.pos.map(p=>p.group))], ds=g.filter(x=>!["all","short","long"].includes(x));
  const sg=opt("f-group",[["all","все"],["short","короткие входы"],["long","длинные входы"]].filter(([v])=>g.includes(v)).concat(dsOrder(ds).map(d=>[d,d])));
  const sk=document.getElementById("f-kind"), sph=document.getElementById("f-phase");
  function draw1(){ const pts=F.pos.filter(p=>p.group===sg.value&&p.kind===sk.value&&(p.phase||"all")===sph.value), tr=[];
    methodsIn(pts).forEach(m=>{ const q=pts.filter(p=>p.method===m&&p.n>=3).sort((a,b)=>a.x-b.x); if(!q.length) return;
      tr.push(...band(q.map(p=>Object.assign({},p,{x:xc(p.x,p.x1)})),m,"acc","lo","hi"));
      tr[tr.length-1].customdata=q.map(p=>[p.n,fmtT(p.x)+"–"+fmtT(p.x1)]); tr[tr.length-1].hovertemplate="%{customdata[1]}: %{y:.1%} · трейсов %{customdata[0]}<extra>"+LABEL(m)+"</extra>"; });
    Plotly.react("f1",tr,layout({xaxis:{type:"log",title:"сгенерировано токенов (позиция в ответе)",tickvals:[256,1000,4000,16000,64000],ticktext:["256","1K","4K","16K","64K"],gridcolor:css("--grid")},
      yaxis:{title:"acceptance",tickformat:".0%",range:yr(tr.filter(t=>t.showlegend!==false).flatMap(t=>t.y)),gridcolor:css("--grid")}}),cfg); }
  sg.onchange=draw1; sk.onchange=draw1; sph.onchange=draw1; draw1();
})();

// 1. input length at a fixed answer window, own vs common text
(function(){
  const modes=["forced","natural"].filter(p=>D[p]&&D[p].input); if(!modes.length) return;
  const sm=document.getElementById("in-mode"); [...sm.options].forEach(o=>{ if(!modes.includes(o.value)) o.remove(); });
  const sd=document.getElementById("in-ds");
  function fillDs(){ const I=D[sm.value].input, prev=sd.value, g=[...new Set(I.bins.map(p=>p.group))].filter(x=>x!=="long");
    sd.innerHTML=[["long","все длинные сеты"]].concat(dsOrder(g).map(d=>[d,d])).map(([v,t])=>`<option value="${v}">${t}</option>`).join("");
    if([...sd.options].some(o=>o.value===prev)) sd.value=prev; }
  function draw(){ const I=D[sm.value].input, pts=I.bins.filter(p=>p.group===sd.value), tr=[];
    methodsIn(pts).filter(m=>m!=="ar").forEach(m=>{ const q=pts.filter(p=>p.method===m&&p.n>=3).sort((a,b)=>a.x-b.x); if(!q.length) return;
      tr.push(...band(q.map(p=>Object.assign({},p,{x:p.p})),m,"acc","lo","hi"));
      tr[tr.length-1].customdata=q.map(p=>[p.n,fmtK(p.x)+"–"+(p.x1>1e8?"…":fmtK(p.x1))]); tr[tr.length-1].hovertemplate="вход %{customdata[1]}: %{y:.1%} · трейсов %{customdata[0]}<extra>"+LABEL(m)+"</extra>"; });
    Plotly.react("in1",tr,layout({xaxis:{type:"log",title:"длина входа, токенов (точка — геометрическое среднее бина)",tickvals:KT,ticktext:KT.map(fmtK),gridcolor:css("--grid")},
      yaxis:{title:"acceptance, первые 2K ответа",tickformat:".0%",range:yr(tr.filter(t=>t.showlegend!==false).flatMap(t=>t.y)),gridcolor:css("--grid")}}),cfg);
    document.getElementById("in1-note").textContent=(sm.value==="forced"?"Все драфтеры на одном тексте AR, поэтому в каждом бине одни и те же документы.":"У каждого драфтера свой текст; состав бинов одинаков (те же промпты), но ответы разные.")
      +" Бины по длине входа удваиваются, точки с менее чем 3 трейсами скрыты; полоса — 95% bootstrap по трейсам."; }
  const f=x=>(x>=0?"+":"")+(100*x).toFixed(2);
  let h="<tr><th>Драфтер</th>"+modes.map(p=>`<th class="num">${p==="forced"?"общий текст":"свой текст"}: п.п. на удвоение [95% CI]</th>`).join("")+"<th class=\"num\">трейсов</th></tr>";
  ORDER.forEach(m=>{ const ss=modes.map(p=>D[p].input.slopes.find(x=>x.method===m)); if(!ss.some(Boolean)) return;
    h+=`<tr><td><span style="color:${COLOR(m)}">■</span> ${LABEL(m)}</td>`+ss.map(s=>`<td class="num">${s?`${f(s.slope)} [${f(s.lo)}; ${f(s.hi)}]`:"—"}</td>`).join("")+`<td class="num">${ss.map(s=>s?s.n:"—").join(" / ")}</td></tr>`; });
  document.getElementById("in-t").innerHTML=h;
  sm.onchange=()=>{ fillDs(); draw(); }; sd.onchange=draw; fillDs(); draw();
})();

// 4. within one trace
(function(){
  const panels=["forced"].filter(p=>D[p]&&D[p].within); if(!panels.length) return;
  const PL={natural:"свой текст · EOS разрешён",noeos:"свой текст · EOS запрещён",forced:"текст AR (teacher forcing) · EOS запрещён"};
  const opt=(id,items)=>{ const s=document.getElementById(id), prev=s.value; s.innerHTML=items.map(([v,t])=>`<option value="${v}">${t}</option>`).join(""); if(items.some(([v])=>v===prev)) s.value=prev; return s; };
  const sp=opt("w-panel",panels.map(p=>[p,PL[p]])), sa=document.getElementById("w-axis"), sk=document.getElementById("w-kind"), sw=document.getElementById("w-phase");
  const W=()=>D[sp.value].within, A=()=>W()[sa.value], E_=()=>W().edges;
  const xc=(x,x1)=>Math.sqrt(Math.max(x,128)*x1);
  function groups(){ const g=A().curves.map(c=>c.group), ds=[...new Set(g)].filter(x=>!["all","short","long"].includes(x));
    return [["all","все"],["short","короткие входы"],["long","длинные входы"]].filter(([v])=>g.includes(v)).concat(dsOrder(ds).map(d=>[d,d])); }
  const sg=opt("w-group",groups());
  function draw1(){ const pts=A().curves.filter(c=>c.group===sg.value&&c.kind===sk.value&&(c.phase||"all")===sw.value), tr=[];
    methodsIn(pts).forEach(m=>{ const q=pts.filter(p=>p.method===m&&p.n>=3).sort((a,b)=>a.x-b.x); if(!q.length) return;
      const c=COLOR(m), x=q.map(p=>xc(p.x,p.x1));
      tr.push({x:x.concat(x.slice().reverse()),y:q.map(p=>100*(p.hi??p.mean)).concat(q.slice().reverse().map(p=>100*(p.lo??p.mean))),fill:"toself",fillcolor:c+"26",line:{width:0},mode:"lines",hoverinfo:"skip",showlegend:false});
      tr.push({x,y:q.map(p=>100*p.mean),name:LABEL(m),mode:"lines+markers",line:{color:c,width:2},marker:{size:8,color:c,line:{color:css("--bg"),width:2}},
        customdata:q.map(p=>[p.acc,p.n,p.cyc,fmtT(p.x)+"–"+fmtT(p.x1)]),hovertemplate:"%{customdata[3]}: %{y:+.1f} п.п. · acc %{customdata[0]:.1%} · трейсов %{customdata[1]} · в цикле %{customdata[2]:.0%}<extra>"+LABEL(m)+"</extra>"}); });
    const ys=tr.filter(t=>t.mode==="lines+markers").flatMap(t=>t.y);
    Plotly.react("w1",tr,layout({xaxis:{type:"log",title:sa.value==="gen"?"сгенерировано токенов в трейсе":"полная длина контекста",tickvals:[256,1000,4000,16000,64000,256000],ticktext:["256","1K","4K","16K","64K","256K"],gridcolor:css("--grid")},
      yaxis:{title:"изменение, п.п.",ticksuffix:"",zeroline:true,zerolinecolor:css("--fg2"),zerolinewidth:1.5,gridcolor:css("--grid"),range:[Math.min(-5,Math.floor(Math.min(...ys,0)/5)*5-2),Math.max(5,Math.ceil(Math.max(...ys,0)/5)*5+2)]}}),cfg); }
  const GM=()=>ORDER.filter(m=>(D[sp.value].grid||{cells:[]}).cells.some(c=>c.method===m)).map(m=>[m,LABEL(m)]);
  const sh=opt("w-hm",GM());
  function draw3(){ const G=D[sp.value].grid, t=document.getElementById("w3"); if(!G){ t.innerHTML=""; return; }
    const ie=G.in_edges, ge=G.gen_edges, kind=sk.value==="d_nc"?"nc":"all", rng=(e,i)=>fmtT(e[i])+"–"+(i+1<e.length?fmtT(e[i+1]):"…");
    const cells=G.cells.filter(c=>c.method===sh.value&&c.kind===kind&&c.n>=3), at=new Map(cells.map(c=>[c.i+":"+c.g,c]));
    const ii=ie.map((_,i)=>i).filter(i=>cells.some(c=>c.i===i)), gg=ge.map((_,g)=>g).filter(g=>cells.some(c=>c.g===g));
    const f=v=>(v>=0?"+":"")+(100*v).toFixed(1), ds=cells.filter(c=>c.g>0).map(c=>c.d).sort((a,b)=>a-b), q=p=>ds.length?ds[Math.min(ds.length-1,Math.floor(p*ds.length))]:0;
    const dmax=Math.max(1e-9,q(0.9)), dmin=Math.min(-1e-9,q(0.1));
    const pos=css("--dpos"), neg=css("--dneg");
    let h=`<tr><th>сгенерировано ↓ / вход →</th>`+ii.map(i=>`<th class="num">${rng(ie,i)}</th>`).join("")+"</tr>";
    gg.forEach(g=>{ h+=`<tr><td>${rng(ge,g)}${g===0?" (база)":""}</td>`+ii.map(i=>{ const c=at.get(i+":"+g); if(!c) return `<td class="heat"></td>`;
      if(g===0) return `<td class="heat" title="acceptance на первых ${fmtT(ge[1])} токенах · трейсов ${c.n}" style="background:var(--dmid)"><b>${(100*c.acc).toFixed(1)}%</b><div class="hn">${c.n}</div></td>`;
      const a=Math.min(1,c.d>=0?c.d/dmax:c.d/dmin), pct=Math.round(4+a*86);
      const tip=`вход ${rng(ie,i)}, сгенерировано ${rng(ge,g)}: ${f(c.d)} п.п.`+(c.lo!=null?` [${f(c.lo)}; ${f(c.hi)}]`:"")+` · acc ${(100*c.acc).toFixed(1)}% (база ${(100*c.base).toFixed(1)}%) · трейсов ${c.n}`;
      return `<td class="heat" title="${tip}" style="background:color-mix(in srgb, ${c.d>=0?pos:neg} ${pct}%, var(--card))${a>0.8?";color:#fff":""}">${f(c.d)}<div class="hn">${c.n}</div></td>`; }).join("")+"</tr>"; });
    t.innerHTML=h; }
  const ek=document.getElementById("we-kind");
  function draw4(){ const box=document.getElementById("w-ent"), E=D[sp.value].entropy;
    if(!E||!E[sa.value]){ box.style.display="none"; return; } box.style.display="";
    const sit=sw.value+":"+(sk.value==="d_nc"?"nc":"all"), d=ek.value==="d";
    const q=E[sa.value].filter(c=>c.group===sg.value&&c.kind===ek.value&&c.sit===sit&&c.n>=3).sort((a,b)=>a.x-b.x), tr=[];
    if(q.length){ const c=css("--fg2"), x=q.map(p=>xc(p.x,p.x1));
      tr.push({x:x.concat(x.slice().reverse()),y:q.map(p=>p.hi??p.mean).concat(q.slice().reverse().map(p=>p.lo??p.mean)),fill:"toself",fillcolor:"rgba(128,128,128,.18)",line:{width:0},mode:"lines",hoverinfo:"skip",showlegend:false});
      tr.push({x,y:q.map(p=>p.mean),name:"энтропия target",mode:"lines+markers",line:{color:c,width:2},marker:{size:8,color:c,line:{color:css("--bg"),width:2}},showlegend:false,
        customdata:q.map(p=>[p.e,p.n,p.cyc,p.think,fmtT(p.x)+"–"+fmtT(p.x1)]),
        hovertemplate:"%{customdata[4]}: "+(d?"%{y:+.3f} нат · ":"")+"энтропия %{customdata[0]:.3f} нат · трейсов %{customdata[1]} · рассуждают %{customdata[3]:.0%} · в цикле %{customdata[2]:.0%}<extra></extra>"}); }
    const xr=document.getElementById("w1").layout?.xaxis?.range;
    Plotly.react("w4",tr,layout({xaxis:{type:"log",range:xr,title:sa.value==="gen"?"сгенерировано токенов в трейсе":"полная длина контекста",tickvals:[256,1000,4000,16000,64000,256000],ticktext:["256","1K","4K","16K","64K","256K"],gridcolor:css("--grid")},
      yaxis:{title:d?"изменение, нат":"энтропия, нат",zeroline:d,zerolinecolor:css("--fg2"),zerolinewidth:1.5,gridcolor:css("--grid"),rangemode:d?"normal":"tozero"}}),cfg); }
  ek.onchange=draw4;
  const all3=()=>{ draw1(); draw3(); draw4(); };
  sp.onchange=()=>{ opt("w-group",groups()); opt("w-hm",GM()); all3(); };
  sa.onchange=()=>{ opt("w-group",groups()); all3(); }; sk.onchange=all3; sw.onchange=()=>{ draw1(); draw4(); }; sg.onchange=all3; sh.onchange=draw3; all3();
})();
}

function setTheme(t){ document.documentElement.dataset.theme=t; try{localStorage.setItem("theme",t);}catch(e){}
  document.getElementById("theme").textContent = t==="dark" ? "☀ светлая" : "☾ тёмная"; renderAll(); }
document.getElementById("theme").onclick=()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark");
setTheme(document.documentElement.dataset.theme||"light");
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="report/data.json")
    p.add_argument("--out", default="report/index.html")
    a = p.parse_args()
    data = json.loads(Path(a.data).read_text())
    intro = ("Один target (Qwen3.8-27B, BF16, нативный контекст 262 144), пять драфтеров, одинаковые промпты. "
             "Под greedy спекуляция в теории не меняет текст, но из-за BF16-численности (зависит от состава батча) длинные ответы "
             "расходятся, поэтому драфтеры проверяются на похожем, но не идентичном тексте; доля побайтово совпавших с AR ответов — в таблице.")
    noeos = ("Короткие промпты из ветки issue11 (MATH-500, HumanEval, UltraChat, BBH, по 20), генерация 49 152 токенов: "
             "ignore_eos и запрет EOS у target через bad_words (как в issue11). AR дополнительно пишет top-20 logprobs target "
             "до запрета, поэтому видно, где модель хотела остановиться.")
    protocol = ("vLLM 0.32, TP=2 на A100 80GB, prefix caching выключен, K=7. EAGLE-4 запускается через vLLM-форк авторов "
                "(eagle_harness, статическое дерево node_top_k=1 = цепочка из 7). Acceptance запроса = сумма принятых draft-токенов / сумма предложенных (обычно 7 за раунд; принимается префикс до первого "
                "несовпадения), без бонусного токена; в окне — среднее по запросам. Код: specbench/ (на основе runner из ветки issue11).")
    forced = ("Эталон — одна жадная траектория target на промпт с запрещённым EOS (49 152 токена; для длинных входов — до полного "
              "контекста 128K), обрезанная на старте точного цикла + 4096 токенов. Каждый драфтер проходит этот текст под teacher "
              "forcing: logits target перед проверкой заменяются one-hot на эталонном токене, поэтому все драфтеры проверяются на "
              "побайтово одном тексте (каждый запрос сверен с эталоном). Это убирает и смену состава окон, и расхождение текстов.")
    html = (HTML.replace("__DATA__", json.dumps(data)).replace("__META__", json.dumps(DRAFTERS))
            .replace("__FORCED_INTRO__", forced)
            .replace("__INTRO__", intro).replace("__NOEOS_INTRO__", noeos).replace("__PROTOCOL__", protocol))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
