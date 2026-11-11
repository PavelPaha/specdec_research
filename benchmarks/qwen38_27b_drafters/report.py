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
th{color:var(--fg2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}td.heat{text-align:center;font-variant-numeric:tabular-nums}
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
<div class="card"><div class="ctitle">Состав трейсов на каждой позиции графика выше</div><div id="f2" class="plot small"></div>
<div class="note">Доли считаются среди трейсов, которые дошли до этой позиции (их число — серая линия и подсказка). Число падает, потому что эталон каждого трейса обрезан на старте цикла + 4K токенов: трейсы с ранним циклом заканчиваются рано, правее остаются трейсы без цикла или с поздним циклом. «Рассуждают» — первый &lt;/think&gt; эталона ещё впереди; «в цикле» — детектор уже нашёл цикл (проверка каждые 1K токенов, с 2K).</div></div>
<h2>3. Внутри одного трейса</h2>
<p>Чтобы отделить эффект длины от смены состава окон, каждый трейс сравнивается сам с собой: acceptance в окне минус acceptance того же трейса на его первых 2K сгенерированных токенов. Окна — полуоктавы; ячейка трейса учитывается, если в ней ≥16 раундов. «Без циклов» — только раунды вне обнаруженных циклов (детектор как в разделе 5), база та же. «Фаза» оставляет раунды только до или только после первого &lt;/think&gt; трейса; база — первые 2K токенов этой фазы (для ответа — от &lt;/think&gt;). Тепловая карта всегда по всем фазам.</p>
<div class="card"><div class="controls">
<label>Панель <select id="w-panel"></select></label>
<label>Ось <select id="w-axis"><option value="gen">позиция в ответе</option><option value="ctx">полный контекст</option></select></label>
<label>Фаза <select id="w-phase"><option value="all">всё</option><option value="think">рассуждение (до &lt;/think&gt;)</option><option value="answer">ответ (после &lt;/think&gt;)</option></select></label>
<label>Раунды <select id="w-kind"><option value="d">все</option><option value="d_nc">без циклов</option></select></label>
<label>Трейсы <select id="w-group"></select></label></div>
<div class="ctitle">Парное изменение acceptance относительно начала трейса</div><div id="w1" class="plot"></div>
<div class="note">Точка — среднее по трейсам изменения в окне (п.п.), полоса — 95% bootstrap по трейсам. Дальние окна есть только у длинных трейсов, их число — в подсказке.</div></div>
<div class="card"><div class="ctitle">Наклон внутри трейса: изменение acceptance на каждое удвоение длины</div><div id="w2" class="plot small"></div><table id="w2t"></table>
<div class="note">Для каждого трейса — взвешенная по числу раундов регрессия acceptance окна на log₂(позиция); точка — среднее по трейсам, отрезок — 95% bootstrap. «Падают» — доля трейсов с отрицательным наклоном.</div></div>
<div class="card"><div class="controls"><label>Драфтер <select id="w-hm"></select></label><label>Сортировка <select id="w-sort"><option value="ds">по сету</option><option value="s">по наклону</option><option value="base">по начальному acceptance</option></select></label></div>
<div class="ctitle">Все трейсы: изменение acceptance относительно начала</div><div id="w3" class="plot" style="height:640px"></div>
<div class="note">Строка — трейс, столбец — окно; синий — выше своего начала, красный — ниже, серый — без изменений (шкала ±30 п.п.). Клик по ячейке открывает трейс в примерах на этой позиции.</div></div>

<h2>4. Естественная генерация: acceptance по полному контексту</h2>
<p>Свой текст, естественная остановка по EOS, thinking включён, greedy, K=7 цепочкой. По оси X — полная длина контекста перед раундом проверки (вход + уже сгенерированное); каждый раунд попадает в своё окно, длинный ответ даёт точки в нескольких окнах. Это то, что видит пользователь, но здесь смешаны длина входа, позиция в ответе и состав сетов — для выводов о длине см. раздел 1. Шкала логарифмическая; точка — среднее по запросам в окне, полоса — 95% bootstrap-интервал по запросам.</p>
<div class="card"><div class="controls">
<label>Сет <select id="c1-ds"></select></label>
<label>Метрика <select id="c1-metric"><option value="acc">acceptance (принято / предложено)</option><option value="tau">принятых токенов за раунд</option></select></label>
</div><div id="c1" class="plot"></div><div class="note" id="c1-note"></div></div>

<h3>По сетам</h3>
<div class="card"><div class="ctitle">Acceptance на всём ответе, по сетам</div><div id="c3" class="plot" style="height:560px"></div><div class="note">Сверху — короткие входы, снизу — длинные. Цвет нормирован на диапазон таблицы.</div></div>

<h2>5. Что происходит, когда EOS запрещён</h2>
<p>Свой текст. __NOEOS_INTRO__</p>
<div class="card"><div class="ctitle">Acceptance по позиции генерации (EOS запрещён)</div><div class="controls"><label>Сет <select id="c4-ds"></select></label></div><div id="c4" class="plot"></div></div>
<div class="card"><div class="ctitle">Доля траекторий, находящихся в цикле</div><div class="controls"><label>Разбивка <select id="c5-by"><option value="method">по методам (свои траектории)</option><option value="dataset">по сетам (AR)</option></select></label></div>
<div id="c5" class="plot small"></div><div class="note">Цикл: последние 2048 токенов периодичны с периодом ≤128 (≥99% совпадений со сдвигом) или последние 8192 — с периодом 129–2048. Детектор как в ветке issue11. «По методам» — каждый метод на своём тексте, поэтому кривые не совпадают с AR; «по сетам» — только траектории AR.</div></div>
<div class="card"><div class="ctitle">Как часто target хочет закончить (EOS — top-1)</div><div id="c6" class="plot small"></div><div class="note">Доля позиций, где у target (AR, сырые logprobs до запрета) EOS — top-1, т.е. модель «хотела» остановиться.</div></div>
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

  // heatmap: dataset x method
  const ms=ORDER.filter(m=>N.overall_ds.some(p=>p.method===m));
  const dss=dsOrder([...new Set(N.overall_ds.map(p=>p.dataset))]);
  const z=dss.map(d=>ms.map(m=>{const p=N.overall_ds.find(q=>q.method===m&&q.dataset===d);return p?p.acc:null;}));
  Plotly.react("c3",[{type:"heatmap",z:z,x:ms.map(LABEL),y:dss,colorscale:[[0,"#cde2fb"],[0.5,"#3987e5"],[1,"#0d366b"]],zmin:Math.min(...z.flat().filter(v=>v!=null)),zmax:Math.max(...z.flat().filter(v=>v!=null)),
    text:z.map(r=>r.map(v=>v==null?"":(100*v).toFixed(0)+"%")),texttemplate:"%{text}",hovertemplate:"%{y} · %{x}: %{z:.1%}<extra></extra>",
    xgap:2,ygap:2,colorbar:{tickformat:".0%",title:""}}], layout({margin:{l:120,r:20,t:10,b:60},hovermode:"closest",xaxis:{side:"bottom"},yaxis:{autorange:"reversed"},shapes:dss.some(d=>LONG.includes(d))&&dss.some(d=>SHORT.includes(d))?[{type:"line",xref:"paper",x0:0,x1:1,y0:dss.filter(d=>!LONG.includes(d)).length-0.5,y1:dss.filter(d=>!LONG.includes(d)).length-0.5,line:{color:css("--fg"),width:2}}]:[]}), cfg);

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
    const pts=E.eos_wanted.sort((a,b)=>a.x-b.x);
    const tr6=[{x:pts.map(p=>p.x+1024),y:pts.map(p=>p.wanted),mode:"lines+markers",line:{color:css("--fg"),width:3},marker:{size:8,color:css("--fg"),line:{color:css("--bg"),width:2}},name:"все сеты",hovertemplate:"%{y:.2%}<extra>все</extra>"}];
    dsOrder([...new Set((E.eos_wanted_ds||[]).map(p=>p.dataset))]).forEach((d,i)=>{const q=E.eos_wanted_ds.filter(p=>p.dataset===d).sort((a,b)=>a.x-b.x);
      tr6.push({x:q.map(p=>p.x+1024),y:q.map(p=>p.wanted),mode:"lines",line:{color:css("--fg2"),width:1.5,dash:DASH[(i+1)%DASH.length]},name:d,hovertemplate:"%{y:.2%}<extra>"+d+"</extra>"});});
    Plotly.react("c6",tr6,layout({xaxis:Object.assign({title:"сгенерировано токенов",gridcolor:css("--grid")},GT),yaxis:{tickformat:".1%",rangemode:"tozero",gridcolor:css("--grid")}}), cfg);
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
  const cp=(F.comp||[]).slice().sort((a,b)=>a.x-b.x), N0=cp.length?cp[0].n:1, cd=cp.map(p=>[p.n,fmtT(p.x)]);
  Plotly.react("f2",[
    {x:cp.map(p=>p.x),y:cp.map(p=>p.n/N0),mode:"lines+markers",line:{color:css("--muted"),width:2},marker:{size:6},name:"дошли до позиции",customdata:cd,hovertemplate:"%{customdata[1]}: %{customdata[0]} трейсов (%{y:.0%})<extra></extra>"},
    {x:cp.map(p=>p.x),y:cp.map(p=>p.think),mode:"lines+markers",line:{color:css("--s1"),width:2},marker:{size:6},name:"из них ещё рассуждают",customdata:cd,hovertemplate:"%{customdata[1]}: %{y:.0%} из %{customdata[0]}<extra>рассуждают</extra>"},
    {x:cp.map(p=>p.x),y:cp.map(p=>p.cyc),mode:"lines+markers",line:{color:css("--s2"),width:2},marker:{size:6},name:"из них уже в цикле",customdata:cd,hovertemplate:"%{customdata[1]}: %{y:.0%} из %{customdata[0]}<extra>в цикле</extra>"}],
    layout({showlegend:true,hovermode:"closest",xaxis:{type:"log",title:"сгенерировано токенов",tickvals:[256,1024,4096,16384,65536],ticktext:["256","1K","4K","16K","64K"],gridcolor:css("--grid")},yaxis:{tickformat:".0%",range:[0,1.02],gridcolor:css("--grid")}}),cfg);
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
  const panels=["forced","natural","noeos"].filter(p=>D[p]&&D[p].within); if(!panels.length) return;
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
  function draw2(){ const S=A().slopes.filter(x=>x.group===sg.value&&(x.phase||"all")===sw.value), ms=ORDER.filter(m=>S.some(x=>x.method===m)), tr=[];
    [["s","все раунды","circle"],["s_nc","без циклов","diamond"]].forEach(([k,name,sym],i)=>{ const q=ms.map(m=>S.find(x=>x.method===m&&x.kind===k)).filter(Boolean);
      tr.push({type:"scatter",mode:"markers",name,y:q.map(x=>LABEL(x.method)),x:q.map(x=>100*x.mean),
        error_x:{type:"data",symmetric:false,array:q.map(x=>100*((x.hi??x.mean)-x.mean)),arrayminus:q.map(x=>100*(x.mean-(x.lo??x.mean))),color:i?css("--muted"):css("--fg2"),thickness:1.5,width:4},
        marker:{size:11,symbol:sym,color:q.map(x=>COLOR(x.method)),line:{color:i?css("--fg"):css("--bg"),width:i?1.5:2}},offsetgroup:k,
        customdata:q.map(x=>[100*x.median,x.neg,x.n]),hovertemplate:"%{x:+.2f} п.п. на удвоение · медиана %{customdata[0]:+.2f} · падают %{customdata[1]:.0%} · трейсов %{customdata[2]}<extra>"+name+"</extra>"}); });
    Plotly.react("w2",tr,layout({hovermode:"closest",scattermode:"group",margin:{l:150,r:20,t:36,b:50},xaxis:{title:"п.п. на удвоение длины",range:(()=>{const v=tr.flatMap(t=>t.x.map((x,i)=>[x+t.error_x.array[i],x-t.error_x.arrayminus[i]])).flat(); const lo=Math.min(0,...v), hi=Math.max(0,...v), pad=(hi-lo)*0.08||1; return [lo-pad,hi+pad];})(),zeroline:true,zerolinecolor:css("--fg2"),zerolinewidth:1.5,gridcolor:css("--grid")},yaxis:{autorange:"reversed",gridcolor:css("--grid")}}),cfg);
    const f=(x)=>x?`${(100*x.mean>=0?"+":"")+(100*x.mean).toFixed(2)} [${x.lo==null?"—":(100*x.lo).toFixed(2)}; ${x.hi==null?"—":(100*x.hi).toFixed(2)}]`:"—";
    let h="<tr><th>Драфтер</th><th class=\"num\">все раунды, п.п./удвоение</th><th class=\"num\">падают</th><th class=\"num\">без циклов</th><th class=\"num\">падают</th><th class=\"num\">трейсов</th></tr>";
    ms.forEach(m=>{ const a=S.find(x=>x.method===m&&x.kind==="s"), b=S.find(x=>x.method===m&&x.kind==="s_nc");
      h+=`<tr><td><span style="color:${COLOR(m)}">■</span> ${LABEL(m)}</td><td class="num">${f(a)}</td><td class="num">${a?(100*a.neg).toFixed(0)+"%":"—"}</td><td class="num">${f(b)}</td><td class="num">${b?(100*b.neg).toFixed(0)+"%":"—"}</td><td class="num">${a?a.n:"—"}</td></tr>`; });
    document.getElementById("w2t").innerHTML=h; }
  const sh=opt("w-hm",ORDER.filter(m=>A().heat[m]).map(m=>[m,LABEL(m)])), ss=document.getElementById("w-sort");
  function draw3(){ const m=sh.value, all=(A().heat[m]||[]), grp=sg.value, ed=E_(), nc=sk.value==="d_nc";
    let rows=all.filter(r=>grp==="all"||(grp==="long"?LONG.includes(r.ds):grp==="short"?!LONG.includes(r.ds):r.ds===grp));
    const ord=dsOrder([...new Set(rows.map(r=>r.ds))]);
    rows=rows.slice().sort(ss.value==="ds"?(a,b)=>ord.indexOf(a.ds)-ord.indexOf(b.ds)||a.p-b.p:ss.value==="s"?(a,b)=>(a.s??0)-(b.s??0):(a,b)=>a.base-b.base);
    const bs=[...new Set(rows.flatMap(r=>r.c.map(c=>c[0])))].sort((a,b)=>a-b);
    const z=rows.map(r=>{ const m_=new Map(r.c.map(c=>[c[0],c])); return bs.map(b=>{ const c=m_.get(b); const v=c?(nc?c[2]:c[1]):null; return v==null?null:100*(v-r.base); }); });
    const cd=rows.map(r=>{ const m_=new Map(r.c.map(c=>[c[0],c])); return bs.map(b=>{ const c=m_.get(b); const v=c?(nc?c[2]:c[1]):null; return [r.id,r.ds,v,c?c[4]:0,c?c[3]:0,r.base,ed[b],fmtT(ed[b])+"–"+fmtT(ed[b+1]),v==null?"":((v-r.base>=0?"+":"")+(100*(v-r.base)).toFixed(1))]; }); });
    const shapes=[]; if(ss.value==="ds"){ let prev=null; rows.forEach((r,i)=>{ if(prev!==null&&r.ds!==prev) shapes.push({type:"line",xref:"paper",x0:0,x1:1,y0:i-0.5,y1:i-0.5,line:{color:css("--fg2"),width:1}}); prev=r.ds; }); }
    const annos=ss.value==="ds"?ord.map(d=>{ const idx=rows.map((r,i)=>r.ds===d?i:-1).filter(i=>i>=0); return {xref:"paper",x:-0.005,xanchor:"right",y:(idx[0]+idx[idx.length-1])/2,text:d,showarrow:false,font:{size:11,color:css("--fg2")}}; }):[];
    const g=document.getElementById("w3"); g.style.height=Math.max(360,Math.min(1400,rows.length*3+120))+"px";
    Plotly.react(g,[{type:"heatmap",z,x:bs.map(b=>fmtT(ed[b])),y:rows.map((r,i)=>i),customdata:cd,zmin:-30,zmax:30,zmid:0,
      colorscale:[[0,css("--dneg")],[0.5,css("--dmid")],[1,css("--dpos")]],xgap:1,ygap:0,colorbar:{title:{text:"п.п.",side:"right"},ticksuffix:""},
      hovertemplate:"%{customdata[1]} · %{customdata[0]}<br>окно %{customdata[7]}: acc %{customdata[2]:.1%} (начало %{customdata[5]:.1%}) · Δ %{customdata[8]} п.п.<br>раундов %{customdata[3]} · в цикле %{customdata[4]:.0%}<extra></extra>"}],
      layout({hovermode:"closest",margin:{l:ss.value==="ds"?110:30,r:20,t:10,b:60},shapes,annotations:annos,xaxis:{title:{text:(sa.value==="gen"?"сгенерировано токенов в трейсе":"полная длина контекста")+" (начало окна)",standoff:12},tickangle:0,type:"category",automargin:true},yaxis:{autorange:"reversed",showticklabels:false,ticks:"",showgrid:false,zeroline:false,title:{text:"трейсы"}}}),cfg);
    g.removeAllListeners&&g.removeAllListeners("plotly_click");
    g.on("plotly_click",ev=>{ const pt=ev.points[0], c=pt.customdata; if(!c) return;
      const r=all.find(x=>x.id===c[0]); const pos=sa.value==="gen"?c[6]:Math.max(0,c[6]-(r?r.p:0));
      window.open("examples.html#"+[sp.value,c[0],m,"",Math.round(pos),1024].map(encodeURIComponent).join("/"),"_blank"); }); }
  const all3=()=>{ draw1(); draw2(); draw3(); };
  sp.onchange=()=>{ opt("w-group",groups()); opt("w-hm",ORDER.filter(m=>A().heat[m]).map(m=>[m,LABEL(m)])); all3(); };
  sa.onchange=()=>{ opt("w-group",groups()); all3(); }; sk.onchange=all3; sw.onchange=()=>{ draw1(); draw2(); }; sg.onchange=all3; sh.onchange=draw3; ss.onchange=draw3; all3();
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
