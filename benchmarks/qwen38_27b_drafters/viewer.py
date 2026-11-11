"""Write report/examples.html: interactive explorer over report/examples/ (built by examples.py).

Usage: python viewer.py --out report/examples.html
Must sit next to examples/ (loads examples/index.js and per-request .js); works over HTTP and from file://.
"""

import argparse
import json
from pathlib import Path

from report import DRAFTERS

HTML = r"""<!doctype html>
<html lang="ru" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<script>try{const t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t;}catch(e){}</script>
<title>Examples Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--muted:#8a8984;--grid:#e8e7e3;--card:#ffffff;--sel:#eef4fd;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--ar:#8a8984;
--tacc:rgba(27,175,122,.20);--trej:rgba(235,104,52,.38);--teos:#c8325a;--cyc:#2a78d6;--cyc2:#1a9e5a;--diff:#eb6834;
--h0:#f3f1ec;--h1:#0f6e4d}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#8f8e88;--grid:#2e2e2c;--card:#222221;--sel:#1f2b3a;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--ar:#8f8e88;
--tacc:rgba(25,158,112,.28);--trej:rgba(217,89,38,.50);--teos:#ff6f91;--cyc:#5b9df0;--cyc2:#4cc38a;--diff:#f08a5d;--h0:#2a2a28;--h1:#3fd39c}}
:root[data-theme="dark"]{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--muted:#8f8e88;--grid:#2e2e2c;--card:#222221;--sel:#1f2b3a;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--ar:#8f8e88;
--tacc:rgba(25,158,112,.28);--trej:rgba(217,89,38,.50);--teos:#ff6f91;--cyc:#5b9df0;--cyc2:#4cc38a;--diff:#f08a5d;--h0:#2a2a28;--h1:#3fd39c}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
header{display:flex;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--grid);position:sticky;top:0;background:var(--bg);z-index:5;flex-wrap:wrap}
header h1{font-size:17px;margin:0 8px 0 0}header a{color:var(--fg2)}
button,select,input{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--grid);border-radius:6px;padding:3px 8px}
button{cursor:pointer}button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}button:disabled{opacity:.4;cursor:default}
.tabs{display:flex;gap:4px}.sp{flex:1}
.wrap{display:grid;grid-template-columns:350px minmax(0,1fr);min-height:calc(100vh - 50px)}
aside{border-right:1px solid var(--grid);padding:12px;display:flex;flex-direction:column;gap:8px;max-height:calc(100vh - 50px);position:sticky;top:50px}
.chips{display:flex;flex-wrap:wrap;gap:4px}.chips button{font-size:12.5px;padding:2px 8px}
.list{overflow:auto;flex:1;border:1px solid var(--grid);border-radius:8px;background:var(--card)}
.row{padding:7px 10px;border-bottom:1px solid var(--grid);cursor:pointer}.row:hover{background:var(--sel)}.row.cur{background:var(--sel);box-shadow:inset 3px 0 0 var(--s1)}
.row .t{font-size:12.5px;font-family:ui-monospace,Menlo,monospace;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .m{font-size:12px;color:var(--fg2);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bar{display:inline-block;width:60px;height:6px;border-radius:3px;background:var(--grid);position:relative;overflow:hidden;vertical-align:middle}.bar i{position:absolute;left:0;top:0;bottom:0;background:var(--s3);border-radius:3px}
.badge{font-size:11px;padding:0 5px;border-radius:4px;border:1px solid var(--grid);color:var(--fg2)}.badge.c{border-color:var(--cyc);color:var(--cyc)}.badge.d{border-color:var(--diff);color:var(--diff)}
main{padding:12px 16px 60px;min-width:0}
.card{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:10px 12px;margin:0 0 12px}
.ctl{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:4px 0}.ctl label{color:var(--fg2)}
.note{font-size:12.5px;color:var(--muted)}h2{font-size:16px;margin:0 0 4px}
#tl{width:100%;display:block;cursor:crosshair;touch-action:none}
#tip{position:fixed;pointer-events:none;background:var(--card);border:1px solid var(--grid);color:var(--fg);border-radius:6px;padding:4px 8px;font-size:12px;display:none;z-index:9;white-space:nowrap}
.cols{display:flex;gap:12px;flex-wrap:wrap}.col{flex:1 1 420px;min-width:0}
.ch{font-size:13px;color:var(--fg2);margin:2px 0 4px}
.tx{white-space:pre-wrap;word-break:break-word;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);border:1px solid var(--grid);border-radius:8px;padding:10px;max-height:62vh;overflow:auto}
.tx .pr{color:var(--muted)}.tx .r{background:var(--trej)}.tx .e{box-shadow:inset 0 -2px 0 var(--teos)}
.tx .gap{color:var(--muted);font-style:italic}
.tx .dv{outline:2px dashed var(--diff)}
.tx .c0{border-bottom:3px solid var(--cyc)}.tx .c1{border-bottom:3px solid var(--cyc2)}
.tx .more{display:block;margin:4px 0;font:12px system-ui;cursor:pointer}
.tx .cb{color:#fff;background:var(--cyc);border-radius:3px;padding:0 4px;font:11px system-ui;margin:0 2px;cursor:pointer}
.unf{white-space:normal;font:12px/1.5 ui-monospace,Menlo,monospace;overflow:auto;max-height:62vh;background:var(--bg);border:1px solid var(--grid);border-radius:8px;padding:6px}
.unf table{border-collapse:collapse}.unf td{padding:0;white-space:pre;vertical-align:top}.unf td.lab{color:var(--muted);padding-right:8px;text-align:right;position:sticky;left:0;background:var(--bg)}
.unf span{border-radius:2px}.unf .x{box-shadow:inset 0 -2px 0 var(--diff);background:color-mix(in srgb,var(--diff) 18%,transparent)}
.unf .nl{color:var(--muted)}
.leg{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--fg2);margin-top:4px}.sw{display:inline-block;width:14px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:4px}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--grid);border-radius:4px;padding:0 4px;color:var(--fg2)}
@media (max-width:900px){.wrap{grid-template-columns:1fr}aside{position:static;max-height:none}.list{max-height:320px}}
</style></head><body>
<header><h1>Примеры</h1><div class="tabs" id="panels"></div><span class="sp"></span>
<a href="index.html">← отчёт</a><button id="theme" type="button"></button></header>
<div id="banner" style="display:none;margin:12px 16px;padding:10px 12px;border:1px solid var(--diff);border-radius:8px;color:var(--fg)"></div>
<div class="wrap">
<aside>
  <div class="chips" id="dss"></div>
  <div class="chips" id="locs" title="где начинается первый цикл (для метода из «acc для»)"></div>
  <div class="ctl"><label>Сортировка</label><select id="sort">
    <option value="p">по длине входа</option><option value="n">по длине ответа</option><option value="acc">по acceptance ↑</option>
    <option value="cyc">по началу цикла</option><option value="div">по расхождению с AR</option></select>
    <label>acc для</label><select id="accm"></select></div>
  <div class="list" id="list"></div>
  <div class="note"><kbd>j</kbd>/<kbd>k</kbd> — следующий/предыдущий пример, <kbd>←</kbd>/<kbd>→</kbd> — сдвиг окна, клик или протяжка по ленте — переход.</div>
</aside>
<main>
  <div class="card"><h2 id="title">—</h2><div class="note" id="sub"></div>
    <div class="ctl" id="mchips"></div>
    <canvas id="tl"></canvas>
    <div class="leg"><span><span class="sw" id="lg-heat"></span>acceptance раунда (светлее — ниже)</span><span><span class="sw" style="background:var(--cyc)"></span>цикл: от начала первого повтора до конца последнего</span>
      <span><span class="sw" style="background:var(--teos)"></span>EOS — top-1 у target (AR)</span><span><span class="sw" style="background:var(--fg);width:3px"></span>конец рассуждения (&lt;/think&gt;)</span><span><span class="sw" style="background:var(--diff)"></span>первый токен, отличный от AR</span><span><span class="sw" style="border:2px solid var(--fg)"></span>окно текста</span></div>
    <div id="period" style="height:150px"></div>
  </div>
  <div class="card">
    <div class="ctl"><label>Позиция в ответе</label><input type="number" id="pos" min="0" style="width:100px"><span class="note" id="ctx"></span>
      <button id="back">← окно</button><button id="fwd">окно →</button>
      <label>окно</label><select id="w"><option>256</option><option selected>1024</option><option>4096</option></select></div>
    <div class="ctl" id="jumps"></div>
    <div class="ctl" id="cycnav"></div>
    <div class="ctl"><label>Текст</label><select id="pm"></select><label>рядом</label><select id="cm"></select>
      <label><input type="checkbox" id="unfold"> развернуть цикл по периоду</label><label>период</label><input type="number" id="per" min="1" style="width:80px" placeholder="авто"></div>
    <div class="leg"><span><span class="sw" style="background:var(--trej)"></span>драфт ошибся: этот токен поставил target, раунд оборвался (только у драфтеров; остальное принято)</span>
      <span><span class="sw" style="box-shadow:inset 0 -2px 0 var(--teos)"></span>EOS top-1</span><span><span class="sw" style="outline:2px dashed var(--diff)"></span>расхождение с AR</span><span><span class="sw" style="background:linear-gradient(90deg,var(--cyc) 50%,var(--cyc2) 50%)"></span>цикл: подчёркивание чередует цвет на каждом повторе фрагмента; ▶ начало / конец ◀</span>
      <span id="lg-unf" style="display:none"><span class="sw" style="background:color-mix(in srgb,var(--diff) 30%,transparent)"></span>токен отличается от того же места предыдущего периода</span></div>
    <div class="cols" id="cols"></div><div class="note" id="err"></div>
  </div>
</main></div><div id="tip"></div>
<script>
const META = __META__;
const ORDER=["dflash2","dspark_radixark","dspark_redhat","eagle4","mtp","ar"];
const $=id=>document.getElementById(id);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLOR=m=>m==="ar"?css("--ar"):css("--s"+(ORDER.indexOf(m)+1));
const LABEL=m=>m==="ar"?"AR (без драфтера)":(META[m]||{label:m}).label;
const SHORTL=m=>m==="ar"?"AR":(META[m]||{label:m}).label.replace(/ \(.*\)/,"")+(m==="dspark_radixark"?" RA":m==="dspark_redhat"?" RH":"");
const K=v=>v==null?"—":v>=1000?(v/1000).toFixed(v>=10000?0:1)+"K":String(v);
const esc=t=>t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const LOC=[["__all","любой цикл"],["think","цикл в рассуждении"],["boundary","цикл с конца рассуждения"],["answer","цикл в ответе"],["none","без цикла"]];
const LOCT={think:"цикл начался внутри рассуждения (до </think> или </think> так и не появился)",boundary:"повторяется блок «конец рассуждения + </think> + ответ»: цикл стартует за ≤8 токенов до </think>",answer:"цикл начался после </think>",none:"цикл не найден"};
const S={index:null,panel:null,ds:"__all",loc:"__all",id:null,doc:null,prep:{},vis:new Set(ORDER),pm:"mtp",cm:"ar",pos:0,w:1024,cache:{}};

// data comes as script files (EXINDEX / EXDATA calls) so the page also works when opened as a local file
const PENDING={};
window.EXINDEX=d=>{ (PENDING.__index||(()=>{}))(d); };
window.EXDATA=(key,b64)=>{ (PENDING[key]||(()=>{}))(b64); };
function script(src){ return new Promise((ok,fail)=>{ const s=document.createElement("script"); s.src=src;
  s.onload=()=>{ s.remove(); ok(); }; s.onerror=()=>{ s.remove(); fail(new Error("не найден "+src)); }; document.head.appendChild(s); }); }
async function loadIndex(){ const p=new Promise(r=>PENDING.__index=r); await script("examples/index.js"); return p; }
async function loadDoc(panel,id){ const key=panel+"/"+id, p=new Promise(r=>PENDING[key]=r);
  await script("examples/"+panel+"/"+encodeURIComponent(id)+".js"); const b64=await p; delete PENDING[key];
  const bin=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));
  return new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"))).json(); }
function banner(msg){ const b=$("banner"); b.style.display=msg?"block":"none"; b.textContent=msg||""; }
function hashSave(){ history.replaceState(null,"","#"+[S.panel,S.id,S.pm,S.cm,S.pos,S.w].map(v=>encodeURIComponent(v??"")).join("/")); }

// ---------- sidebar ----------
const locOf=r=>{ const x=r.m[$("accm").value]||r.m.ar||[]; return x[6]; };
function rows(){ const all=S.index[S.panel].filter(r=>S.ds==="__all"||r.dataset===S.ds); return S.loc==="__all"?all:all.filter(r=>locOf(r)===S.loc); }
function drawLocs(){ const all=S.index[S.panel].filter(r=>S.ds==="__all"||r.dataset===S.ds), c={};
  all.forEach(r=>{ const l=locOf(r); c[l]=(c[l]||0)+1; });
  $("locs").innerHTML=LOC.filter(([v])=>v==="__all"||c[v]).map(([v,t])=>`<button data-v="${v}" class="${S.loc===v?"on":""}" title="${LOCT[v]||""}">${t} <span class="note">${v==="__all"?all.length:c[v]}</span></button>`).join("");
  $("locs").querySelectorAll("button").forEach(b=>b.onclick=()=>{ S.loc=b.dataset.v; drawLocs(); drawList(); const r=sortRows(rows())[0]; if(r) load(r.id); }); }
function sortRows(rs){ const k=$("sort").value, m=$("accm").value, big=1e12;
  const f={p:r=>r.p, n:r=>(r.m[m]||r.m.ar)[0], acc:r=>(r.m[m]||[])[1]??big, cyc:r=>(r.m[m]||[])[2]??big, div:r=>(r.m[m]||[])[4]??big}[k];
  return rs.slice().sort((a,b)=>f(a)-f(b)||a.id.localeCompare(b.id)); }
function drawChips(){ const all=S.index[S.panel], c={}; all.forEach(r=>c[r.dataset]=(c[r.dataset]||0)+1);
  $("dss").innerHTML=[["__all","все",all.length]].concat(Object.keys(c).sort().map(d=>[d,d,c[d]])).map(([v,t,n])=>`<button data-v="${v}" class="${S.ds===v?"on":""}">${t} <span class="note">${n}</span></button>`).join("");
  $("dss").querySelectorAll("button").forEach(b=>b.onclick=()=>{S.ds=b.dataset.v; drawChips(); drawLocs(); drawList(); const r=sortRows(rows())[0]; if(r) load(r.id);}); }
function drawList(){ const m=$("accm").value;
  $("list").innerHTML=sortRows(rows()).map(r=>{ const x=r.m[m]||r.m.ar||[]; const acc=x[1];
    const cyc=x[2]!=null?`<span class="badge c" title="первый цикл начинается с позиции ${x[2]}, повтор каждые ${x[3]} ток.; ${LOCT[x[6]]||""}">цикл ${K(x[2])} · ${({think:"в рассуждении",boundary:"с конца рассужд.",answer:"в ответе"})[x[6]]||""}</span>`:"";
    const dv=m!=="ar"&&x[4]!=null?`<span class="badge d" title="первый токен, отличный от AR">≠AR ${K(x[4])}</span>`:"";
    const cut=S.panel==="natural"&&r.max_new&&x[0]>=r.max_new?`<span class="badge" title="упёрлось в лимит генерации">лимит</span>`:"";
    return `<div class="row ${r.id===S.id?"cur":""}" data-id="${r.id}"><div class="t">${r.id}</div><div class="m"><span>${r.dataset}</span><span>вход ${K(r.p)}</span><span>ответ ${K(x[0])}</span>
      ${acc!=null?`<span><span class="bar"><i style="width:${(100*acc).toFixed(0)}%"></i></span> ${(100*acc).toFixed(0)}%</span>`:""}${cyc}${dv}${cut}</div></div>`; }).join("");
  $("list").querySelectorAll(".row").forEach(e=>e.onclick=()=>load(e.dataset.id));
  const cur=$("list").querySelector(".row.cur"); cur&&cur.scrollIntoView({block:"nearest"}); }

// ---------- per-request prep ----------
function prep(m){ const e=S.doc.methods[m]; if(!e) return null; if(S.prep[m]) return S.prep[m];
  const n=e.l.length, off=new Int32Array(n+1); for(let i=0;i<n;i++) off[i+1]=off[i]+e.l[i];
  const role=new Uint8Array(n), info=new Int8Array(n).fill(-1);
  if(e.g) for(let j=0;j<e.g.length;j++){ const g=e.g[j], a=e.a[j];
    for(let t=0;t<a&&g+t<n;t++){ role[g+t]=1; info[g+t]=a; } if(g+a<n){ role[g+a]=a<7?2:0; info[g+a]=a; } }
  const cycAt=new Int16Array(Math.ceil(n/1024)+1); (e.cyc||[]).forEach(([t,p])=>{ cycAt[Math.floor((t-1)/1024)]=p; }); // period of the 1K block ending at t
  const seg=e.seg||[], segP=new Int16Array(n+1), segK=new Int16Array(n+1); seg.forEach(([a,b,p],k)=>{ segP.fill(p,a,Math.min(n,b)); segK.fill(k+1,a,Math.min(n,b)); });
  const eos=new Set(e.eos||[]);
  const piece=i=>e.t.slice(off[i],off[i+1]);
  return S.prep[m]={e,n,off,role,info,cycAt,seg,segP,segK,eos,piece}; }
function periodAt(P,pos){ if(P.e.seg) return P.segP[Math.min(pos,P.n)]||0;
  const b=Math.floor(pos/1024); for(let i=b;i<Math.min(P.cycAt.length,b+3);i++) if(P.cycAt[i]) return P.cycAt[i]; return 0; }

async function load(id,keep){ S.id=id; S.prep={}; $("err").textContent="";
  const key=S.panel+"/"+id;
  try{ S.doc=S.cache[key]||(S.cache[key]=await loadDoc(S.panel,id)); banner(""); }
  catch(e){ banner("Не удалось загрузить пример: "+e.message); return; }
  const ms=ORDER.filter(m=>S.doc.methods[m]);
  if(!ms.includes(S.pm)) S.pm=ms.find(m=>m!=="ar")||ms[0]; if(S.cm&&!ms.includes(S.cm)) S.cm="";
  const fill=(sel,opts,v)=>{ sel.innerHTML=opts.map(([k,t])=>`<option value="${k}">${t}</option>`).join(""); sel.value=v; };
  fill($("pm"),ms.map(m=>[m,LABEL(m)]),S.pm); fill($("cm"),[["","—"]].concat(ms.map(m=>[m,LABEL(m)])),S.cm);
  $("mchips").innerHTML='<span class="note">лента:</span>'+ms.map(m=>`<button data-m="${m}" class="${S.vis.has(m)?"on":""}" style="${S.vis.has(m)?`background:${COLOR(m)};border-color:${COLOR(m)};color:#fff`:""}">${SHORTL(m)}</button>`).join("");
  $("mchips").querySelectorAll("button").forEach(b=>b.onclick=()=>{ const m=b.dataset.m; S.vis.has(m)?S.vis.delete(m):S.vis.add(m); load(S.id,true); });
  const d=S.doc, r=S.index[S.panel].find(r=>r.id===id)||{};
  $("title").textContent=id; $("sub").textContent=`${d.dataset} · вход ${d.prompt_tokens} токенов · ответы: `+ms.map(m=>`${SHORTL(m)} ${K(d.methods[m].l.length)}`).join(", ")+(S.panel==="natural"&&r.max_new?` · лимит ${K(r.max_new)}`:"");
  if(!keep) S.pos=Math.min(S.pos, maxN()); drawList(); drawJumps(); drawTimeline(); drawPeriod(); drawText(); hashSave(); }
const maxN=()=>Math.max(...Object.values(S.doc.methods).map(e=>e.l.length));

// ---------- timeline ----------
const TL={rowH:20,gap:6,left:62,top:4,axis:22};
function tlMethods(){ return ORDER.filter(m=>S.doc&&S.doc.methods[m]&&S.vis.has(m)); }
function mix(a,b,t){ const pa=a.match(/\w\w/g).map(h=>parseInt(h,16)), pb=b.match(/\w\w/g).map(h=>parseInt(h,16));
  return "rgb("+pa.map((v,i)=>Math.round(v+(pb[i]-v)*t)).join(",")+")"; }
function drawTimeline(){ const cv=$("tl"), ms=tlMethods(), dpr=window.devicePixelRatio||1;
  const W=cv.clientWidth||cv.parentNode.clientWidth, H=TL.top+ms.length*(TL.rowH+TL.gap)+TL.axis;
  cv.width=W*dpr; cv.height=H*dpr; cv.style.height=H+"px"; const g=cv.getContext("2d"); g.scale(dpr,dpr); g.clearRect(0,0,W,H);
  const N=maxN(), pw=W-TL.left-8, X=t=>TL.left+t/N*pw, h0=css("--h0").replace("#",""), h1=css("--h1").replace("#","");
  g.font="11px system-ui"; g.textBaseline="middle";
  ms.forEach((m,k)=>{ const P=prep(m), y=TL.top+k*(TL.rowH+TL.gap);
    g.fillStyle=css("--fg2"); g.fillText(SHORTL(m),4,y+TL.rowH/2);
    g.fillStyle=css("--grid"); g.fillRect(X(0),y,X(P.n)-X(0),TL.rowH);
    if(P.e.g){ const bins=new Float32Array(Math.ceil(pw)), cnt=new Float32Array(Math.ceil(pw));
      P.e.g.forEach((gg,j)=>{ const b=Math.min(bins.length-1,Math.floor(gg/N*pw)); bins[b]+=P.e.a[j]; cnt[b]+=7; });
      let last=-1; for(let b=0;b<bins.length;b++){ if(cnt[b]){ last=bins[b]/cnt[b]; } if(last<0||b/pw*N>P.n) continue;
        g.fillStyle=mix(h0,h1,Math.max(0,Math.min(1,last))); g.fillRect(TL.left+b,y+5,1.02,TL.rowH-5); } }
    else { g.fillStyle=css("--muted"); g.globalAlpha=.25; g.fillRect(X(0),y+5,X(P.n)-X(0),TL.rowH-5); g.globalAlpha=1; }
    g.fillStyle=css("--cyc");
    if(P.e.seg) P.seg.forEach(([a,b])=>{ g.fillRect(X(a),y,Math.max(1,X(b)-X(a)),4); g.fillRect(X(a),y,1.5,TL.rowH); g.fillRect(X(b)-1.5,y,1.5,TL.rowH); });
    else (P.e.cyc||[]).forEach(([t])=>g.fillRect(X(t-1024),y,Math.max(1,X(t)-X(t-1024)),4));
    if(P.e.eos){ g.fillStyle=css("--teos"); g.globalAlpha=.7; P.e.eos.forEach(t=>g.fillRect(X(t),y+TL.rowH/2,1,TL.rowH/2)); g.globalAlpha=1; }
    if(P.e.think!=null){ g.fillStyle=css("--fg"); g.fillRect(X(P.e.think)-1,y-2,2,TL.rowH+4); }
    if(P.e.div!=null){ const x=X(P.e.div); g.fillStyle=css("--diff"); g.beginPath(); g.moveTo(x-5,y-1); g.lineTo(x+5,y-1); g.lineTo(x,y+7); g.fill(); } });
  // window + axis
  const yb=TL.top+ms.length*(TL.rowH+TL.gap); const half=S.w/2;
  g.strokeStyle=css("--fg"); g.lineWidth=1.5; g.strokeRect(X(Math.max(0,S.pos-half)),1,Math.max(2,X(S.pos+half)-X(Math.max(0,S.pos-half))),yb-TL.gap+2);
  g.fillStyle=css("--muted"); g.textBaseline="top"; const step=[1000,2000,4000,8000,16000,32000].find(s=>N/s<=10)||64000;
  for(let t=0;t<=N;t+=step){ const x=X(t); g.fillRect(x,yb-TL.gap+3,1,4); g.fillText(K(t),x-8,yb-TL.gap+8); }
  g.fillText("позиция",4,yb-TL.gap+8); }
function tlAt(ev){ const cv=$("tl"), r=cv.getBoundingClientRect(), x=ev.clientX-r.left, y=ev.clientY-r.top, N=maxN(), pw=r.width-TL.left-8;
  const t=Math.round(Math.max(0,Math.min(N,(x-TL.left)/pw*N))); const k=Math.floor((y-TL.top)/(TL.rowH+TL.gap)); return {t, m:tlMethods()[k]}; }
function tlTip(ev){ if(!S.doc) return; const {t,m}=tlAt(ev), tip=$("tip"); if(!m){ tip.style.display="none"; return; }
  const P=prep(m); let s=`<b>${LABEL(m)}</b><br>позиция ${t} · контекст ${S.doc.prompt_tokens+t}`;
  if(P.e.g){ let lo=0,hi=P.e.g.length; while(lo<hi){const mid=(lo+hi)>>1; if(P.e.g[mid]<=t) lo=mid+1; else hi=mid;}
    const a=P.e.a.slice(Math.max(0,lo-16),lo+16); if(a.length) s+=`<br>acceptance вокруг: ${(100*a.reduce((x,y)=>x+y,0)/(7*a.length)).toFixed(0)}%`; }
  const p=periodAt(P,t); if(p){ const sg=P.seg.find(([a,b])=>a<=t&&t<b); s+=`<br>цикл, период ${p}`+(sg?` · [${sg[0]}, ${sg[1]})`:""); } if(t>=P.n) s+="<br>(ответ закончился)";
  tip.innerHTML=s; tip.style.display="block"; tip.style.left=(ev.clientX+14)+"px"; tip.style.top=(ev.clientY+12)+"px"; }
let drag=false;
$("tl").addEventListener("pointerdown",ev=>{ drag=true; $("tl").setPointerCapture(ev.pointerId); setPos(tlAt(ev).t,true); });
$("tl").addEventListener("pointermove",ev=>{ tlTip(ev); if(drag) setPos(tlAt(ev).t,true); });
$("tl").addEventListener("pointerup",()=>{ drag=false; drawText(); hashSave(); });
$("tl").addEventListener("pointerleave",()=>$("tip").style.display="none");

// ---------- period chart ----------
function drawPeriod(){ const ms=tlMethods(), tr=[];
  ms.forEach(m=>{ const c=S.doc.methods[m].cyc||[]; if(!c.length) return;
    tr.push({x:c.map(v=>v[0]),y:c.map(v=>v[1]),mode:"markers",name:SHORTL(m),marker:{size:5,color:COLOR(m)},hovertemplate:"позиция %{x}: период %{y}<extra>"+LABEL(m)+"</extra>"}); });
  const el=$("period"); el.style.display=tr.length?"block":"none"; if(!tr.length){ Plotly.purge(el); return; }
  Plotly.react(el,tr,{paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:css("--fg"),size:11},margin:{l:62,r:8,t:6,b:26},
    showlegend:false,hovermode:"closest",hoverlabel:{bgcolor:css("--card"),bordercolor:css("--grid"),font:{color:css("--fg")}},
    xaxis:{range:[0,maxN()],gridcolor:css("--grid"),zeroline:false},yaxis:{type:"log",title:{text:"период",font:{size:11}},gridcolor:css("--grid"),zeroline:false},
    shapes:[{type:"line",x0:S.pos,x1:S.pos,yref:"paper",y0:0,y1:1,line:{color:css("--fg"),width:1,dash:"dot"}}]},{displaylogo:false,responsive:true,displayModeBar:false});
  el.removeAllListeners&&el.removeAllListeners("plotly_click"); el.on("plotly_click",ev=>setPos(Math.round(ev.points[0].x))); }

// ---------- jumps ----------
function drawJumps(){ const P=prep(S.pm), ar=prep("ar"), J=[["начало",0]];
  const eos=(ar&&ar.e.eos&&ar.e.eos[0]); if(eos!=null) J.push(["первый EOS top-1",eos]);
  if(P&&P.e.think!=null) J.push(["конец рассуждения (&lt;/think&gt;)",P.e.think]);
  if(P&&P.seg.length){ const [a,b,p]=P.seg[0]; J.push([`начало цикла`,a],[`конец цикла`,Math.max(a,b-1)]); }
  else if(P&&P.e.cyc&&P.e.cyc.length){ const [t,p]=P.e.cyc[0]; J.push([`начало цикла (p=${p})`,Math.max(0,t-(p>128?8192:2048))]); }
  if(P&&P.e.div!=null) J.push(["расхождение с AR",P.e.div]);
  if(P&&P.e.g&&P.e.g.length>64){ let best=1e9,bi=0,s=0; const w=32; for(let i=0;i<P.e.a.length;i++){ s+=P.e.a[i]; if(i>=w) s-=P.e.a[i-w]; if(i>=w&&s<best){best=s;bi=i;} }
    J.push([`худший участок (${(100*best/(7*w)).toFixed(0)}%)`,P.e.g[Math.max(0,bi-w/2)]]); }
  if(P) J.push(["конец",Math.max(0,P.n-1)]);
  $("jumps").innerHTML='<span class="note">перейти:</span>'+J.map(([t,v])=>`<button data-v="${v}">${t}</button>`).join("");
  $("jumps").querySelectorAll("button").forEach(b=>b.onclick=()=>setPos(+b.dataset.v)); drawCycNav(); }
function drawCycNav(){ const P=prep(S.pm), el=$("cycnav"); if(!P||!P.seg.length){ el.innerHTML=""; return; }
  el.innerHTML=`<span class="note">циклы (${LABEL(S.pm)}): ${P.seg.length}</span><select id="cycsel">`+P.seg.map(([a,b,p],k)=>
    `<option value="${k}">${k+1}. [${a}, ${b}) · фрагмент ${p} ток. × ${((b-a)/p).toFixed(1)} повт.</option>`).join("")+
    `</select><button id="cyc-a">▶ начало</button><button id="cyc-b">конец ◀</button><button id="cyc-n">следующий цикл →</button>`;
  const sel=$("cycsel"), cur=P.seg.findIndex(([a,b])=>a<=S.pos&&S.pos<b); if(cur>=0) sel.value=cur;
  $("cyc-a").onclick=()=>setPos(P.seg[+sel.value][0]); $("cyc-b").onclick=()=>setPos(P.seg[+sel.value][1]-1);
  $("cyc-n").onclick=()=>{ const k=P.seg.findIndex(([a])=>a>S.pos); if(k>=0){ sel.value=k; setPos(P.seg[k][0]); } }; sel.onchange=()=>setPos(P.seg[+sel.value][0]); }
function setPos(t,light){ S.pos=Math.max(0,Math.min(maxN(),Math.round(t))); $("pos").value=S.pos;
  $("ctx").textContent=`контекст ${S.doc.prompt_tokens+S.pos}`; drawTimeline();
  if(!light){ drawText(); drawPeriodLine(); hashSave(); } }
function drawPeriodLine(){ const el=$("period"); if(el.layout&&el.style.display!=="none") Plotly.relayout(el,{"shapes[0].x0":S.pos,"shapes[0].x1":S.pos}); }

// ---------- text ----------
function column(m){ const P=prep(m); if(!P) return ""; const d=S.doc, half=S.w/2, g0=Math.max(0,S.pos-half), g1=Math.min(P.n,S.pos+half);
  const head=`<div class="ch"><b style="color:${COLOR(m)}">■</b> ${LABEL(m)} · ответ ${K(P.n)}${P.e.div!=null?` · ≠AR с ${P.e.div}`:""}${periodAt(P,S.pos)?` · <span style="color:var(--cyc)">в цикле, повтор каждые ${periodAt(P,S.pos)} ток.</span>`:""}</div>`;
  if(S.pos>P.n) return `<div class="col">${head}<div class="tx"><span class="gap">[ответ закончился на позиции ${P.n}]</span></div></div>`;
  if($("unfold").checked) return `<div class="col">${head}${unfold(m,P)}</div>`;
  let h="";
  if(g0===0){ h+=`<span class="gap">[промпт: ${d.prompt_tokens} токенов${d.prompt_skipped?`; показаны первые 512 и последние 2048`:""}]</span>\n<span class="pr">${esc(d.prompt_skipped?d.prompt_head+"\n…\n"+d.prompt_tail:d.prompt_head+d.prompt_tail)}</span>\n<span class="gap">[— ответ —]</span>\n`; }
  else h+=`<button class="more" data-d="-1">↑ раньше (выше ещё ${g0} ток.)</button>`;
  const sa=new Map(), sb=new Map(); P.seg.forEach(([a,b,p],k)=>{ sa.set(a,[k,p,b]); sb.set(b,[k,p,a]); });
  for(let i=g0;i<g1;i++){ if(sb.has(i)){ const [k,p,a]=sb.get(i); h+=`<span class="cb" data-v="${a}" title="конец цикла ${k+1}: позиции [${a}, ${i}), фрагмент из ${p} ток. повторяется ${((i-a)/p).toFixed(1)} раз; клик — к началу">конец цикла ${k+1} ◀</span>`; }
    if(sa.has(i)){ const [k,p,b]=sa.get(i); h+=`<span class="cb" data-v="${b-1}" title="цикл ${k+1}: позиции [${i}, ${b}), фрагмент из ${p} ток. повторяется ${((b-i)/p).toFixed(1)} раз; клик — к концу">▶ цикл ${k+1}: повтор каждые ${p} ток.</span>`; }
    const sk=P.segK[i], r=sk?Math.floor((i-P.seg[sk-1][0])/P.seg[sk-1][2]):-1;
    const c=(P.role[i]===1?"a":P.role[i]===2?"r":"")+(P.eos.has(i)?" e":"")+(i===S.pos?" cur":"")+(i===P.e.div?" dv":"")+(sk?(r%2?" c1":" c0"):"");
    const t=esc(P.piece(i)); h+=`<span${c?` class="${c.trim()}"`:""} data-i="${i}"${sk?` title="цикл ${sk}, повтор ${r+1}"`:""}>${t}</span>`; }
  if(sb.has(g1)&&g1<=P.n){ const [k,p,a]=sb.get(g1); h+=`<span class="cb" data-v="${a}" title="конец цикла ${k+1}">конец цикла ${k+1} ◀</span>`; }
  if(g1<P.n) h+=`<button class="more" data-d="1">↓ дальше (ниже ещё ${P.n-g1} ток.)</button>`;
  return `<div class="col">${head}<div class="tx">${h}</div></div>`; }
function unfold(m,P){ let p=+$("per").value||periodAt(P,S.pos);
  if(!p) return `<div class="tx"><span class="gap">[в этой позиции цикл не обнаружен; задайте период вручную]</span></div>`;
  const rowsN=Math.max(8,Math.min(40,Math.floor(4*S.w/p))), start=Math.max(0,S.pos-Math.floor(rowsN/2)*p);
  let h="<table>"; for(let r=0;r<rowsN;r++){ const a=start+r*p; if(a>=P.n) break; h+=`<tr><td class="lab">${a}</td><td>`;
    for(let i=a;i<Math.min(P.n,a+p);i++){ const t=P.piece(i), diff=i-p>=0&&P.piece(i-p)!==t;
      const c=(P.role[i]===1?"a":P.role[i]===2?"r":"")+(diff&&r>0?" x":"")+(i===S.pos?" cur":"");
      h+=`<span class="${c.trim()}" data-i="${i}" title="позиция ${i} · фаза ${(i-start)%p}">${esc(t).replace(/\n/g,'<span class="nl">⏎</span>')}</span>`; }
    h+="</td></tr>"; }
  return `<div class="unf tx">${h}</table></div>`; }
function drawText(){ if(!S.doc) return; S.pm=$("pm").value; S.cm=$("cm").value;
  const ms=[S.pm].concat(S.cm&&S.cm!==S.pm?[S.cm]:[]); $("cols").innerHTML=ms.map(column).join("");
  $("lg-unf").style.display=$("unfold").checked?"":"none";
  $("cols").querySelectorAll(".tx").forEach(b=>{ const c=b.querySelector(".cur"); if(c) b.scrollTop=c.offsetTop-b.clientHeight/2; });
  $("cols").querySelectorAll("[data-i]").forEach(s=>s.ondblclick=()=>setPos(+s.dataset.i));
  $("cols").querySelectorAll(".cb").forEach(s=>s.onclick=()=>setPos(+s.dataset.v));
  $("cols").querySelectorAll(".more").forEach(b=>b.onclick=()=>setPos(S.pos+(+b.dataset.d)*S.w)); }

// ---------- wiring ----------
function setPanel(p,id){ S.panel=p; S.ds="__all"; S.loc="__all";
  $("panels").innerHTML=Object.keys(S.index).map(k=>`<button data-p="${k}" class="${k===p?"on":""}">${{natural:"свой текст · EOS разрешён",noeos:"свой текст · EOS запрещён",forced:"текст AR (teacher forcing) · EOS запрещён"}[k]||k}</button>`).join("");
  $("panels").querySelectorAll("button").forEach(b=>b.onclick=()=>setPanel(b.dataset.p));
  const r=id&&S.index[p].find(r=>r.id===id); if(r) S.ds=r.dataset; drawChips(); drawLocs(); drawList(); load(r?r.id:sortRows(rows())[0].id, !!r); }
$("sort").onchange=drawList; $("accm").onchange=()=>{ drawLocs(); drawList(); };
$("pos").onchange=()=>setPos(+$("pos").value); $("w").onchange=()=>{S.w=+$("w").value; setPos(S.pos);};
$("back").onclick=()=>setPos(S.pos-S.w/2); $("fwd").onclick=()=>setPos(S.pos+S.w/2);
$("pm").onchange=()=>{ S.pm=$("pm").value; drawJumps(); drawText(); hashSave(); }; $("cm").onchange=()=>{ drawText(); hashSave(); };
$("unfold").onchange=drawText; $("per").onchange=drawText;
document.addEventListener("keydown",ev=>{ if(/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
  if(ev.key==="ArrowRight") setPos(S.pos+S.w/2); else if(ev.key==="ArrowLeft") setPos(S.pos-S.w/2);
  else if(ev.key==="j"||ev.key==="k"){ const rs=sortRows(rows()), i=rs.findIndex(r=>r.id===S.id), n=rs[i+(ev.key==="j"?1:-1)]; if(n) load(n.id); } });
window.addEventListener("resize",()=>S.doc&&drawTimeline());
function setTheme(t){ document.documentElement.dataset.theme=t; try{localStorage.setItem("theme",t);}catch(e){}
  $("theme").textContent=t==="dark"?"☀ светлая":"☾ тёмная"; if(S.doc){ drawTimeline(); drawPeriod(); load(S.id,true); } }
$("theme").onclick=()=>setTheme(document.documentElement.dataset.theme==="dark"?"light":"dark");
setTheme(document.documentElement.dataset.theme||"light");
$("lg-heat").style.background=`linear-gradient(90deg,${css("--h0")},${css("--h1")})`;
(async()=>{ try{ S.index=await loadIndex(); }
  catch(e){ banner("Нет данных примеров ("+e.message+"). Их строит examples.py; папка examples/ должна лежать рядом с этой страницей."); return; }
  $("accm").innerHTML=ORDER.filter(m=>m!=="ar").map(m=>`<option value="${m}">${SHORTL(m)}</option>`).join(""); $("accm").value="mtp";
  const h=location.hash.slice(1).split("/").map(decodeURIComponent);
  if(h[2]) S.pm=h[2]; if(h[3]!==undefined) S.cm=h[3]; if(h[4]) S.pos=+h[4]; if(h[5]){ S.w=+h[5]; $("w").value=h[5]; }
  $("pos").value=S.pos; setPanel(S.index[h[0]]?h[0]:Object.keys(S.index)[0], h[1]); })();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/examples.html")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(HTML.replace("__META__", json.dumps(DRAFTERS)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
