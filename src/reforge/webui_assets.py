"""Static single-page app served by :mod:`reforge.webui`.

Kept as a Python string (rather than a package data file) so ``pip install``
and ``python -m reforge webui`` work identically without packaging rules or
filesystem lookups. The page is dependency-free: system fonts, inline SVG,
vanilla JS, and a canvas chart.

The palette is a 2026 take on the forge: near-black neutral surfaces with
warm ember highlights, violet/cyan data accents, and translucent "liquid
glass" panels inspired by recent macOS and developer-tool releases.
"""

INDEX_HTML = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark light">
<title>Reforge</title>
<script>
(function () {
  try {
    var choice = new URLSearchParams(location.search).get("theme")
      || localStorage.getItem("reforge.theme") || "auto";
    var resolved = choice === "auto"
      ? (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark")
      : choice;
    document.documentElement.dataset.theme = resolved;
    document.documentElement.style.colorScheme = resolved;
  } catch (e) {}
})();
</script>
<style>
  :root{
    --font:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",
      "Segoe UI Variable","Segoe UI",system-ui,sans-serif;
    --mono:"SF Mono",ui-monospace,"Cascadia Code","Cascadia Mono",Consolas,monospace;
    --bg:#09090b; --bg-2:#0d0d10;
    --surface:rgba(24,24,27,.72); --surface-solid:#18181b;
    --surface-2:rgba(39,39,42,.62); --surface-3:rgba(63,63,70,.35);
    --line:rgba(255,255,255,.085); --line-strong:rgba(255,255,255,.16);
    --ink:#f4f4f5; --ink-2:#d4d4d8; --muted:#a1a1aa; --faint:#71717a;
    --accent:#ff7a45; --accent-2:#ffb86b; --accent-ink:#1c0d06;
    --violet:#a78bfa; --cyan:#67e8f9; --ok:#30d158; --warn:#ffd60a; --bad:#ff453a;
    --shadow:0 24px 70px rgba(0,0,0,.42),0 2px 8px rgba(0,0,0,.28);
    --shadow-sm:0 6px 18px rgba(0,0,0,.22);
    --inner:inset 0 1px 0 rgba(255,255,255,.06);
    --blur:saturate(180%) blur(24px);
    --r:16px; --r-sm:11px; --r-xs:8px;
    --ease:cubic-bezier(.2,.8,.2,1);
    --topbar:rgba(9,9,11,.66);
  }
  [data-theme="light"]{
    --bg:#f4f3f0; --bg-2:#eceae6;
    --surface:rgba(255,255,255,.74); --surface-solid:#fff;
    --surface-2:rgba(244,244,245,.86); --surface-3:rgba(228,228,231,.7);
    --line:rgba(24,24,27,.09); --line-strong:rgba(24,24,27,.17);
    --ink:#18181b; --ink-2:#3f3f46; --muted:#71717a; --faint:#a1a1aa;
    --accent:#e25822; --accent-2:#f59e0b; --accent-ink:#fff;
    --violet:#7c3aed; --cyan:#0891b2; --ok:#16a34a; --warn:#ca8a04; --bad:#dc2626;
    --shadow:0 24px 70px rgba(24,24,27,.13),0 2px 8px rgba(24,24,27,.06);
    --shadow-sm:0 6px 18px rgba(24,24,27,.08);
    --inner:inset 0 1px 0 rgba(255,255,255,.75);
    --topbar:rgba(244,243,240,.72);
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{
    margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 var(--font);
    -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility;
    min-height:100vh;
  }
  body.modal-open{overflow:hidden}
  .ambient{position:fixed;inset:0;overflow:hidden;pointer-events:none;z-index:0}
  .ambient i{position:absolute;display:block;border-radius:50%;filter:blur(110px);opacity:.2}
  .ambient .a1{width:52vw;height:52vw;left:-14vw;top:-24vw;
    background:radial-gradient(circle,#ff5c2a 0%,rgba(255,92,42,0) 68%)}
  .ambient .a2{width:46vw;height:46vw;right:-16vw;bottom:-20vw;
    background:radial-gradient(circle,#7c3aed 0%,rgba(124,58,237,0) 68%);opacity:.14}
  .ambient .a3{width:34vw;height:34vw;left:52%;top:42%;
    background:radial-gradient(circle,#22d3ee 0%,rgba(34,211,238,0) 70%);opacity:.07}
  [data-theme="light"] .ambient i{opacity:.12}
  .app{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column}
  .main{width:100%;max-width:1220px;margin:0 auto;padding:28px 24px 72px;flex:1}
  .topbar{
    position:sticky;top:0;z-index:60;display:flex;align-items:center;gap:18px;
    padding:9px 22px;background:var(--topbar);backdrop-filter:var(--blur);
    -webkit-backdrop-filter:var(--blur);border-bottom:1px solid var(--line);
  }
  .brand{display:flex;align-items:center;gap:9px;flex:none}
  .brand-mark{display:grid;place-items:center;width:30px;height:30px;
    border-radius:9px;background:linear-gradient(145deg,#2a160e,#160d09);
    border:1px solid rgba(255,180,120,.22);box-shadow:var(--inner),var(--shadow-sm)}
  [data-theme="light"] .brand-mark{background:linear-gradient(145deg,#fff1e6,#ffe0c7);
    border-color:rgba(226,88,34,.2)}
  .brand-name{font-size:15px;font-weight:650;letter-spacing:-.02em}
  .brand-sub{font-size:11px;color:var(--faint);letter-spacing:.04em;
    text-transform:uppercase;margin-left:1px}
  .menubar{display:flex;align-items:center;gap:1px;flex:1;min-width:0}
  .menu{position:relative}
  .menu-trigger{height:28px;padding:0 10px;border:0;border-radius:7px;
    background:transparent;color:var(--ink-2);font:inherit;font-size:13px;
    cursor:pointer;transition:background .14s var(--ease),color .14s var(--ease)}
  .menu-trigger:hover,.menu.open .menu-trigger{background:var(--surface-2);color:var(--ink)}
  .menu-pop{position:absolute;top:calc(100% + 8px);left:0;min-width:250px;
    padding:6px;border:1px solid var(--line-strong);border-radius:13px;
    background:var(--surface);backdrop-filter:var(--blur);
    -webkit-backdrop-filter:var(--blur);box-shadow:var(--shadow);
    opacity:0;transform:translateY(-5px) scale(.985);pointer-events:none;
    transition:opacity .14s var(--ease),transform .14s var(--ease);z-index:70}
  .menu.open .menu-pop{opacity:1;transform:none;pointer-events:auto}
  .menu-item{display:flex;align-items:center;gap:9px;width:100%;height:30px;
    padding:0 9px;border:0;border-radius:7px;background:transparent;
    color:var(--ink);font:inherit;font-size:13px;text-align:left;cursor:pointer;
    white-space:nowrap}
  .menu-item:hover{background:var(--accent);color:var(--accent-ink)}
  .menu-item:hover .kbd{color:inherit;opacity:.7}
  .menu-ico{width:16px;display:inline-flex;justify-content:center;opacity:.75}
  .menu-check{width:14px;text-align:center;opacity:0;font-size:11px}
  .menu-item.checked .menu-check{opacity:1}
  .menu-label{padding:5px 9px 3px;font-size:10.5px;color:var(--faint);
    text-transform:uppercase;letter-spacing:.08em}
  .menu-sep{height:1px;background:var(--line);margin:5px 7px}
  .kbd{margin-left:auto;color:var(--faint);font-size:11px;font-variant-numeric:tabular-nums}
  .top-actions{display:flex;align-items:center;gap:8px;flex:none}
  .status-pill{display:inline-flex;align-items:center;gap:7px;height:28px;
    padding:0 11px;border-radius:999px;border:1px solid var(--line);
    background:var(--surface);font-size:12px;color:var(--muted);white-space:nowrap}
  .status-pill .dot{width:7px;height:7px;border-radius:50%;background:var(--faint)}
  .status-pill.running .dot{background:var(--accent);box-shadow:0 0 0 4px rgba(255,122,69,.16);
    animation:pulse 1.6s infinite}
  .status-pill.stopping .dot{background:var(--warn);animation:pulse 1.2s infinite}
  .status-pill.done .dot{background:var(--ok)}
  .status-pill.error .dot{background:var(--bad)}
  .status-pill.done{color:var(--ok);border-color:rgba(48,209,88,.24)}
  .status-pill.error{color:var(--bad);border-color:rgba(255,69,58,.24)}
  @keyframes pulse{50%{opacity:.35}}
  @keyframes spin{to{transform:rotate(360deg)}}
  .icon-btn{display:inline-grid;place-items:center;width:32px;height:32px;
    border:1px solid transparent;border-radius:8px;background:transparent;
    color:var(--muted);cursor:pointer;transition:background .14s var(--ease),
    color .14s var(--ease),border-color .14s var(--ease)}
  .icon-btn:hover{background:var(--surface-2);color:var(--ink);border-color:var(--line)}
  .icon-btn.tiny{width:26px;height:26px;font-size:12px}
  .icon-btn.danger:hover{color:var(--bad);border-color:rgba(255,69,58,.3)}
  .btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;
    height:34px;padding:0 14px;border-radius:9px;border:1px solid var(--line);
    background:var(--surface-2);color:var(--ink);font:inherit;font-size:13px;
    font-weight:550;cursor:pointer;white-space:nowrap;
    transition:transform .12s var(--ease),background .14s var(--ease),
    border-color .14s var(--ease),box-shadow .14s var(--ease)}
  .btn:hover{background:var(--surface-3);border-color:var(--line-strong)}
  .btn:active{transform:scale(.985)}
  .btn:disabled{opacity:.42;cursor:not-allowed;transform:none}
  .btn.primary{background:linear-gradient(180deg,var(--accent-2),var(--accent));
    border-color:transparent;color:var(--accent-ink);
    box-shadow:0 8px 26px rgba(255,122,69,.24),var(--inner)}
  .btn.primary:hover{filter:brightness(1.05);background:linear-gradient(180deg,var(--accent-2),var(--accent))}
  .btn.ghost{background:transparent}
  .btn.ghost:hover{background:var(--surface-2)}
  .btn.small{height:28px;padding:0 10px;font-size:12px;border-radius:8px}
  .btn.wide{width:100%;height:42px;font-size:13.5px;border-radius:10px}
  .btn.danger{color:var(--bad);border-color:rgba(255,69,58,.26)}
  .btn.danger:hover{background:rgba(255,69,58,.1);border-color:rgba(255,69,58,.4)}
  .spinner{width:14px;height:14px;border:2px solid currentColor;
    border-right-color:transparent;border-radius:50%;animation:spin .7s linear infinite}
  .row{display:flex;align-items:center;gap:8px}
  .card{position:relative;background:var(--surface);border:1px solid var(--line);
    border-radius:var(--r);box-shadow:var(--shadow-sm),var(--inner);
    backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);padding:19px 20px}
  .card + .card{margin-top:14px}
  .card-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px}
  .card-head h2{margin:0;font-size:13px;font-weight:600;letter-spacing:-.01em;color:var(--ink-2)}
  .card-note{font-size:11.5px;color:var(--faint)}
  .eyebrow{font-size:10.5px;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--accent)}
  .hero{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;
    margin:8px 0 26px;flex-wrap:wrap}
  .hero h1{margin:6px 0 4px;font-size:34px;line-height:1.1;font-weight:680;
    letter-spacing:-.035em}
  .hero p{margin:0;color:var(--muted);font-size:14.5px;max-width:620px}
  .hero-tags{display:flex;gap:7px;flex-wrap:wrap;padding-bottom:4px}
  .tag{display:inline-flex;align-items:center;height:25px;padding:0 10px;
    border:1px solid var(--line);border-radius:999px;background:var(--surface);
    color:var(--muted);font-size:11.5px}
  .setup-grid{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px;align-items:start}
  .setup-main{min-width:0}
  .setup-side{position:sticky;top:76px}
  .field-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  .field-grid.four{grid-template-columns:repeat(4,minmax(0,1fr))}
  .field{display:flex;flex-direction:column;gap:6px;min-width:0}
  .field > label{font-size:11.5px;font-weight:550;color:var(--muted)}
  .field input,.field select{width:100%;height:37px;padding:0 11px;border:1px solid var(--line);
    border-radius:9px;background:var(--surface-2);color:var(--ink);font:inherit;
    font-size:13px;transition:border-color .14s var(--ease),box-shadow .14s var(--ease)}
  .field input::placeholder{color:var(--faint)}
  .field input:focus,.field select:focus{outline:none;border-color:rgba(255,122,69,.6);
    box-shadow:0 0 0 3px rgba(255,122,69,.12)}
  .input-wrap{display:flex;align-items:center;gap:6px;height:37px;padding:0 4px 0 0;
    border:1px solid var(--line);border-radius:9px;background:var(--surface-2);
    transition:border-color .14s var(--ease),box-shadow .14s var(--ease)}
  .input-wrap:focus-within{border-color:rgba(255,122,69,.6);
    box-shadow:0 0 0 3px rgba(255,122,69,.12)}
  .input-wrap input{height:100%;border:0;background:transparent;box-shadow:none!important;padding:0 10px}
  .input-wrap input:focus{box-shadow:none}
  .browse-btn{height:27px;padding:0 9px;border:1px solid var(--line);border-radius:7px;
    background:var(--surface);color:var(--muted);font:inherit;font-size:11.5px;
    cursor:pointer;flex:none;transition:color .14s var(--ease),border-color .14s var(--ease)}
  .browse-btn:hover{color:var(--accent);border-color:rgba(255,122,69,.45)}
  .field-hint{font-size:11px;color:var(--faint);line-height:1.4}
  .link-btn{padding:0;border:0;background:none;color:var(--accent);font:inherit;
    font-size:11px;cursor:pointer}
  .link-btn:hover{text-decoration:underline}
  .toggle-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 18px;margin-top:16px}
  .toggle{display:flex;align-items:center;gap:11px;cursor:pointer;user-select:none}
  .toggle > span:last-child{display:flex;flex-direction:column;min-width:0}
  .toggle b{font-size:12.5px;font-weight:550;color:var(--ink)}
  .toggle small{font-size:11px;color:var(--faint)}
  .switch{position:relative;width:38px;height:22px;flex:none;display:inline-block}
  .switch input{position:absolute;inset:0;opacity:0;cursor:pointer;margin:0}
  .switch .track{position:absolute;inset:0;border-radius:999px;background:var(--surface-3);
    border:1px solid var(--line);transition:background .18s var(--ease),border-color .18s var(--ease)}
  .switch .track::after{content:"";position:absolute;width:16px;height:16px;left:2px;top:2px;
    border-radius:50%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.3);
    transition:transform .18s var(--ease)}
  .switch input:checked + .track{background:var(--accent);border-color:transparent}
  .switch input:checked + .track::after{transform:translateX(16px)}
  .switch input:focus-visible + .track{box-shadow:0 0 0 3px rgba(255,122,69,.2)}
  .dpo-field{margin-top:16px;max-width:260px}
  .hint-block{margin-top:14px;padding:10px 12px;border-radius:9px;
    background:rgba(255,184,107,.07);border:1px solid rgba(255,184,107,.14);
    color:var(--muted);font-size:11.5px;line-height:1.5}
  .segmented{display:inline-flex;padding:3px;gap:2px;border-radius:9px;
    background:var(--surface-2);border:1px solid var(--line)}
  .segmented button{height:25px;padding:0 10px;border:0;border-radius:6px;
    background:transparent;color:var(--muted);font:inherit;font-size:11.5px;
    font-weight:550;cursor:pointer;transition:background .14s var(--ease),color .14s var(--ease)}
  .segmented button.active{background:var(--surface-solid);color:var(--ink);
    box-shadow:0 1px 4px rgba(0,0,0,.22)}
  .section-copy{margin:-4px 0 14px;color:var(--muted);font-size:12.5px}
  .stage-row{display:grid;grid-template-columns:26px minmax(0,1fr) 74px 74px auto;
    gap:9px;align-items:end;padding:9px;border:1px solid var(--line);
    border-radius:11px;background:var(--surface-2);margin-bottom:8px}
  .stage-num{display:grid;place-items:center;width:24px;height:24px;margin-bottom:6px;
    border-radius:50%;background:var(--surface-3);color:var(--muted);
    font:600 11px var(--mono)}
  .stage-fields{display:flex;flex-direction:column;gap:5px;min-width:0}
  .stage-fields input,.stage-mini input{width:100%;height:32px;padding:0 8px;
    border:1px solid var(--line);border-radius:7px;background:var(--surface);
    color:var(--ink);font:inherit;font-size:12px}
  .stage-fields input:focus,.stage-mini input:focus{outline:none;
    border-color:rgba(255,122,69,.6)}
  .stage-fields .input-wrap{height:32px}
  .stage-fields .input-wrap input{height:30px}
  .stage-mini{display:flex;flex-direction:column;gap:5px}
  .stage-mini label{font-size:10px;color:var(--faint)}
  .stage-actions{display:flex;gap:3px;padding-bottom:3px}
  .batch-header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;flex-wrap:wrap}
  .batch-mode-toggle{display:flex;align-items:center;gap:8px}
  .batch-resume{display:flex;flex-direction:column;gap:9px;padding:12px 14px;margin-bottom:12px;
    border:1px solid rgba(255,214,10,.3);border-radius:11px;background:rgba(255,214,10,.07)}
  .batch-resume-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
  .batch-resume-title{display:flex;align-items:center;gap:8px;font-size:12.5px;font-weight:600;color:var(--ink)}
  .batch-resume-sub{font-size:11px;color:var(--muted)}
  .batch-resume-path{font:10.5px var(--mono);color:var(--faint);word-break:break-all}
  .batch-resume-stages{display:flex;gap:4px;flex-wrap:wrap}
  .batch-resume-stages i{display:inline-flex;align-items:center;justify-content:center;width:22px;height:18px;
    border-radius:5px;font:600 9.5px var(--mono);border:1px solid var(--line);
    background:var(--surface-2);color:var(--faint);font-style:normal}
  .batch-resume-stages i.done{background:rgba(48,209,88,.14);border-color:rgba(48,209,88,.3);color:var(--ok)}
  .batch-resume-stages i.active{background:rgba(255,122,69,.14);border-color:rgba(255,122,69,.35);color:var(--accent)}
  .batch-resume-stages i.failed{background:rgba(255,69,58,.14);border-color:rgba(255,69,58,.3);color:var(--bad)}
  .batch-resume-actions{display:flex;gap:8px;flex-wrap:wrap}
  .empty-inline{padding:14px;border:1px dashed var(--line);border-radius:9px;
    color:var(--faint);font-size:12px;text-align:center}
  .disclosure{padding:0;overflow:hidden}
  .disclosure summary{display:flex;align-items:center;justify-content:space-between;
    gap:12px;padding:16px 20px;cursor:pointer;list-style:none;font-size:13px;
    font-weight:600;color:var(--ink-2);user-select:none}
  .disclosure summary::-webkit-details-marker{display:none}
  .disclosure summary::before{content:"›";display:inline-block;margin-right:8px;
    color:var(--faint);transition:transform .18s var(--ease)}
  .disclosure[open] summary::before{transform:rotate(90deg)}
  .disclosure summary > span:first-of-type{margin-right:auto}
  .summary-hint{font-size:11px;font-weight:400;color:var(--faint)}
  .details-body{padding:0 20px 18px;border-top:1px solid var(--line)}
  .preview-head{display:flex;align-items:center;justify-content:space-between;
    padding:12px 0 8px}
  .mono-label{font:11px var(--mono);color:var(--faint)}
  pre{margin:0;padding:13px 14px;border:1px solid var(--line);border-radius:9px;
    background:rgba(0,0,0,.2);color:var(--muted);font:11.5px/1.6 var(--mono);
    white-space:pre-wrap;word-break:break-word;max-height:280px;overflow:auto}
  [data-theme="light"] pre{background:rgba(24,24,27,.035)}
  .warn-text{color:var(--warn)}
  .run-card{position:relative;overflow:hidden}
  .run-card::before{content:"";position:absolute;inset:-40% 30% auto -20%;height:160px;
    background:radial-gradient(ellipse,rgba(255,122,69,.16),transparent 70%);
    pointer-events:none}
  .run-summary{display:flex;flex-direction:column;gap:1px;margin:2px 0 14px}
  .run-row{display:flex;align-items:center;justify-content:space-between;gap:14px;
    padding:7px 0;border-bottom:1px solid var(--line)}
  .run-row:last-child{border-bottom:0}
  .run-row span{font-size:11.5px;color:var(--faint);flex:none}
  .run-row b{font-size:12px;font-weight:500;color:var(--ink-2);min-width:0;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right}
  .run-status{display:flex;align-items:flex-start;gap:7px;padding:9px 10px;
    border-radius:8px;font-size:11.5px;line-height:1.45;margin-bottom:12px}
  .run-status .dot{width:6px;height:6px;border-radius:50%;margin-top:5px;flex:none}
  .run-status.ok{background:rgba(48,209,88,.08);color:var(--ok)}
  .run-status.ok .dot{background:var(--ok)}
  .run-status.bad{background:rgba(255,69,58,.08);color:var(--bad)}
  .run-status.bad .dot{background:var(--bad)}
  .run-hint{margin-top:11px;font-size:10.5px;line-height:1.5;color:var(--faint)}
  .tips-card ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:11px}
  .tips-card li{position:relative;padding-left:17px;font-size:11.5px;line-height:1.55;color:var(--muted)}
  .tips-card li::before{content:"";position:absolute;left:2px;top:6px;width:5px;height:5px;
    border-radius:50%;background:var(--accent);opacity:.7}
  .tips-card b{color:var(--ink-2);font-weight:550}
  .badge{display:inline-flex;align-items:center;height:21px;padding:0 8px;
    border-radius:999px;border:1px solid var(--line);background:var(--surface-2);
    color:var(--muted);font-size:10.5px;font-weight:550;white-space:nowrap}
  .badge.ok{color:var(--ok);border-color:rgba(48,209,88,.24);background:rgba(48,209,88,.08)}
  .badge.warn{color:var(--warn);border-color:rgba(255,214,10,.24);background:rgba(255,214,10,.08)}
  .badge.bad{color:var(--bad);border-color:rgba(255,69,58,.24);background:rgba(255,69,58,.08)}
  .badge.violet{color:var(--violet);border-color:rgba(167,139,250,.24);background:rgba(167,139,250,.08)}
  .banner{display:flex;align-items:flex-start;gap:12px;padding:12px 14px;
    border:1px solid rgba(255,69,58,.28);background:rgba(255,69,58,.08);
    border-radius:10px;margin-bottom:16px;color:var(--ink-2);font-size:12.5px;
    white-space:pre-wrap}
  .banner.warn{border-color:rgba(255,214,10,.28);background:rgba(255,214,10,.07)}
  .banner .banner-x{margin-left:auto;border:0;background:transparent;color:var(--muted);
    font-size:17px;line-height:1;cursor:pointer;padding:0 2px}
  .footer{display:flex;justify-content:space-between;gap:16px;padding:16px 24px;
    border-top:1px solid var(--line);color:var(--faint);font-size:11px}
  [hidden]{display:none!important}
  .mono{font-family:var(--mono)}
  @media (max-width:980px){
    .setup-grid{grid-template-columns:1fr}
    .setup-side{position:static}
    .field-grid.four{grid-template-columns:repeat(2,minmax(0,1fr))}
    .brand-sub{display:none}
  }
  @media (max-width:720px){
    .main{padding:20px 15px 60px}
    .topbar{padding:8px 14px;gap:10px}
    .menubar{overflow-x:auto;scrollbar-width:none}
    .menubar::-webkit-scrollbar{display:none}
    .hero h1{font-size:27px}
    .field-grid,.field-grid.four,.toggle-grid{grid-template-columns:1fr}
    .stage-row{grid-template-columns:24px minmax(0,1fr);align-items:start}
    .stage-mini,.stage-actions{grid-column:2}
    .stage-mini{display:grid;grid-template-columns:1fr 1fr;gap:6px}
    .stage-mini label{display:none}
    .status-pill{display:none}
    .footer{flex-direction:column}
  }
  @media (prefers-reduced-motion:reduce){
    *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;
      transition-duration:.01ms!important;scroll-behavior:auto!important}
  }
  /* ── progress ───────────────────────────────────────────────────── */
  .view{animation:viewIn .26s var(--ease)}
  @keyframes viewIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
  .progress-head{padding:20px 22px 18px;overflow:hidden}
  .progress-head::before{content:"";position:absolute;inset:-60% 40% auto -10%;height:220px;
    background:radial-gradient(ellipse,rgba(255,122,69,.14),transparent 70%);pointer-events:none}
  .phases{display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin-bottom:17px}
  .stage-chip{display:inline-flex;align-items:center;gap:9px;margin:-8px 0 15px;padding:7px 12px;
    border:1px solid rgba(167,139,250,.3);border-radius:9px;background:rgba(167,139,250,.08);
    font-size:12px;color:var(--ink-2)}
  .stage-chip b{font-weight:600;color:var(--ink)}
  .stage-chip .mono{font-size:11px;color:var(--violet)}
  .stage-chip .resumed{font-size:10px;color:var(--warn);border:1px solid rgba(255,214,10,.3);
    border-radius:999px;padding:1px 7px}
  .stage-chip .stage-ds{font:11px var(--mono);color:var(--muted);max-width:280px;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .phase{display:inline-flex;align-items:center;gap:7px;height:27px;padding:0 10px;
    border:1px solid var(--line);border-radius:999px;color:var(--faint);
    font-size:11px;font-weight:550;letter-spacing:.01em;transition:all .18s var(--ease)}
  .phase-dot{display:inline-grid;place-items:center;width:12px;height:12px;flex:none}
  .phase.active{border-color:rgba(255,122,69,.42);color:var(--accent);
    background:rgba(255,122,69,.07)}
  .phase.done{border-color:rgba(48,209,88,.24);color:var(--ok)}
  .phase.failed{border-color:rgba(255,69,58,.28);color:var(--bad)}
  .phase .spinner{width:10px;height:10px;border-width:1.5px}
  .phase-arrow{display:inline-flex;color:var(--faint);opacity:.5}
  .progress-main{display:flex;align-items:flex-end;justify-content:space-between;gap:20px}
  .progress-title{font-size:23px;font-weight:640;letter-spacing:-.025em;margin:3px 0 5px}
  .tqdm{font:12px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
  .pct-big{font-size:42px;line-height:1;font-weight:660;letter-spacing:-.04em;
    font-variant-numeric:tabular-nums;background:linear-gradient(180deg,var(--accent-2),var(--accent));
    -webkit-background-clip:text;background-clip:text;color:transparent;padding-bottom:2px}
  .bar{position:relative;height:9px;border-radius:999px;background:var(--surface-3);
    margin:18px 0 10px;overflow:hidden}
  .bar i{position:absolute;inset:0 auto 0 0;width:0;border-radius:999px;
    background:linear-gradient(90deg,var(--accent),var(--accent-2));
    box-shadow:0 0 16px rgba(255,122,69,.35);transition:width .55s var(--ease)}
  .bar.indet i{width:22%;animation:sweep 1.2s ease-in-out infinite}
  @keyframes sweep{0%{left:-22%}100%{left:100%}}
  .progress-foot{display:flex;gap:18px;flex-wrap:wrap;color:var(--faint);
    font:11px var(--mono);font-variant-numeric:tabular-nums}
  .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));
    gap:9px;margin:14px 0}
  .metric{display:flex;flex-direction:column;gap:5px;padding:11px 13px;
    border:1px solid var(--line);border-radius:11px;background:var(--surface);
    box-shadow:var(--inner);min-width:0}
  .metric-label{font-size:10px;color:var(--faint);letter-spacing:.06em;
    text-transform:uppercase;white-space:nowrap}
  .metric-value{font:16px/1.2 var(--mono);font-variant-numeric:tabular-nums;
    color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .metric-value small{font-size:10px;color:var(--faint);font-weight:400}
  .metric.accent .metric-value{color:var(--accent)}
  .metric.ok .metric-value{color:var(--ok)}
  .metric.violet .metric-value{color:var(--violet)}
  .metric.cyan .metric-value{color:var(--cyan)}
  .progress-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(290px,1fr);
    gap:14px;align-items:start;margin-top:14px}
  .progress-side{display:flex;flex-direction:column;gap:14px;min-width:0}
  .progress-side .card + .card{margin-top:0}
  .chart-card{min-width:0}
  .chart-wrap{position:relative;height:286px;border:1px solid var(--line);
    border-radius:11px;background:rgba(0,0,0,.14);overflow:hidden}
  [data-theme="light"] .chart-wrap{background:rgba(255,255,255,.38)}
  canvas#chart{display:block;width:100%;height:100%}
  .chart-empty{position:absolute;inset:0;display:grid;place-items:center;
    color:var(--faint);font-size:12px;pointer-events:none}
  .chart-tip{position:absolute;z-index:3;min-width:170px;padding:9px 10px;
    border:1px solid var(--line-strong);border-radius:9px;
    background:var(--surface-solid);box-shadow:var(--shadow);pointer-events:none;
    font:11px var(--mono);color:var(--muted)}
  .tip-step{font-weight:600;color:var(--ink);margin-bottom:5px}
  .tip-row{display:flex;justify-content:space-between;gap:18px;padding:1px 0}
  .tip-row b{color:var(--ink-2);font-weight:500}
  .analysis{display:flex;gap:7px;flex-wrap:wrap;margin-top:11px}
  .analysis-item{display:flex;flex-direction:column;gap:1px;padding:7px 10px;
    border:1px solid var(--line);border-radius:8px;background:var(--surface-2);min-width:82px}
  .analysis-item span{font-size:9.5px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}
  .analysis-item b{font:12px var(--mono);color:var(--ink-2);font-weight:500}
  .analysis-item.grade-a b{color:var(--ok)} .analysis-item.grade-b b{color:var(--accent)}
  .analysis-item.grade-c b{color:var(--warn)} .analysis-item.grade-d b{color:var(--bad)}
  .alog{margin-top:9px;font:11px/1.7 var(--mono);color:var(--faint);white-space:pre-wrap}
  .health-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
  .health-grid > div{display:flex;flex-direction:column;gap:2px;padding:8px 10px;
    border:1px solid var(--line);border-radius:8px;background:var(--surface-2)}
  .health-grid span{font-size:9.5px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}
  .health-grid b{font:13px var(--mono);font-weight:500;color:var(--ink-2)}
  .patience{height:5px;border-radius:999px;background:var(--surface-3);margin:12px 0 9px;overflow:hidden}
  .patience i{display:block;height:100%;width:0;border-radius:999px;background:var(--ok);
    transition:width .4s var(--ease),background .2s var(--ease)}
  .patience i.warn{background:var(--warn)}
  .patience i.bad{background:var(--bad)}
  .hint{margin:0;color:var(--faint);font-size:11px;line-height:1.55}
  .ckpt-body{font-size:11.5px;color:var(--faint);line-height:1.5;margin-bottom:10px}
  .ckpt-step{font:600 13px var(--mono);color:var(--ink-2);margin-bottom:3px}
  .ckpt-body .path{font:10.5px/1.45 var(--mono);color:var(--faint);word-break:break-all;
    max-height:44px;overflow:hidden}
  .kv-grid{display:flex;flex-direction:column;gap:1px}
  .kv-grid > div{display:flex;justify-content:space-between;gap:12px;padding:6px 0;
    border-bottom:1px solid var(--line);font-size:11.5px}
  .kv-grid > div:last-child{border-bottom:0}
  .kv-grid span{color:var(--faint)}
  .kv-grid b{font:11.5px var(--mono);font-weight:500;color:var(--ink-2);text-align:right;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .timeline{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:2px}
  .timeline li{display:flex;gap:9px;padding:6px 0;border-bottom:1px solid var(--line)}
  .timeline li:last-child{border-bottom:0}
  .sig-dot{width:6px;height:6px;border-radius:50%;margin-top:5px;flex:none;background:var(--faint)}
  .signal.ok .sig-dot{background:var(--ok)} .signal.warn .sig-dot{background:var(--warn)}
  .signal.bad .sig-dot{background:var(--bad)} .signal.info .sig-dot{background:var(--accent)}
  .signal > div{display:flex;flex-direction:column;gap:1px;min-width:0}
  .sig-text{font-size:11px;color:var(--muted);line-height:1.4;word-break:break-word}
  .sig-time{font:9.5px var(--mono);color:var(--faint)}
  .logpane pre{max-height:330px;min-height:70px}
  .check{display:inline-flex;align-items:center;gap:6px;color:var(--muted);cursor:pointer}
  .check input{accent-color:var(--accent);width:14px;height:14px}
  .check.small{font-size:11px}
  .collapsed pre{display:none}
  /* ── results ────────────────────────────────────────────────────── */
  .result-hero{display:flex;align-items:center;gap:18px;padding:24px;overflow:hidden}
  .result-hero::before{content:"";position:absolute;inset:-70% 55% auto -10%;height:260px;
    background:radial-gradient(ellipse,rgba(48,209,88,.16),transparent 70%);pointer-events:none}
  .result-hero.failed::before{background:radial-gradient(ellipse,rgba(255,69,58,.15),transparent 70%)}
  .result-icon{display:grid;place-items:center;width:52px;height:52px;border-radius:50%;
    background:rgba(48,209,88,.12);border:1px solid rgba(48,209,88,.24);color:var(--ok);flex:none}
  .result-hero.failed .result-icon{background:rgba(255,69,58,.1);
    border-color:rgba(255,69,58,.24);color:var(--bad)}
  .result-hero h2{margin:2px 0 3px;font-size:24px;letter-spacing:-.03em;font-weight:640}
  .result-hero p{margin:0;color:var(--muted);font-size:12.5px;max-width:720px}
  .result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:11px;margin-top:14px}
  .result-card{display:flex;flex-direction:column;gap:4px;padding:14px 15px;
    border:1px solid var(--line);border-radius:12px;background:var(--surface);
    box-shadow:var(--inner);min-width:0}
  .result-card span{font-size:10px;color:var(--faint);text-transform:uppercase;letter-spacing:.06em}
  .result-card b{font:13px var(--mono);font-weight:500;color:var(--ink-2);
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .result-card.path b{white-space:normal;word-break:break-all;font-size:11px;line-height:1.5}
  .result-card .btn{margin-top:7px;align-self:flex-start}
  .result-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
  .report-body .kv-grid{margin-top:4px}
  details.raw{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
  details.raw summary{cursor:pointer;color:var(--faint);font-size:11px}
  details.raw pre{margin-top:9px}
  /* ── modals ─────────────────────────────────────────────────────── */
  .modal{position:fixed;inset:0;z-index:100;display:grid;place-items:center;padding:24px}
  .modal-backdrop{position:absolute;inset:0;background:rgba(0,0,0,.46);
    backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
  [data-theme="light"] .modal-backdrop{background:rgba(24,24,27,.28)}
  .modal-panel{position:relative;width:min(860px,100%);max-height:min(760px,90vh);
    display:flex;flex-direction:column;border:1px solid var(--line-strong);
    border-radius:16px;background:var(--surface-solid);box-shadow:var(--shadow);
    animation:modalIn .2s var(--ease);overflow:hidden}
  @keyframes modalIn{from{opacity:0;transform:translateY(8px) scale(.99)}to{opacity:1;transform:none}}
  .modal-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;
    padding:17px 19px 14px;border-bottom:1px solid var(--line)}
  .modal-head h3{margin:0;font-size:16px;letter-spacing:-.02em;font-weight:620}
  .modal-head p{margin:3px 0 0;color:var(--faint);font-size:11.5px}
  .modal-foot{display:flex;align-items:center;justify-content:space-between;gap:14px;
    padding:13px 17px;border-top:1px solid var(--line);background:var(--surface-2)}
  .browser-panel{height:min(680px,88vh)}
  .browser-body{display:grid;grid-template-columns:190px minmax(0,1fr);min-height:0;flex:1}
  .browse-side{padding:10px;border-right:1px solid var(--line);overflow:auto;background:var(--surface-2)}
  .side-label{padding:8px 8px 4px;font-size:9.5px;color:var(--faint);
    text-transform:uppercase;letter-spacing:.08em}
  .place{display:flex;align-items:center;gap:8px;width:100%;height:30px;padding:0 8px;
    border:0;border-radius:7px;background:transparent;color:var(--muted);
    font:inherit;font-size:11.5px;text-align:left;cursor:pointer}
  .place:hover{background:var(--surface-3);color:var(--ink)}
  .place.active{background:rgba(255,122,69,.12);color:var(--accent)}
  .place svg{width:14px;height:14px;flex:none}
  .browser-main{display:flex;flex-direction:column;min-width:0;min-height:0}
  .browser-toolbar{display:flex;align-items:center;gap:7px;padding:11px 12px 8px}
  .path-input{flex:1;height:31px;min-width:0;padding:0 10px;border:1px solid var(--line);
    border-radius:7px;background:var(--surface-2);color:var(--ink);font:11.5px var(--mono)}
  .path-input:focus{outline:none;border-color:rgba(255,122,69,.6)}
  .crumbs{display:flex;align-items:center;gap:2px;padding:0 12px 8px;overflow-x:auto;
    scrollbar-width:none;white-space:nowrap}
  .crumbs::-webkit-scrollbar{display:none}
  .crumb{height:23px;padding:0 7px;border:0;border-radius:5px;background:transparent;
    color:var(--faint);font:11px var(--mono);cursor:pointer;flex:none}
  .crumb:hover{background:var(--surface-2);color:var(--ink-2)}
  .crumb:last-child{color:var(--ink-2)}
  .crumb-sep{color:var(--faint);opacity:.45;display:inline-flex}
  .browser-search{padding:0 12px 9px}
  .browser-search input{width:100%;height:31px;padding:0 10px;border:1px solid var(--line);
    border-radius:7px;background:var(--surface-2);color:var(--ink);font:12px var(--font)}
  .browser-search input:focus{outline:none;border-color:rgba(255,122,69,.6)}
  .browse-list{flex:1;min-height:0;overflow:auto;padding:0 8px 8px}
  .entry{display:flex;align-items:center;gap:9px;width:100%;height:36px;padding:0 9px;
    border:0;border-radius:7px;background:transparent;color:var(--ink-2);
    font:inherit;font-size:12px;text-align:left;cursor:default}
  .entry:hover{background:var(--surface-2)}
  .entry.selected{background:rgba(255,122,69,.13);color:var(--ink)}
  .entry-ico{display:inline-flex;color:var(--accent);flex:none}
  .entry.file .entry-ico{color:var(--faint)}
  .entry-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .entry-meta{font:10px var(--mono);color:var(--faint);flex:none}
  .mini-badge{display:inline-block;margin-left:6px;padding:1px 5px;border-radius:4px;
    background:rgba(167,139,250,.13);color:var(--violet);font-size:9px;
    vertical-align:1px}
  .browse-error{padding:18px;color:var(--bad);font-size:12px;line-height:1.5}
  .browse-empty{display:flex;flex-direction:column;align-items:center;gap:4px;
    padding:38px 16px;color:var(--faint);font-size:12px;text-align:center}
  .browse-new-row{display:flex;gap:6px;padding:8px 12px;border-top:1px solid var(--line);
    background:var(--surface-2)}
  .browse-new-row input{flex:1;height:31px;padding:0 10px;border:1px solid var(--line);
    border-radius:7px;background:var(--surface);color:var(--ink);font:12px var(--font)}
  .selected-path{font:10.5px/1.45 var(--mono);color:var(--faint);word-break:break-all;
    max-width:56%}
  .info-panel{width:min(560px,100%)}
  .info-body{padding:18px 20px 22px;overflow:auto}
  .info-body h4{margin:0 0 4px;font-size:12.5px;color:var(--ink-2)}
  .info-body p{margin:0 0 14px;color:var(--muted);font-size:12px;line-height:1.6}
  .shortcut{display:flex;justify-content:space-between;gap:20px;padding:7px 0;
    border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}
  .shortcut:last-child{border-bottom:0}
  .shortcut kbd{font:10.5px var(--mono);color:var(--ink-2);background:var(--surface-2);
    border:1px solid var(--line);border-radius:5px;padding:2px 6px}
  .confirm-panel{width:min(420px,100%)}
  .confirm-panel .modal-head{border-bottom:0;padding-bottom:4px}
  .confirm-panel .modal-head p{color:var(--muted);font-size:12.5px;line-height:1.6}
  .confirm-panel .modal-foot{background:transparent;border-top:0;justify-content:flex-end}
  .toast-root{position:fixed;right:20px;bottom:20px;z-index:200;display:flex;
    flex-direction:column;gap:8px;pointer-events:none}
  .toast{display:flex;align-items:center;gap:9px;max-width:360px;padding:10px 13px;
    border:1px solid var(--line-strong);border-radius:10px;background:var(--surface-solid);
    box-shadow:var(--shadow);color:var(--ink-2);font-size:12px;
    opacity:0;transform:translateY(8px);transition:opacity .18s var(--ease),transform .18s var(--ease)}
  .toast.in{opacity:1;transform:none}
  .toast-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);flex:none}
  .toast.ok .toast-dot{background:var(--ok)}
  .toast.warn .toast-dot{background:var(--warn)}
  .toast.bad .toast-dot{background:var(--bad)}
  @media (max-width:980px){
    .progress-grid{grid-template-columns:1fr}
    .progress-side{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))}
  }
  @media (max-width:720px){
    .progress-side{grid-template-columns:1fr}
    .browser-body{grid-template-columns:1fr}
    .browse-side{display:flex;gap:4px;overflow-x:auto;border-right:0;
      border-bottom:1px solid var(--line);padding:8px}
    .side-label{display:none}
    .place{width:auto;white-space:nowrap}
    .modal{padding:10px}
    .browser-panel{height:92vh}
    .selected-path{max-width:100%}
    .browser-foot{flex-direction:column;align-items:stretch}
    .browser-foot .row{justify-content:flex-end}
    .metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    .result-hero{align-items:flex-start}
    .toast-root{left:14px;right:14px;bottom:14px}
  }
</style>
</head>
<body>
<div class="ambient" aria-hidden="true"><i class="a1"></i><i class="a2"></i><i class="a3"></i></div>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          <path d="M12.2 2.8c2.8 4.1 5.1 6.5 5.1 9.7a5.3 5.3 0 0 1-10.6 0c0-3.2 2.3-5.6 5.1-9.7Z" fill="url(#flame)"/>
          <path d="M12.2 9.4c1.2 1.7 2.1 2.8 2.1 4.1a2.1 2.1 0 1 1-4.2 0c0-1.3.9-2.4 2.1-4.1Z" fill="#ffe0bd"/>
          <defs><linearGradient id="flame" x1="7" y1="3" x2="17" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#ffb86b"/><stop offset="1" stop-color="#ff5c2a"/>
          </linearGradient></defs>
        </svg>
      </span>
      <span class="brand-name">Reforge</span>
      <span class="brand-sub">LoRA studio</span>
    </div>

    <nav class="menubar" id="menubar" aria-label="Application menu">
      <div class="menu" data-menu="file">
        <button type="button" class="menu-trigger" aria-expanded="false">File</button>
        <div class="menu-pop" role="menu">
          <button type="button" class="menu-item" data-action="new-run" role="menuitem">New run<span class="kbd">⌘N</span></button>
          <button type="button" class="menu-item" data-action="browse-dataset" role="menuitem">Choose dataset…<span class="kbd">⌘O</span></button>
          <button type="button" class="menu-item" data-action="browse-base" role="menuitem">Choose base model…<span class="kbd">⌘⇧O</span></button>
          <button type="button" class="menu-item" data-action="browse-output" role="menuitem">Choose output folder…<span class="kbd">⌘⌥O</span></button>
          <div class="menu-sep"></div>
          <button type="button" class="menu-item" data-action="reveal-output" role="menuitem">Reveal output folder</button>
        </div>
      </div>
      <div class="menu" data-menu="run">
        <button type="button" class="menu-trigger" aria-expanded="false">Run</button>
        <div class="menu-pop" role="menu">
          <button type="button" class="menu-item" data-action="start" role="menuitem">Start training<span class="kbd">⌘↵</span></button>
          <button type="button" class="menu-item" data-action="stop" role="menuitem">Stop training<span class="kbd">⌘.</span></button>
          <div class="menu-sep"></div>
          <button type="button" class="menu-item" data-action="copy-command" role="menuitem">Copy command</button>
        </div>
      </div>
      <div class="menu" data-menu="view">
        <button type="button" class="menu-trigger" aria-expanded="false">View</button>
        <div class="menu-pop" role="menu">
          <button type="button" class="menu-item" data-action="toggle-console" role="menuitem">Toggle console<span class="kbd">⌘J</span></button>
          <div class="menu-sep"></div>
          <div class="menu-label">Appearance</div>
          <button type="button" class="menu-item" data-action="theme" data-theme-choice="auto" role="menuitem"><span class="menu-check">✓</span>Auto</button>
          <button type="button" class="menu-item" data-action="theme" data-theme-choice="light" role="menuitem"><span class="menu-check">✓</span>Light</button>
          <button type="button" class="menu-item" data-action="theme" data-theme-choice="dark" role="menuitem"><span class="menu-check">✓</span>Dark</button>
        </div>
      </div>
      <div class="menu" data-menu="help">
        <button type="button" class="menu-trigger" aria-expanded="false">Help</button>
        <div class="menu-pop" role="menu">
          <button type="button" class="menu-item" data-action="shortcuts" role="menuitem">Keyboard shortcuts</button>
          <button type="button" class="menu-item" data-action="about" role="menuitem">About Reforge</button>
        </div>
      </div>
    </nav>

    <div class="top-actions">
      <span class="status-pill idle" id="chip"><span class="dot"></span><span id="chip-text">Ready</span></span>
      <button type="button" class="icon-btn" id="btn-theme" title="Appearance" aria-label="Appearance"></button>
      <button type="button" class="btn small danger" id="btn-stop-top" hidden>Stop</button>
    </div>
  </header>

  <main class="main">
    <div id="msg-banner" class="banner" hidden></div>

    <section id="view-setup" class="view">
      <div class="hero">
        <div>
          <div class="eyebrow">LOCAL LORA STUDIO</div>
          <h1>Fine-tune a model.</h1>
          <p>Load a dataset, pick a base model, and let Reforge handle the rest.</p>
        </div>
        <div class="hero-tags">
          <span class="tag">Offline by default</span>
          <span class="tag">Agent-native</span>
        </div>
      </div>

      <div class="setup-grid">
        <div class="setup-main">
          <section class="card">
            <div class="card-head"><h2>Paths</h2><span class="card-note">Files stay on this machine</span></div>
            <div class="field-grid">
              <div class="field">
                <label for="p-base">Base model</label>
                <div class="input-wrap">
                  <input type="text" id="p-base" list="models" placeholder="Choose a local model folder…" autocomplete="off" spellcheck="false">
                  <button type="button" class="browse-btn" data-browse="base" title="Browse for a model folder">Browse…</button>
                </div>
                <datalist id="models"></datalist>
              </div>
              <div class="field">
                <label for="p-out">Output folder</label>
                <div class="input-wrap">
                  <input type="text" id="p-out" placeholder="Default: ~/Reforge_Output" autocomplete="off" spellcheck="false">
                  <button type="button" class="browse-btn" data-browse="out" title="Browse for an output folder">Browse…</button>
                </div>
                <div class="field-hint">Checkpoints, logs, and the merged model land here. <button type="button" class="link-btn" id="btn-use-default">Use default</button></div>
              </div>
            </div>
            <div class="field-grid" id="single-dataset-fields" style="margin-top:14px">
              <div class="field">
                <label for="p-ds">Dataset</label>
                <div class="input-wrap">
                  <input type="text" id="p-ds" placeholder="Choose a .parquet file…" autocomplete="off" spellcheck="false">
                  <button type="button" class="browse-btn" data-browse="dataset" title="Browse for a .parquet dataset">Browse…</button>
                </div>
              </div>
              <div class="field">
                <label for="p-hf">…or Hugging Face dataset ID</label>
                <div class="input-wrap">
                  <input type="text" id="p-hf" placeholder="org/dataset" autocomplete="off" spellcheck="false">
                </div>
              </div>
            </div>
          </section>

          <section class="card">
            <div class="card-head"><h2>Training</h2><span class="card-note">Overrides model-profile defaults</span></div>
            <div class="field-grid four">
              <div class="field"><label for="p-epoch">Epochs</label><input type="text" id="p-epoch" placeholder="1" inputmode="numeric"></div>
              <div class="field"><label for="p-steps">Fixed steps</label><input type="text" id="p-steps" placeholder="—" inputmode="numeric"><div class="field-hint">Overrides epochs</div></div>
              <div class="field"><label for="p-batch">Batch size</label><input type="text" id="p-batch" placeholder="1" inputmode="numeric"></div>
              <div class="field"><label for="p-grad">Grad accumulation</label><input type="text" id="p-grad" placeholder="Profile default" inputmode="numeric"></div>
            </div>
            <div class="toggle-grid">
              <label class="toggle"><span class="switch"><input type="checkbox" id="p-force"><span class="track"></span></span><span><b>Force full run</b><small>Disable EMA early stopping</small></span></label>
              <label class="toggle"><span class="switch"><input type="checkbox" id="p-lmhead"><span class="track"></span></span><span><b>Train lm_head</b><small>New-token / capacity runs</small></span></label>
            </div>
          </section>

          <section class="card">
            <div class="card-head"><h2>Data &amp; flags</h2></div>
            <div class="toggle-grid">
              <label class="toggle"><span class="switch"><input type="checkbox" id="p-latex"><span class="track"></span></span><span><b>Preserve LaTeX</b><small>Keep equations intact</small></span></label>
              <label class="toggle"><span class="switch"><input type="checkbox" id="p-code"><span class="track"></span></span><span><b>Preserve code</b><small>Keep indentation and formatting</small></span></label>
              <label class="toggle"><span class="switch"><input type="checkbox" id="p-image"><span class="track"></span></span><span><b>Multimodal</b><small>Image + text training</small></span></label>
              <label class="toggle"><span class="switch"><input type="checkbox" id="p-dpo"><span class="track"></span></span><span><b>DPO</b><small>Preference-pair tuning</small></span></label>
            </div>
            <div class="field dpo-field">
              <label for="p-dpo-beta">DPO beta</label>
              <input type="text" id="p-dpo-beta" placeholder="0.1" inputmode="decimal">
              <div class="field-hint">KL strength vs. the reference model. Higher stays closer to the SFT model.</div>
            </div>
            <div class="hint-block">SFT → DPO chains: leave the base model empty after an SFT merge and Reforge will use the merged model automatically.</div>
          </section>

          <section class="card">
            <div class="card-head">
              <h2>Run mode</h2>
              <div class="segmented" id="mode-tabs">
                <button type="button" id="mode-single" class="active">Single run</button>
                <button type="button" id="mode-batch">Batch pipeline</button>
              </div>
            </div>
            <p class="section-copy" id="mode-copy">Train one dataset with one merged output.</p>
            <div id="batch-editor" hidden>
              <div id="batch-resume" hidden></div>
              <div class="batch-header">
                <div class="batch-mode-toggle">
                  <button type="button" class="btn ghost small" id="btn-load-manifest">Load manifest.json</button>
                  <input type="file" id="manifest-file" accept=".json" hidden>
                </div>
                <p class="hint-block">Stage 1 uses the base model above; each later stage chains from the previous stage's merged output.</p>
              </div>
              <div id="stage-list"></div>
              <button type="button" class="btn ghost small" id="btn-addstage">+ Add stage</button>
            </div>
          </section>

          <details class="card disclosure" id="preview-card">
            <summary><span>Command preview</span><span class="summary-hint">What Reforge will launch</span></summary>
            <div class="details-body">
              <div class="preview-head"><span class="mono-label">python -m reforge</span><button type="button" class="btn ghost small" id="btn-copy-command">Copy</button></div>
              <pre id="cmd-preview">Fill in the fields to see the command…</pre>
            </div>
          </details>
        </div>

        <aside class="setup-side">
          <section class="card run-card">
            <div class="card-head"><h2>Run</h2><span class="badge neutral" id="run-mode">Single</span></div>
            <div class="run-summary">
              <div class="run-row"><span>Base</span><b id="run-base">—</b></div>
              <div class="run-row"><span>Dataset</span><b id="run-dataset">—</b></div>
              <div class="run-row"><span>Output</span><b id="run-output">—</b></div>
            </div>
            <div class="run-status ok" id="run-status"><span class="dot"></span>Ready to launch</div>
            <button type="button" class="btn primary wide" id="btn-start">Start training</button>
            <div class="run-hint">Runs locally. Checkpoints save every 25 steps; interrupted runs resume automatically.</div>
          </section>
          <section class="card tips-card">
            <div class="card-head"><h2>Good to know</h2></div>
            <ul class="tips">
              <li><b>Early stopping</b> watches the EMA loss, not single noisy steps.</li>
              <li><b>Checkpoints</b> are written every 25 steps and kept in the output folder.</li>
              <li><b>Batch pipeline</b> chains each stage's merged model into the next base.</li>
            </ul>
          </section>
        </aside>
      </div>
    </section>

    <section id="view-progress" class="view" hidden>
      <section class="card progress-head">
        <div class="phases" id="phases"></div>
        <div class="stage-chip" id="stage-chip" hidden></div>
        <div class="progress-main">
          <div>
            <div class="eyebrow" id="progress-eyebrow">LIVE RUN</div>
            <div class="progress-title" id="progress-title">Starting…</div>
            <div class="tqdm" id="tqdm">waiting for the child…</div>
          </div>
          <div class="pct-big" id="pct-big">—</div>
        </div>
        <div class="bar" id="bar"><i></i></div>
        <div class="progress-foot">
          <span id="progress-elapsed">elapsed —</span>
          <span id="progress-eta">eta —</span>
          <span id="progress-rate">— it/s</span>
        </div>
      </section>

      <div class="metric-grid" id="stats"></div>

      <div class="progress-grid">
        <section class="card chart-card">
          <div class="card-head">
            <h2>Training curve</h2>
            <div class="segmented" id="chart-tabs">
              <button type="button" data-metric="loss" class="active">Loss</button>
              <button type="button" data-metric="grad_norm">Grad norm</button>
              <button type="button" data-metric="lr">LR</button>
              <button type="button" data-metric="epoch">Epoch</button>
            </div>
          </div>
          <div class="chart-wrap" id="chart-wrap">
            <canvas id="chart"></canvas>
            <div class="chart-empty" id="chart-empty">Waiting for the first step…</div>
            <div class="chart-tip" id="chart-tip" hidden></div>
          </div>
          <div class="analysis" id="analysis"></div>
          <div class="alog" id="alog"></div>
        </section>

        <div class="progress-side">
          <section class="card health-card">
            <div class="card-head"><h2>Early stopping</h2><span class="badge neutral" id="es-verdict">waiting</span></div>
            <div class="health-grid">
              <div><span>Smoothed loss (EMA)</span><b id="es-ema">—</b></div>
              <div><span>Best smoothed</span><b id="es-best">—</b></div>
              <div><span>Steps since best</span><b id="es-since">—</b></div>
              <div><span>Patience</span><b id="es-patience">—</b></div>
            </div>
            <div class="patience"><i id="patience-fill"></i></div>
            <p class="hint" id="es-hint">Checks run every 25 steps. A single noisy loss spike is ignored — only the EMA verdict decides.</p>
          </section>
          <section class="card">
            <div class="card-head"><h2>Checkpoints</h2><span class="badge neutral" id="ckpt-count">0</span></div>
            <div class="ckpt-body" id="ckpt-body">Checkpoints are written every 25 steps.</div>
            <button type="button" class="btn ghost small" id="btn-reveal-ckpt" hidden>Show in folder</button>
          </section>
          <section class="card">
            <div class="card-head"><h2>Run plan</h2></div>
            <div class="kv-grid" id="plan-body"></div>
          </section>
          <section class="card">
            <div class="card-head"><h2>Recent activity</h2></div>
            <ul class="timeline" id="timeline"></ul>
          </section>
        </div>
      </div>

      <section class="card logpane" id="console-card">
        <div class="card-head">
          <h2>Console</h2>
          <div class="row">
            <button type="button" class="btn ghost small" id="btn-copy-log">Copy</button>
            <label class="check small"><input type="checkbox" id="log-follow" checked> Follow</label>
          </div>
        </div>
        <pre id="log"></pre>
      </section>
    </section>

    <section id="view-results" class="view" hidden>
      <section class="card result-hero" id="res-hero"></section>
      <div class="result-grid" id="res-body"></div>
      <details class="card disclosure" id="report-card" hidden>
        <summary><span>Training report</span><span class="summary-hint">training_report.json</span></summary>
        <div class="details-body" id="report-body"></div>
      </details>
      <section class="card logpane">
        <div class="card-head"><h2>Console tail</h2><button type="button" class="btn ghost small" id="btn-copy-res-log">Copy</button></div>
        <pre id="res-log"></pre>
      </section>
    </section>
  </main>

  <footer class="footer"><span>Reforge</span><span>Local LoRA fine-tuning · <span id="footer-state">ready</span></span></footer>
</div>

<div id="browser" class="modal" hidden>
  <div class="modal-backdrop" data-close="browser"></div>
  <div class="modal-panel browser-panel" role="dialog" aria-modal="true" aria-labelledby="browse-title">
    <header class="modal-head">
      <div><h3 id="browse-title">Choose a folder</h3><p id="browse-subtitle">Navigate to a folder and choose it</p></div>
      <button type="button" class="icon-btn" id="browse-close" aria-label="Close">×</button>
    </header>
    <div class="browser-body">
      <aside class="browse-side" id="browse-side"></aside>
      <div class="browser-main">
        <div class="browser-toolbar">
          <button type="button" class="icon-btn" id="browse-up" title="Up one level">↑</button>
          <input type="text" id="browse-path" class="path-input" spellcheck="false" aria-label="Folder path">
          <button type="button" class="btn ghost small" id="browse-go">Go</button>
          <label class="check small"><input type="checkbox" id="browse-hidden"> Hidden</label>
          <label class="check small" id="browse-all-wrap"><input type="checkbox" id="browse-all"> All files</label>
        </div>
        <div class="crumbs" id="browse-crumbs"></div>
        <div class="browser-search"><input type="search" id="browse-search" placeholder="Filter this folder…" spellcheck="false"></div>
        <div class="browse-list" id="browse-list"></div>
        <div class="browse-new-row" id="browse-new-row" hidden>
          <input type="text" id="browse-new-name" placeholder="New folder name" spellcheck="false">
          <button type="button" class="btn primary small" id="browse-new-create">Create</button>
          <button type="button" class="btn ghost small" id="browse-new-cancel">Cancel</button>
        </div>
      </div>
    </div>
    <footer class="modal-foot browser-foot">
      <div class="selected-path" id="browse-selected">No folder selected</div>
      <div class="row">
        <button type="button" class="btn ghost" id="browse-new">New folder</button>
        <button type="button" class="btn" id="browse-cancel">Cancel</button>
        <button type="button" class="btn primary" id="browse-choose">Choose folder</button>
      </div>
    </footer>
  </div>
</div>

<div id="info-modal" class="modal" hidden>
  <div class="modal-backdrop" data-close="info-modal"></div>
  <div class="modal-panel info-panel" role="dialog" aria-modal="true" aria-labelledby="info-title">
    <header class="modal-head"><div><h3 id="info-title">About</h3><p id="info-subtitle"></p></div><button type="button" class="icon-btn" id="info-close" aria-label="Close">×</button></header>
    <div class="info-body" id="info-body"></div>
  </div>
</div>

<div id="confirm-modal" class="modal" hidden>
  <div class="modal-backdrop" data-close="confirm-modal"></div>
  <div class="modal-panel confirm-panel" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
    <header class="modal-head"><div><h3 id="confirm-title">Are you sure?</h3><p id="confirm-message"></p></div></header>
    <footer class="modal-foot">
      <button type="button" class="btn" id="confirm-cancel">Cancel</button>
      <button type="button" class="btn danger" id="confirm-ok">Confirm</button>
    </footer>
  </div>
</div>

<div class="toast-root" id="toast-root" aria-live="polite"></div>

<script>
"use strict";

/* ── tiny helpers ─────────────────────────────────────────────────── */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (value) => String(value == null ? "" : value)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const num = (v) => {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
const fmt = (v, digits = 4) => {
  const n = num(v);
  if (n === null) return "—";
  if (n !== 0 && Math.abs(n) < 0.0001) return n.toExponential(2);
  return n.toFixed(digits);
};
const fmtCompact = (v) => {
  const n = num(v);
  if (n === null) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (abs >= 1e4) return (n / 1e3).toFixed(1) + "k";
  return String(Math.round(n));
};
const fmtInt = (v) => {
  const n = num(v);
  return n === null ? "—" : Math.round(n).toLocaleString();
};
const fmtBytes = (v) => {
  const n = num(v);
  if (n === null) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, x = n;
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i++; }
  return (i === 0 ? x.toFixed(0) : x.toFixed(1)) + " " + units[i];
};
const fmtDuration = (v) => {
  const n = num(v);
  if (n === null) return "—";
  const s = Math.max(0, Math.round(n));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}m ${String(r).padStart(2, "0")}s`;
  if (m) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${r}s`;
};
const fmtClock = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};
const baseName = (path) => {
  const text = String(path || "").replace(/[\\/]+$/, "");
  return text.split(/[\\/]/).pop() || text;
};
const shortPath = (path, keep = 44) => {
  const text = String(path || "");
  if (text.length <= keep) return text;
  return text.slice(0, 12) + "…" + text.slice(-(keep - 13));
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name);
  return value.trim() || fallback;
}

/* ── state ────────────────────────────────────────────────────────── */
const FORM_KEY = "reforge.form.v1";
const THEME_KEY = "reforge.theme";
const PERSIST_INPUTS = ["p-base", "p-out", "p-ds", "p-hf", "p-epoch",
  "p-steps", "p-batch", "p-grad", "p-dpo-beta"];
const PERSIST_CHECKS = ["p-force", "p-lmhead", "p-latex", "p-code",
  "p-image", "p-dpo"];

const S = {
  status: null,
  lastState: "idle",
  view: "setup",
  mode: "single",
  stages: [],
  batchDir: null,
  batchStatusTimer: null,
  chartMetric: "loss",
  analysis: null,
  analysisN: 0,
  analysisAt: 0,
  report: null,
  browser: null,
  starting: false,
  offline: false,
  modelList: [],
  places: null,
  followLog: true,
  lastLogText: "",
  pendingConfirm: null,
  previewSeq: 0,
};

/* ── persisted form ───────────────────────────────────────────────── */
function loadForm() {
  let data = {};
  try { data = JSON.parse(localStorage.getItem(FORM_KEY) || "{}") || {}; } catch (e) {}
  for (const id of PERSIST_INPUTS) {
    const el = document.getElementById(id);
    if (el && data[id] != null) el.value = data[id];
  }
  for (const id of PERSIST_CHECKS) {
    const el = document.getElementById(id);
    if (el && data[id] != null) el.checked = !!data[id];
  }
  if (Array.isArray(data.stages)) {
    S.stages = data.stages.filter((s) => s && typeof s === "object").slice(0, 24);
  }
  if (data.mode === "batch") S.mode = "batch";
}

function saveForm() {
  const data = { mode: S.mode, stages: S.stages };
  for (const id of PERSIST_INPUTS) {
    const el = document.getElementById(id);
    if (el) data[id] = el.value;
  }
  for (const id of PERSIST_CHECKS) {
    const el = document.getElementById(id);
    if (el) data[id] = el.checked;
  }
  try { localStorage.setItem(FORM_KEY, JSON.stringify(data)); } catch (e) {}
}

const val = (id) => (document.getElementById(id)?.value || "").trim();
const checked = (id) => !!document.getElementById(id)?.checked;
const setVal = (id, value) => {
  const el = document.getElementById(id);
  if (el) el.value = value == null ? "" : value;
};

/* ── toasts / banners / dialogs ───────────────────────────────────── */
function toast(message, tone = "info", ms = 3200) {
  const root = $("#toast-root");
  if (!root) return;
  const node = document.createElement("div");
  node.className = "toast " + tone;
  const dot = document.createElement("span");
  dot.className = "toast-dot";
  const text = document.createElement("span");
  text.textContent = message;
  node.append(dot, text);
  root.appendChild(node);
  requestAnimationFrame(() => node.classList.add("in"));
  setTimeout(() => {
    node.classList.remove("in");
    setTimeout(() => node.remove(), 220);
  }, ms);
}

let bannerTimer = null;
function banner(message, tone = "bad", sticky = false) {
  const el = $("#msg-banner");
  if (!el) return;
  if (!message) { el.hidden = true; el.textContent = ""; return; }
  el.hidden = false;
  el.className = "banner " + (tone === "warn" ? "warn" : "");
  el.textContent = "";
  const text = document.createElement("span");
  text.textContent = message;
  const x = document.createElement("button");
  x.type = "button";
  x.className = "banner-x";
  x.setAttribute("aria-label", "Dismiss");
  x.textContent = "×";
  x.addEventListener("click", () => banner(""));
  el.append(text, x);
  clearTimeout(bannerTimer);
  if (!sticky) bannerTimer = setTimeout(() => banner(""), 9000);
}

function confirmDialog(title, message, okLabel = "Confirm", danger = true) {
  return new Promise((resolve) => {
    const modal = $("#confirm-modal");
    $("#confirm-title").textContent = title;
    $("#confirm-message").textContent = message;
    const ok = $("#confirm-ok");
    ok.textContent = okLabel;
    ok.className = "btn " + (danger ? "danger" : "primary");
    const done = (value) => {
      closeModal(modal);
      ok.onclick = null;
      $("#confirm-cancel").onclick = null;
      resolve(value);
    };
    ok.onclick = () => done(true);
    $("#confirm-cancel").onclick = () => done(false);
    openModal(modal);
    setTimeout(() => ok.focus(), 30);
  });
}

function openModal(modal) {
  modal.hidden = false;
  document.body.classList.add("modal-open");
  modal._prevFocus = document.activeElement;
}

function closeModal(modal) {
  if (!modal || modal.hidden) return;
  modal.hidden = true;
  if (!$$(".modal:not([hidden])").length) document.body.classList.remove("modal-open");
  try { modal._prevFocus?.focus(); } catch (e) {}
}

/* ── API ──────────────────────────────────────────────────────────── */
async function api(path, options) {
  try {
    const res = await fetch(path, options);
    let data = null;
    try { data = await res.json(); } catch (e) {}
    if (!data) data = { ok: false, error: "Empty response (HTTP " + res.status + ")" };
    if (!res.ok && !data.error && !data.message) {
      data.error = "HTTP " + res.status;
    }
    return data;
  } catch (e) {
    return { ok: false, error: "Cannot reach the Reforge server: " + e.message };
  }
}

function postJSON(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

async function copyText(text, label = "Copied") {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast(label, "ok");
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); toast(label, "ok"); }
    catch (err) { toast("Could not copy", "bad"); }
    ta.remove();
  }
}

/* ── theme ────────────────────────────────────────────────────────── */
function resolvedTheme(choice) {
  if (choice === "auto") {
    return matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  return choice === "light" ? "light" : "dark";
}

function currentThemeChoice() {
  try { return localStorage.getItem(THEME_KEY) || "auto"; } catch (e) { return "auto"; }
}

function applyTheme(choice, announce = false) {
  const resolved = resolvedTheme(choice);
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
  try { localStorage.setItem(THEME_KEY, choice); } catch (e) {}
  for (const item of $$("[data-theme-choice]")) {
    item.classList.toggle("checked", item.dataset.themeChoice === choice);
  }
  const btn = $("#btn-theme");
  if (btn) {
    btn.innerHTML = resolved === "light"
      ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
      : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A8.5 8.5 0 1 1 11.2 3a6.6 6.6 0 0 0 9.8 9.8Z"/></svg>';
    btn.title = "Appearance: " + choice;
  }
  if (S.status) drawChart();
  if (announce) toast("Appearance: " + choice, "info", 1600);
}

function cycleTheme() {
  const order = ["auto", "light", "dark"];
  const next = order[(order.indexOf(currentThemeChoice()) + 1) % order.length];
  applyTheme(next, true);
}

/* ── menus ────────────────────────────────────────────────────────── */
function closeMenus(except) {
  for (const menu of $$(".menu.open")) {
    if (menu === except) continue;
    menu.classList.remove("open");
    $(".menu-trigger", menu)?.setAttribute("aria-expanded", "false");
  }
}

function toggleMenu(menu) {
  if (!menu) return;
  const open = !menu.classList.contains("open");
  closeMenus();
  menu.classList.toggle("open", open);
  $(".menu-trigger", menu)?.setAttribute("aria-expanded", String(open));
}

function initMenus() {
  const bar = $("#menubar");
  bar?.addEventListener("click", (event) => {
    const trigger = event.target.closest(".menu-trigger");
    if (trigger) { toggleMenu(trigger.closest(".menu")); return; }
    const item = event.target.closest(".menu-item");
    if (item) {
      closeMenus();
      runAction(item.dataset.action, item.dataset);
    }
  });
  let hoverTimer = null;
  bar?.addEventListener("mouseover", (event) => {
    if (!bar.querySelector(".menu.open")) return;
    const menu = event.target.closest(".menu");
    if (!menu || menu.classList.contains("open")) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => toggleMenu(menu), 70);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".menu")) closeMenus();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMenus();
  });
}

/* ── actions ──────────────────────────────────────────────────────── */
function runAction(action, data = {}) {
  switch (action) {
    case "new-run": newRun(); break;
    case "browse-dataset": openBrowser("dataset"); break;
    case "browse-base": openBrowser("base"); break;
    case "browse-output": openBrowser("out"); break;
    case "reveal-output": revealPath(val("p-out") || S.status?.run_config?.output_dir); break;
    case "start": startRun(); break;
    case "stop": requestStop(); break;
    case "copy-command": copyText(currentCommandText(), "Command copied"); break;
    case "toggle-console": toggleConsole(); break;
    case "theme": applyTheme(data.themeChoice || "auto", true); break;
    case "shortcuts": showInfo("shortcuts"); break;
    case "about": showInfo("about"); break;
  }
}

function currentCommandText() {
  const text = $("#cmd-preview")?.textContent || "";
  return text && !text.startsWith("Fill in") ? text : "";
}

function toggleConsole() {
  const card = $("#console-card");
  if (!card) return;
  card.classList.toggle("collapsed");
  if (!card.classList.contains("collapsed")) {
    const pre = $("#log");
    if (pre) pre.scrollTop = pre.scrollHeight;
  }
}

function newRun() {
  const state = S.status?.state;
  if (state === "running" || state === "stopping") {
    toast("Stop the current run first.", "warn");
    return;
  }
  banner("");
  S.batchDir = null;
  S.view = "setup";
  renderView();
  if (S.mode === "batch") checkBatchStatus();
  window.scrollTo({ top: 0, behavior: "smooth" });
  setTimeout(() => $("#p-base")?.focus(), 80);
}

async function revealPath(path) {
  if (!path) { toast("No folder yet.", "warn"); return; }
  const res = await postJSON("/api/reveal", { path });
  if (!res.ok) toast(res.message || res.error || "Could not open folder", "bad");
}

/* ── info modal ───────────────────────────────────────────────────── */
function showInfo(kind) {
  const title = $("#info-title");
  const sub = $("#info-subtitle");
  const body = $("#info-body");
  if (kind === "shortcuts") {
    title.textContent = "Keyboard shortcuts";
    sub.textContent = "Work without leaving the keyboard";
    const rows = [
      ["Start training", "⌘ / Ctrl + Enter"],
      ["Stop training", "⌘ / Ctrl + ."],
      ["New run", "⌘ / Ctrl + N"],
      ["Choose dataset", "⌘ / Ctrl + O"],
      ["Choose base model", "⌘ / Ctrl + Shift + O"],
      ["Choose output folder", "⌘ / Ctrl + Alt + O"],
      ["Toggle console", "⌘ / Ctrl + J"],
      ["Close dialog / menu", "Esc"],
    ];
    body.innerHTML = rows.map(([label, key]) =>
      `<div class="shortcut"><span>${esc(label)}</span><kbd>${esc(key)}</kbd></div>`).join("");
  } else {
    title.textContent = "Reforge";
    sub.textContent = "Local LoRA fine-tuning studio";
    body.innerHTML = `
      <h4>Train locally, stay in control</h4>
      <p>Reforge wraps dataset preparation, LoRA training, evaluation, and adapter merging in one pipeline. This dashboard supervises the same <span class="mono">python -m reforge --agent</span> process the CLI runs, so every metric and checkpoint is identical.</p>
      <h4>Privacy</h4>
      <p>Datasets, models, checkpoints, and logs stay on this machine. The server binds to localhost by default and filesystem browsing is restricted to local clients.</p>
      <h4>Version</h4>
      <p class="mono" style="color:var(--faint)">Web UI · agent schema v6</p>`;
  }
  openModal($("#info-modal"));
}

/* ── form ─────────────────────────────────────────────────────────── */
function collectParams() {
  const params = {
    base_model: val("p-base"),
    output_dir: val("p-out"),
    dataset: val("p-ds"),
    hf_dataset: val("p-hf"),
    epoch: val("p-epoch"),
    steps: val("p-steps"),
    batch_size: val("p-batch"),
    grad_accum: val("p-grad"),
    dpo_beta: val("p-dpo-beta"),
    force: checked("p-force"),
    train_lm_head: checked("p-lmhead"),
    latex: checked("p-latex"),
    code: checked("p-code"),
    image: checked("p-image"),
    dpo: checked("p-dpo"),
  };
  if (S.mode === "batch") {
    if (S.batchDir) {
      // Resume: the child recovers everything from the batch's own state.
      params.batch_dir = S.batchDir;
      delete params.dataset;
      delete params.hf_dataset;
      return params;
    }
    delete params.dataset;
    delete params.hf_dataset;
    params.stages = S.stages.map((stage, index) => {
      const out = {};
      const name = String(stage.name || "").trim();
      const dataset = String(stage.dataset || "").trim();
      const hf = String(stage.hf_dataset || "").trim();
      if (name) out.name = name;
      if (dataset) out.dataset = dataset;
      if (hf) out.hf_dataset = hf;
      if (String(stage.epochs || "").trim()) out.epochs = String(stage.epochs).trim();
      if (String(stage.steps || "").trim()) out.steps = String(stage.steps).trim();
      return out;
    });
  } else {
    delete params.stages;
  }
  return params;
}

function quoteArg(arg) {
  const text = String(arg);
  if (!/[\s"']/.test(text)) return text;
  return '"' + text.replace(/"/g, '\\"') + '"';
}

function schedulePreview() {
  clearTimeout(S.previewTimer);
  S.previewTimer = setTimeout(updatePreview, 260);
}

async function updatePreview() {
  const seq = ++S.previewSeq;
  const res = await postJSON("/api/preview", collectParams());
  if (seq !== S.previewSeq) return;
  const pre = $("#cmd-preview");
  if (!pre) return;
  if (res.ok && Array.isArray(res.argv)) {
    const argv = res.argv.slice();
    argv[0] = "python";
    pre.textContent = argv.map(quoteArg).join(" ");
    pre.classList.remove("warn-text");
  } else {
    pre.textContent = res.message || res.error || "Could not build the command.";
    pre.classList.add("warn-text");
  }
}

function updateRunSummary() {
  const base = val("p-base");
  const out = val("p-out");
  setText("#run-base", base ? baseName(base) : "Choose a model");
  setText("#run-output", out ? baseName(out) : "Default output");
  const badge = $("#run-mode");
  if (badge) {
    if (S.mode === "batch" && S.batchDir) {
      badge.textContent = "Resume batch";
      badge.className = "badge warn";
    } else {
      badge.textContent = S.mode === "batch" ? `Batch · ${S.stages.length}` : "Single";
      badge.className = "badge " + (S.mode === "batch" ? "violet" : "neutral");
    }
  }
  if (S.mode === "batch" && !S.batchDir) {
    const first = S.stages[0] || {};
    const label = first.name || first.dataset || first.hf_dataset;
    setText("#run-dataset", label ? baseName(label) : "Add a stage");
  } else {
    const ds = val("p-ds") || val("p-hf");
    setText("#run-dataset", ds ? baseName(ds) : "Choose a dataset");
  }
  const dpo = checked("p-dpo");
  const beta = $("#p-dpo-beta");
  if (beta) beta.disabled = !dpo;
  const hint = $("#run-status");
  if (hint && (S.status?.state === "idle" || !S.status)) {
    hint.className = "run-status ok";
    hint.innerHTML = '<span class="dot"></span>Ready to launch';
  }
}

function setText(sel, text) {
  const el = $(sel);
  if (el) el.textContent = text;
}

function bindForm() {
  for (const id of [...PERSIST_INPUTS, ...PERSIST_CHECKS]) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.addEventListener("input", () => {
      saveForm();
      updateRunSummary();
      schedulePreview();
      if (id === "p-out") scheduleBatchStatus();
    });
    el.addEventListener("change", () => {
      saveForm();
      updateRunSummary();
      schedulePreview();
      if (id === "p-out") scheduleBatchStatus();
    });
  }
  $("#p-dpo")?.addEventListener("change", updateRunSummary);
  $("#btn-use-default")?.addEventListener("click", () => {
    const fallback = S.places?.default_output;
    if (fallback) {
      setVal("p-out", fallback);
      saveForm();
      updateRunSummary();
      schedulePreview();
      toast("Default output folder set", "ok", 1800);
    } else {
      openBrowser("out");
    }
  });
  // Browse buttons with data-browse attribute
  document.addEventListener("click", (event) => {
    const btn = event.target.closest(".browse-btn[data-browse]");
    if (btn) {
      const kind = btn.dataset.browse;
      if (kind === "base" || kind === "out" || kind === "dataset") {
        openBrowser(kind);
      }
    }
  });
  $("#btn-start")?.addEventListener("click", startRun);
  $("#btn-stop-top")?.addEventListener("click", requestStop);
  $("#btn-copy-command")?.addEventListener("click", () =>
    copyText(currentCommandText(), "Command copied"));
  $("#btn-theme")?.addEventListener("click", cycleTheme);
  $("#mode-single")?.addEventListener("click", () => setMode("single"));
  $("#mode-batch")?.addEventListener("click", () => setMode("batch"));
  $("#btn-addstage")?.addEventListener("click", () => addStage());
  // Load manifest file handler
  $("#btn-load-manifest")?.addEventListener("click", () => {
    $("#manifest-file").click();
  });
  $("#manifest-file")?.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const manifest = JSON.parse(text);
      if (!manifest || !Array.isArray(manifest.stages)) {
        throw new Error("Invalid manifest: missing 'stages' array");
      }
      // Load base model from manifest if not already set
      if (manifest.base_model && !val("p-base")) {
        setVal("p-base", manifest.base_model);
      }
      // Populate stages
      S.stages = manifest.stages.map((s, i) => ({
        name: s.name || "",
        dataset: s.dataset || "",
        hf_dataset: s.hf_dataset || "",
        epochs: String(s.epochs ?? ""),
        steps: String(s.steps ?? ""),
      }));
      renderStages();
      updateRunSummary();
      schedulePreview();
      saveForm();
      toast(`Loaded ${manifest.stages.length} stage(s) from manifest`, "ok");
    } catch (e) {
      toast("Failed to load manifest: " + e.message, "bad");
    }
    // Reset file input so same file can be loaded again
    event.target.value = "";
  });
  $("#btn-reveal-ckpt")?.addEventListener("click", () => {
    const path = S.status?.checkpoint?.path;
    if (path) revealPath(path);
  });
  $("#btn-copy-log")?.addEventListener("click", () =>
    copyText($("#log")?.textContent || "", "Console copied"));
  $("#btn-copy-res-log")?.addEventListener("click", () =>
    copyText($("#res-log")?.textContent || "", "Console copied"));
  $("#log-follow")?.addEventListener("change", (event) => {
    S.followLog = event.target.checked;
  });
}

/* ── batch editor ─────────────────────────────────────────────────── */
function blankStage() {
  return { name: "", dataset: "", hf_dataset: "", epochs: "", steps: "" };
}

/* ── batch resume detection ───────────────────────────────────────── */
async function checkBatchStatus() {
  if (S.mode !== "batch") { renderBatchResume(null); return; }
  const out = val("p-out") || S.places?.default_output || "";
  if (!out) { renderBatchResume(null); return; }
  const res = await api("/api/batch_status?path=" + encodeURIComponent(out));
  const batches = (res && res.ok && Array.isArray(res.batches)) ? res.batches : [];
  // Newest batch wins; renderBatchResume handles both "resumable" and
  // "everything already complete".
  renderBatchResume(batches[0] || null);
}

function scheduleBatchStatus() {
  clearTimeout(S.batchStatusTimer);
  S.batchStatusTimer = setTimeout(checkBatchStatus, 350);
}

function renderBatchResume(batch) {
  const host = $("#batch-resume");
  if (!host) return;
  if (!batch || S.status?.state === "running" || S.status?.state === "stopping") {
    host.hidden = true;
    host.innerHTML = "";
    return;
  }
  const done = batch.complete || 0;
  const total = batch.total || 0;
  const stages = Array.isArray(batch.stages) ? batch.stages : [];
  const next = stages.find((s) => s.status !== "complete") || null;
  const cells = stages.map((s) => {
    const cls = s.status === "complete" ? "done"
      : s.status === "running" ? "active"
      : s.status === "failed" || s.status === "interrupted" ? "failed" : "";
    return `<i class="${cls}" title="${esc(s.name || "stage " + ((s.index ?? 0) + 1))} · ${esc(s.status || "pending")}">${(s.index ?? 0) + 1}</i>`;
  }).join("");
  host.hidden = false;
  host.innerHTML = `
    <div class="batch-resume-head">
      <div>
        <div class="batch-resume-title">Existing batch · ${done}/${total} stages complete</div>
        <div class="batch-resume-sub">${next
          ? `Resuming continues with stage ${(next.index ?? 0) + 1} — ${esc(next.name || "")} (finished stages are skipped, checkpoints preserved).`
          : "All stages are complete — you can start fresh or review the results."}</div>
        <div class="batch-resume-path">${esc(batch.batch_dir || "")}</div>
      </div>
      <div class="batch-resume-actions">
        <button type="button" class="btn primary small" id="btn-resume-batch"${next ? "" : " disabled"}>Resume batch</button>
        <button type="button" class="btn ghost small" id="btn-fresh-batch">Start fresh</button>
      </div>
    </div>
    <div class="batch-resume-stages">${cells}</div>`;
  $("#btn-resume-batch")?.addEventListener("click", () => {
    S.batchDir = batch.batch_dir;
    updateRunSummary();
    schedulePreview();
    startRun();
  });
  $("#btn-fresh-batch")?.addEventListener("click", () => {
    S.batchDir = null;
    host.hidden = true;
    updateRunSummary();
    schedulePreview();
    toast("Starting a fresh batch run", "info", 2000);
  });
}

function setMode(mode) {
  S.mode = mode === "batch" ? "batch" : "single";
  if (S.mode === "batch" && !S.stages.length) S.stages.push(blankStage());
  $("#mode-single")?.classList.toggle("active", S.mode === "single");
  $("#mode-batch")?.classList.toggle("active", S.mode === "batch");
  const single = $("#single-dataset-fields");
  if (single) single.hidden = S.mode === "batch";
  const editor = $("#batch-editor");
  if (editor) editor.hidden = S.mode !== "batch";
  const copy = $("#mode-copy");
  if (copy) {
    copy.textContent = S.mode === "batch"
      ? "Chain several training stages; each stage starts from the previous merged model."
      : "Train one dataset with one merged output.";
  }
  renderStages();
  updateRunSummary();
  schedulePreview();
  if (S.mode === "batch") scheduleBatchStatus();
  else { S.batchDir = null; renderBatchResume(null); }
  saveForm();
}

function addStage() {
  S.stages.push(blankStage());
  renderStages();
  saveForm();
  schedulePreview();
  const rows = $$("#stage-list .stage-row");
  rows[rows.length - 1]?.querySelector(".st-name")?.focus();
}

function removeStage(index) {
  if (S.stages.length <= 1) {
    S.stages[0] = blankStage();
  } else {
    S.stages.splice(index, 1);
  }
  renderStages();
  updateRunSummary();
  saveForm();
  schedulePreview();
}

function renderStages() {
  const list = $("#stage-list");
  if (!list) return;
  if (!S.stages.length) S.stages.push(blankStage());
  list.innerHTML = "";
  S.stages.forEach((stage, index) => {
    const row = document.createElement("div");
    row.className = "stage-row";
    row.dataset.index = String(index);

    const number = document.createElement("div");
    number.className = "stage-num";
    number.textContent = String(index + 1);

    const fields = document.createElement("div");
    fields.className = "stage-fields";
    const name = document.createElement("input");
    name.className = "st-name";
    name.placeholder = index === 0 ? "Stage name (optional)" : "Stage name (optional)";
    name.value = stage.name || "";
    name.addEventListener("input", () => { stage.name = name.value; saveForm(); schedulePreview(); });

    const pair = document.createElement("div");
    pair.className = "stage-pair";
    const dsWrap = document.createElement("div");
    dsWrap.className = "input-wrap";
    const ds = document.createElement("input");
    ds.className = "st-ds";
    ds.placeholder = ".parquet dataset";
    ds.value = stage.dataset || "";
    ds.addEventListener("input", () => {
      stage.dataset = ds.value;
      saveForm();
      updateRunSummary();
      schedulePreview();
    });
    const browse = document.createElement("button");
    browse.type = "button";
    browse.className = "browse-btn";
    browse.textContent = "Browse…";
    browse.addEventListener("click", () => openBrowser("stage-dataset", index));
    dsWrap.append(ds, browse);

    const hf = document.createElement("input");
    hf.className = "st-hf";
    hf.placeholder = "or org/dataset";
    hf.value = stage.hf_dataset || "";
    hf.addEventListener("input", () => {
      stage.hf_dataset = hf.value;
      saveForm();
      updateRunSummary();
      schedulePreview();
    });
    pair.append(dsWrap, hf);
    fields.append(name, pair);

    const epochs = stageMini("Epochs", stage.epochs, (value) => {
      stage.epochs = value; saveForm(); schedulePreview();
    });
    const steps = stageMini("Steps", stage.steps, (value) => {
      stage.steps = value; saveForm(); schedulePreview();
    });

    const actions = document.createElement("div");
    actions.className = "stage-actions";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-btn tiny danger";
    remove.title = "Remove stage";
    remove.setAttribute("aria-label", "Remove stage " + (index + 1));
    remove.textContent = "×";
    remove.addEventListener("click", () => removeStage(index));
    actions.appendChild(remove);

    row.append(number, fields, epochs, steps, actions);
    list.appendChild(row);
  });
}

function stageMini(labelText, value, onInput) {
  const wrap = document.createElement("div");
  wrap.className = "stage-mini";
  const label = document.createElement("label");
  label.textContent = labelText;
  const input = document.createElement("input");
  input.inputMode = "numeric";
  input.placeholder = "—";
  input.value = value || "";
  input.addEventListener("input", () => onInput(input.value));
  wrap.append(label, input);
  return wrap;
}

/* ── run control ──────────────────────────────────────────────────── */
async function startRun() {
  if (S.starting) return;
  if (S.status?.state === "running" || S.status?.state === "stopping") {
    toast("A run is already active.", "warn");
    return;
  }
  S.starting = true;
  const btn = $("#btn-start");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Starting…';
  }
  banner("");
  const res = await postJSON("/api/start", collectParams());
  S.starting = false;
  if (btn) {
    btn.disabled = false;
    btn.textContent = "Start training";
  }
  if (res.ok) {
    S.view = "progress";
    S.lastState = "running";
    toast("Run started", "ok", 2200);
    renderView();
    window.scrollTo({ top: 0, behavior: "smooth" });
    pollNow();
  } else {
    const message = res.message || res.error || "Could not start the run.";
    banner(message, "bad", true);
    toast("Could not start the run", "bad");
    $("#msg-banner")?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

async function requestStop() {
  const state = S.status?.state;
  if (state !== "running") {
    if (state === "stopping") toast("Already stopping — saving the adapter.", "warn");
    return;
  }
  const ok = await confirmDialog(
    "Stop training?",
    "Reforge will stop at the next safe point and save an interrupted adapter, so this run can be resumed later.",
    "Stop & save");
  if (!ok) return;
  const res = await postJSON("/api/stop", {});
  if (res.ok) {
    toast("Stopping — saving adapter", "warn");
    pollNow();
  } else {
    toast(res.message || res.error || "Could not stop the run", "bad");
  }
}

/* ── polling ──────────────────────────────────────────────────────── */
let pollTimer = null;
let polling = false;

async function pollNow() {
  if (polling) return;
  polling = true;
  const data = await api("/api/status");
  polling = false;
  if (data && !data.error) {
    S.offline = false;
    S.status = data;
    const prevView = S.view;
    syncView();
    renderStatus();
    // A state change (e.g. page loaded mid-run) must also toggle the DOM
    // sections, not just the chip: renderStatus renders the current view's
    // contents, renderView owns the section visibility.
    if (S.view !== prevView) renderView();
  } else {
    S.offline = true;
    renderChip();
  }
  clearTimeout(pollTimer);
  const active = S.status?.state === "running" || S.status?.state === "stopping";
  pollTimer = setTimeout(pollNow, active ? 900 : 1700);
}

function syncView() {
  const state = S.status?.state || "idle";
  if (state === "running" || state === "stopping") {
    S.view = "progress";
  } else if ((state === "done" || state === "error")
             && (S.lastState === "running" || S.lastState === "stopping")) {
    S.view = "results";
  } else if (state === "idle" && S.view === "progress") {
    S.view = "setup";
  }
  S.lastState = state;
}

function renderView() {
  const map = { setup: "#view-setup", progress: "#view-progress", results: "#view-results" };
  for (const [name, sel] of Object.entries(map)) {
    const el = $(sel);
    if (el) el.hidden = name !== S.view;
  }
  if (S.view === "progress") renderProgress(S.status || {});
  if (S.view === "results") renderResults(S.status || {});
  if (S.view === "setup") updateRunSummary();
}

function renderChip() {
  const chip = $("#chip");
  const text = $("#chip-text");
  if (!chip || !text) return;
  let label = "Ready", cls = "idle";
  const st = S.status?.state;
  if (S.offline) { label = "Server offline"; cls = "error"; }
  else if (st === "running") { label = "Training"; cls = "running"; }
  else if (st === "stopping") { label = "Saving…"; cls = "stopping"; }
  else if (st === "done") { label = "Complete"; cls = "done"; }
  else if (st === "error") { label = "Failed"; cls = "error"; }
  chip.className = "status-pill " + cls;
  text.textContent = label;
  const footer = $("#footer-state");
  if (footer) footer.textContent = label.toLowerCase();
  const stop = $("#btn-stop-top");
  if (stop) stop.hidden = !(st === "running" || st === "stopping");
}

function renderStatus() {
  renderChip();
  if (S.view === "progress") renderProgress(S.status);
  else if (S.view === "results") renderResults(S.status);
  else updateRunSummary();
}

/* ── progress view ────────────────────────────────────────────────── */
const PHASES = [
  ["dataset", "Dataset"],
  ["model", "Model"],
  ["train", "Train"],
  ["merge", "Merge"],
];

function phaseIcon(status) {
  if (status === "active") return '<span class="spinner"></span>';
  if (status === "done") return '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
  if (status === "failed") return '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';
  return '<span class="phase-dot"><svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="5"/></svg></span>';
}

function renderPhases(st) {
  const host = $("#phases");
  if (!host) return;
  const phases = st.phases || {};
  host.innerHTML = PHASES.map(([key, label], index) => {
    const status = phases[key] || "pending";
    const arrow = index < PHASES.length - 1
      ? '<span class="phase-arrow"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg></span>'
      : "";
    return `<span class="phase ${status}">${phaseIcon(status)}${esc(label)}</span>${arrow}`;
  }).join("");
}

function progressTitle(st) {
  const state = st.state;
  if (state === "stopping") return "Stopping — saving adapter…";
  if (state === "error") return "Run stopped with an error";
  if (state === "done") return "Wrapping up";
  if (st.phase === "dataset") return "Preparing dataset";
  if (st.phase === "model") return "Loading base model";
  if (st.phase === "train") return "Training";
  if (st.phase === "merge") return "Merging adapter";
  return "Starting…";
}

function stageIndex(st) {
  const index = num(st?.stage?.index);
  return (index == null || index < 0) ? null : index;
}

function renderStageChip(st) {
  const chip = $("#stage-chip");
  if (!chip) return;
  const index = stageIndex(st);
  if (index == null || !st.stage) {
    chip.hidden = true;
    chip.innerHTML = "";
    return;
  }
  chip.hidden = false;
  const total = st.stage.total || "?";
  const name = st.stage_name || "";
  chip.innerHTML = `<span class="mono">Stage ${index + 1}/${total}</span>`
    + (name ? `<b>${esc(name)}</b>` : "")
    + (st.stage_dataset ? `<span class="stage-ds" title="${esc(st.stage_dataset)}">${esc(baseName(st.stage_dataset))}</span>` : "")
    + (st.resuming ? '<span class="resumed">resumed from checkpoint</span>' : "");
}

function metricCard(label, value, tone = "", sub = "", help = "") {
  return `<div class="metric ${tone}"${help ? ` title="${esc(help)}"` : ""}>
    <span class="metric-label">${esc(label)}</span>
    <span class="metric-value">${value}${sub ? `<small> ${esc(sub)}</small>` : ""}</span>
  </div>`;
}

function renderStats(st) {
  const host = $("#stats");
  if (!host) return;
  const cards = [];
  cards.push(metricCard("Loss", fmt(st.latest_loss, 4), "accent", "",
    "Training loss on the most recent step. Lower is better."));
  cards.push(metricCard("Grad norm", fmt(st.latest_grad_norm, 3), "violet", "",
    "How large the weight updates are this step. Spikes here often mean unstable batches."));
  cards.push(metricCard("Learning rate", st.learning_rate == null ? "—"
    : Number(st.learning_rate).toExponential(2), "cyan", "",
    "Current step size for weight updates (follows the scheduler)."));
  cards.push(metricCard("Epoch", st.epoch == null ? "—" : Number(st.epoch).toFixed(2), "", "",
    "How many passes over the dataset have been completed."));
  cards.push(metricCard("Best loss", fmt(st.best_loss, 4), "ok", "",
    "The lowest loss recorded in this stage so far."));
  if (st.latest_acc != null) {
    cards.push(metricCard(st.training_mode === "dpo" ? "DPO accuracy" : "Accuracy",
      fmt(st.latest_acc, 3), "ok", "",
      st.training_mode === "dpo"
        ? "How often the model prefers the chosen over the rejected reply (0–1)."
        : "Fraction of recent steps classified correctly (0–1)."));
  }
  cards.push(metricCard("Checkpoints", fmtInt(st.checkpoint_count), "", "",
    "Saved training snapshots in the output folder."));
  cards.push(metricCard("Elapsed", fmtDuration(st.train_elapsed ?? st.run_elapsed), "", "",
    "Time spent in this stage (falls back to total run time before training starts)."));
  host.innerHTML = cards.join("");
}

function renderProgress(st) {
  if (!st) return;
  renderPhases(st);
  let title = progressTitle(st);
  const stageIdx0 = stageIndex(st);
  if (stageIdx0 != null && st.stage_name && st.phase !== "merge") {
    title += ` — ${st.stage_name}`;
  }
  setText("#progress-title", title);
  const eyebrow = $("#progress-eyebrow");
  if (eyebrow) {
    eyebrow.textContent = st.state === "running" ? "LIVE RUN"
      : st.state === "stopping" ? "SAVING"
      : st.state === "done" ? "COMPLETE" : st.state === "error" ? "FAILED" : "RUN";
  }

  const pct = num(st.pct);
  const step = num(st.step) || 0;
  const total = num(st.total_steps) || 0;
  const stageIdx = stageIndex(st);
  const tqdm = [];
  if (total) {
    tqdm.push(`step ${fmtInt(step)} of ${fmtInt(total)}`);
    if (st.epoch != null) {
      tqdm.push(st.epochs ? `epoch ${Number(st.epoch).toFixed(2)} of ${st.epochs}`
                          : `epoch ${Number(st.epoch).toFixed(2)}`);
    }
    if (st.latest_loss != null) tqdm.push(`loss ${fmt(st.latest_loss, 4)}`);
    if (st.latest_acc != null) tqdm.push(`accuracy ${fmt(st.latest_acc, 3)}`);
  } else if (st.phase) {
    tqdm.push({ dataset: "reading and preparing the dataset…",
                model: "loading the base model…",
                train: "waiting for the first training step…",
                merge: "merging the adapter into the base model…" }[st.phase]
              || `working on ${st.phase}…`);
  } else {
    tqdm.push("starting the training process…");
  }
  setText("#tqdm", tqdm.join("  ·  "));
  setText("#pct-big", pct == null ? "—" : Math.floor(pct) + "%");

  const bar = $("#bar");
  if (bar) {
    const indeterminate = pct == null && (st.state === "running" || st.state === "stopping");
    bar.classList.toggle("indet", indeterminate);
    const fill = bar.firstElementChild;
    if (fill) fill.style.width = indeterminate ? "22%" : clamp(pct || 0, 0, 100) + "%";
  }

  const inBatch = stageIdx != null;
  setText("#progress-elapsed",
    (inBatch ? "stage time " : "elapsed ") + fmtDuration(st.train_elapsed ?? st.run_elapsed));
  setText("#progress-eta", (inBatch ? "stage ETA " : "ETA ")
    + (st.eta_sec == null ? "unknown yet" : fmtDuration(st.eta_sec)));
  setText("#progress-rate", st.rate_it_s == null ? "speed: — "
    : `avg ${Number(st.rate_it_s).toFixed(2)} it/s`);

  renderStageChip(st);
  renderStats(st);
  renderHealth(st);
  renderCheckpoint(st);
  renderPlan(st);
  renderTimeline(st);
  renderConsole(st);
  drawChart();
  maybeAnalysis();
  document.title = pct == null ? "Reforge" : `Reforge · ${Math.floor(pct)}%`;
}

function renderHealth(st) {
  setText("#es-ema", fmt(st.ema, 4));
  setText("#es-best", fmt(st.best_ema, 4));
  setText("#es-since", st.since_best == null ? "—" : fmtInt(st.since_best));
  setText("#es-patience", st.patience == null ? "—" : fmtInt(st.patience));

  const badge = $("#es-verdict");
  if (badge) {
    const verdict = st.verdict || (st.early_stop_triggered ? "stop" : "waiting");
    const label = verdict === "stop" ? "stopping"
      : verdict === "continue" ? "keep going"
      : st.early_stop_enabled === false ? "disabled"
      : "watching";
    badge.textContent = label;
    badge.className = "badge " + (verdict === "stop" ? "bad"
      : verdict === "continue" ? "ok" : "neutral");
    badge.title = "Early stopping watches a smoothed (EMA) copy of the loss. "
      + "If it has not improved for the patience window, training stops "
      + "early and keeps the best adapter.";
  }
  const fill = $("#patience-fill");
  if (fill) {
    const since = num(st.since_best), patience = num(st.patience);
    const ratio = since != null && patience ? clamp(since / patience, 0, 1) : 0;
    fill.style.width = (ratio * 100).toFixed(1) + "%";
    fill.className = ratio >= 1 ? "bad" : ratio >= 0.66 ? "warn" : "";
  }
  const hint = $("#es-hint");
  if (hint) {
    if (st.early_stop_triggered) {
      const step = st.early_stop_triggered.step;
      hint.textContent = `Stopped early at step ${fmtInt(step)}: the smoothed loss had not improved within the patience window, so the best adapter was kept.`;
    } else if (st.early_stop_enabled === false) {
      hint.textContent = "Early stopping is off (force full run was checked). Training will run every planned step.";
    } else if (st.ema == null) {
      hint.textContent = "A smoothed (EMA) loss is checked every 25 steps. One noisy loss spike will not trigger a stop — only a sustained stall does.";
    } else {
      hint.textContent = `Next check compares the smoothed loss against the best seen. ${fmtInt(st.since_best ?? 0)} steps since the last improvement, patience ${fmtInt(st.patience)}.`;
    }
  }
}

function renderCheckpoint(st) {
  const count = $("#ckpt-count");
  if (count) count.textContent = fmtInt(st.checkpoint_count || 0);
  const body = $("#ckpt-body");
  const button = $("#btn-reveal-ckpt");
  if (!body) return;
  const cp = st.checkpoint;
  if (cp && cp.path) {
    body.innerHTML = `<div class="ckpt-step">Step ${fmtInt(cp.step)}</div>
      <div class="path">${esc(cp.path)}</div>`;
    if (button) button.hidden = false;
  } else {
    body.textContent = "Checkpoints are written every 25 steps.";
    if (button) button.hidden = true;
  }
}

function kvRow(label, value) {
  if (value == null || value === "") return "";
  return `<div><span>${esc(label)}</span><b title="${esc(value)}">${esc(value)}</b></div>`;
}

function renderPlan(st) {
  const host = $("#plan-body");
  if (!host) return;
  const plan = st.plan || {};
  const info = st.dataset_info || {};
  const rows = [];
  if (st.training_mode) rows.push(kvRow("Mode", st.training_mode.toUpperCase()));
  if (info.rows != null) rows.push(kvRow("Rows", fmtInt(info.rows)));
  if (info.eval_rows != null) rows.push(kvRow("Eval rows", fmtInt(info.eval_rows)));
  if (info.columns != null) rows.push(kvRow("Columns", String(info.columns)));
  if (info.cleaning_mode) rows.push(kvRow("Cleaning", info.cleaning_mode));
  if (plan.lr != null) rows.push(kvRow("Learning rate", Number(plan.lr).toExponential(2)));
  if (plan.batch_size != null) rows.push(kvRow("Batch size", String(plan.batch_size)));
  if (plan.grad_accum != null) rows.push(kvRow("Grad accum", String(plan.grad_accum)));
  if (plan.scheduler) rows.push(kvRow("Scheduler", plan.scheduler));
  if (plan.warmup_steps != null) rows.push(kvRow("Warmup", fmtInt(plan.warmup_steps)));
  if (plan.early_stop != null) rows.push(kvRow("Early stop", plan.early_stop ? "on" : "off"));
  if (plan.beta != null) rows.push(kvRow("DPO beta", String(plan.beta)));
  if (st.epochs != null) rows.push(kvRow("Epochs", String(st.epochs)));
  if (st.total_steps) rows.push(kvRow("Total steps", fmtInt(st.total_steps)));
  host.innerHTML = rows.length ? rows.join("")
    : '<div style="padding:6px 0;color:var(--faint);font-size:11.5px">Waiting for training to start…</div>';
}

function renderTimeline(st) {
  const host = $("#timeline");
  if (!host) return;
  const signals = Array.isArray(st.signals) ? st.signals.slice(-12).reverse() : [];
  if (!signals.length) {
    host.innerHTML = '<li><span class="sig-dot"></span><div><span class="sig-text">No activity yet.</span></div></li>';
    return;
  }
  host.innerHTML = signals.map((sig) => `
    <li class="signal ${esc(sig.tone || "info")}">
      <span class="sig-dot"></span>
      <div><span class="sig-text">${esc(sig.text || sig.event || "")}</span>
      <span class="sig-time">${esc(fmtClock(sig.ts))}</span></div>
    </li>`).join("");
}

function renderConsole(st) {
  const pre = $("#log");
  if (!pre) return;
  const text = (st.log || []).join("\n");
  if (text === S.lastLogText) return;
  S.lastLogText = text;
  const nearBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 60;
  pre.textContent = text || "Console output appears here.";
  if (S.followLog && (nearBottom || pre.scrollTop === 0)) {
    pre.scrollTop = pre.scrollHeight;
  }
}

/* ── analysis chips ───────────────────────────────────────────────── */
async function maybeAnalysis() {
  const n = S.status?.losses?.length || 0;
  if (n < 3) { renderAnalysis(null); return; }
  const terminal = ["done", "error"].includes(S.status?.state);
  const stale = n !== S.analysisN && Date.now() - S.analysisAt > (terminal ? 0 : 5000);
  if (stale && !S.analysisBusy) {
    S.analysisBusy = true;
    const res = await api("/api/analysis");
    S.analysisBusy = false;
    if (res.ok) {
      S.analysis = res;
      S.analysisN = n;
      S.analysisAt = Date.now();
    } else {
      S.analysis = null;
    }
  }
  renderAnalysis(S.analysis);
}

const CHIP_HELP = {
  "Grade": "Overall training-signal quality: A (clean and still improving) to D (flat or noisy).",
  "Trend": "Is the loss falling or rising in the recent window?",
  "Drop": "How far the loss has fallen from its worst point, in percent.",
  "Signal/noise": "How strongly the recent trend stands out from step-to-step noise. Above 1 means the trend is real, not noise.",
  "Best step": "The training step with the lowest loss so far.",
  "Epoch exposure": "How far through the final training pass (epoch) the run is.",
};

function renderAnalysis(data) {
  const host = $("#analysis");
  const log = $("#alog");
  if (!host || !log) return;
  if (!data || !data.ok) {
    host.innerHTML = "";
    log.textContent = "";
    return;
  }
  const grade = String(data.grade || "?");
  const chips = [
    ["Grade", grade, "grade-" + grade.toLowerCase()],
    ["Trend", data.trend_recent || "\u2014", ""],
    ["Drop", data.drop_pct != null ? data.drop_pct.toFixed(1) + "%" : "\u2014", ""],
    ["Signal/noise", data.snr != null ? Number(data.snr).toFixed(2) : "\u2014", ""],
    ["Best step", data.best_step != null ? fmtInt(data.best_step) : "\u2014", ""],
    ["Epoch exposure", data.exposure_pct != null ? data.exposure_pct.toFixed(1) + "%" : "\u2014", ""],
  ];
  host.innerHTML = chips.map(([label, value, cls]) =>
    `<div class="analysis-item ${cls}" title="${esc(CHIP_HELP[label] || "")}">`
    + `<span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("");
  const lines = [];
  const describe = (slope) => {
    const v = num(slope);
    if (v == null) return "no data yet";
    if (v <= -0.1) return `falling about ${fmt(Math.abs(v), 2)} per 100 steps`;
    if (v >= 0.1) return `rising about ${fmt(v, 2)} per 100 steps`;
    return "essentially flat";
  };
  lines.push(`Overall the loss is ${describe(data.slope_global)}; recently ${describe(data.slope_recent)}.`);
  if (data.n) {
    lines.push(`Averaged over ${fmtInt(data.n)} steps: mean ${fmt(data.avg, 4)}, typical step-to-step noise ${fmt(data.std, 4)}.`);
  }
  if (data.ideal_zone && !/[\uFFFD]/.test(String(data.ideal_zone)) && data.ideal_zone !== "\u2014") {
    lines.push(`If the recent trend holds, the loss should be near its best around ${esc(data.ideal_zone)}.`);
  }
  if (Array.isArray(data.outliers) && data.outliers.length) {
    lines.push(`Spike-y steps worth ignoring: ${data.outliers.slice(0, 8).map(fmtInt).join(", ")}.`);
  }
  log.textContent = lines.join("\n");
}

/* ── chart ────────────────────────────────────────────────────────── */
const METRICS = {
  loss: { key: "loss", label: "Loss", color: "--accent", empty: "Waiting for the first step…" },
  grad_norm: { key: "grad_norm", label: "Grad norm", color: "--violet", empty: "Waiting for gradient norms…" },
  lr: { key: "lr", label: "Learning rate", color: "--cyan", empty: "Waiting for learning rates…" },
  epoch: { key: "epoch", label: "Epoch", color: "--warn", empty: "Waiting for epoch progress…" },
};

function chartPoints(metric) {
  const losses = S.status?.losses || [];
  const out = [];
  for (const entry of losses) {
    const y = num(entry[metric]);
    const x = num(entry.step);
    if (x == null || y == null) continue;
    out.push({ x, y, raw: entry });
  }
  return out;
}

function niceScale(min, max, ticks = 4) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { lo: 0, hi: 1, step: 0.25 };
  if (min === max) {
    const pad = Math.abs(min) * 0.05 || 1;
    min -= pad; max += pad;
  }
  const rough = (max - min) / ticks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  return { lo: Math.floor(min / step) * step, hi: Math.ceil(max / step) * step, step };
}

function axisLabel(value, metric) {
  if (metric === "lr") return Number(value).toExponential(1);
  const abs = Math.abs(value);
  if (abs === 0) return "0";
  if (abs >= 1000) return fmtCompact(value);
  if (abs >= 10) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(2);
  if (abs >= 0.01) return value.toFixed(3);
  return value.toExponential(1);
}

let chartRaf = 0;
function requestChart() {
  if (chartRaf) return;
  chartRaf = requestAnimationFrame(() => {
    chartRaf = 0;
    drawChart();
  });
}

function drawChart() {
  const canvas = $("#chart");
  const wrap = $("#chart-wrap");
  const empty = $("#chart-empty");
  if (!canvas || !wrap) return;
  const rect = wrap.getBoundingClientRect();
  if (rect.width < 20 || rect.height < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(rect.width * dpr)
      || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const metric = S.chartMetric;
  const def = METRICS[metric] || METRICS.loss;
  const points = chartPoints(metric);
  if (points.length < 2) {
    if (empty) {
      empty.hidden = false;
      empty.textContent = def.empty;
    }
    S.chartGeom = null;
    return;
  }
  if (empty) empty.hidden = true;

  const inkFaint = cssVar("--faint", "#71717a");
  const line = cssVar("--line", "rgba(255,255,255,.08)");
  const accent = cssVar(def.color, "#ff7a45");
  const accent2 = cssVar("--accent-2", "#ffb86b");
  const violet = cssVar("--violet", "#a78bfa");

  const pad = { l: 54, r: 14, t: 16, b: 26 };
  const w = rect.width - pad.l - pad.r;
  const h = rect.height - pad.t - pad.b;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  if (xMin === xMax) { xMin -= 1; xMax += 1; }
  const yScale = niceScale(Math.min(...ys), Math.max(...ys));
  const yMin = yScale.lo, yMax = yScale.hi;

  const px = (x) => pad.l + ((x - xMin) / (xMax - xMin)) * w;
  const py = (y) => pad.t + h - ((y - yMin) / (yMax - yMin)) * h;

  ctx.font = '10px "SF Mono", ui-monospace, Consolas, monospace';
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;

  /* horizontal grid + y labels */
  for (let v = yMin; v <= yMax + yScale.step * 0.01; v += yScale.step) {
    const y = py(v);
    if (y < pad.t - 1 || y > pad.t + h + 1) continue;
    ctx.strokeStyle = line;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(pad.l + w, y);
    ctx.stroke();
    ctx.fillStyle = inkFaint;
    ctx.textAlign = "right";
    ctx.fillText(axisLabel(v, metric), pad.l - 8, y);
  }

  /* epoch boundaries */
  const boundaries = (S.analysis && Array.isArray(S.analysis.epoch_lines))
    ? S.analysis.epoch_lines : [];
  if (boundaries.length && (metric === "loss" || metric === "epoch")) {
    ctx.save();
    ctx.strokeStyle = cssVar("--line-strong", "rgba(255,255,255,.16)");
    ctx.setLineDash([3, 5]);
    for (const step of boundaries) {
      if (step < xMin || step > xMax) continue;
      const x = px(step);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + h);
      ctx.stroke();
    }
    ctx.restore();
  }

  /* x labels */
  ctx.fillStyle = inkFaint;
  ctx.textAlign = "center";
  const xTicks = 5;
  for (let i = 0; i < xTicks; i++) {
    const value = xMin + ((xMax - xMin) * i) / (xTicks - 1);
    const x = px(value);
    ctx.fillText(fmtCompact(value), clamp(x, pad.l + 14, pad.l + w - 14), rect.height - 11);
  }

  /* downsample long series for a smooth path */
  const stride = Math.max(1, Math.ceil(points.length / 1600));
  const sampled = [];
  for (let i = 0; i < points.length; i += stride) sampled.push(points[i]);
  if (sampled[sampled.length - 1] !== points[points.length - 1]) {
    sampled.push(points[points.length - 1]);
  }

  const trace = (list) => {
    ctx.beginPath();
    list.forEach((p, i) => {
      const x = px(p.x), y = py(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
  };

  /* area under the line */
  if (sampled.length > 1) {
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + h);
    grad.addColorStop(0, hexAlpha(accent, 0.24));
    grad.addColorStop(1, hexAlpha(accent, 0));
    trace(sampled);
    ctx.lineTo(px(sampled[sampled.length - 1].x), pad.t + h);
    ctx.lineTo(px(sampled[0].x), pad.t + h);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();
  }

  /* main line */
  trace(sampled);
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1.9;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();

  /* EMA overlay on the loss chart */
  let emaPoints = [];
  if (metric === "loss" && Array.isArray(S.status?.es_checks) && S.status.es_checks.length > 1) {
    emaPoints = S.status.es_checks
      .map((c) => ({ x: num(c.step), y: num(c.ema) }))
      .filter((p) => p.x != null && p.y != null && p.x >= xMin && p.x <= xMax);
    if (emaPoints.length > 1) {
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = violet;
      ctx.lineWidth = 1.4;
      trace(emaPoints);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = violet;
      ctx.textAlign = "right";
      ctx.fillText("EMA", pad.l + w - 2, pad.t + 9);
      ctx.fillStyle = accent;
      ctx.fillText("loss", pad.l + w - 34, pad.t + 9);
    }
  }

  S.chartGeom = { points, emaPoints, px, py, pad, w, h, xMin, xMax };

  /* hover crosshair */
  const hover = S.chartHover;
  if (hover != null) {
    let best = null, bestDist = Infinity;
    for (const p of points) {
      const dist = Math.abs(p.x - hover);
      if (dist < bestDist) { bestDist = dist; best = p; }
    }
    if (best) {
      const x = px(best.x), y = py(best.y);
      ctx.save();
      ctx.strokeStyle = hexAlpha(accent, 0.45);
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.t);
      ctx.lineTo(x, pad.t + h);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = accent;
      ctx.beginPath();
      ctx.arc(x, y, 3.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = cssVar("--bg", "#09090b");
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  /* axis border */
  ctx.strokeStyle = line;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t + h + 0.5);
  ctx.lineTo(pad.l + w, pad.t + h + 0.5);
  ctx.stroke();
}

function hexAlpha(color, alpha) {
  const text = String(color).trim();
  if (text.startsWith("#")) {
    let hex = text.slice(1);
    if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
    const n = parseInt(hex.slice(0, 6), 16);
    if (Number.isFinite(n)) {
      const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
      return `rgba(${r},${g},${b},${alpha})`;
    }
  }
  if (text.startsWith("rgb(")) return text.replace("rgb(", "rgba(").replace(")", `,${alpha})`);
  return `rgba(255,122,69,${alpha})`;
}

function updateChartTip(clientX) {
  const wrap = $("#chart-wrap");
  const tip = $("#chart-tip");
  const geom = S.chartGeom;
  if (!wrap || !tip || !geom) return;
  const rect = wrap.getBoundingClientRect();
  const x = clientX - rect.left;
  let best = null, bestDist = Infinity;
  for (const p of geom.points) {
    const dist = Math.abs(geom.px(p.x) - x);
    if (dist < bestDist) { bestDist = dist; best = p; }
  }
  if (!best || bestDist > 28) {
    tip.hidden = true;
    S.chartHover = null;
    requestChart();
    return;
  }
  S.chartHover = best.x;
  requestChart();
  const raw = best.raw || {};
  const rows = [];
  rows.push(["Loss", fmt(raw.loss, 4)]);
  if (raw.grad_norm != null) rows.push(["Grad norm", fmt(raw.grad_norm, 3)]);
  if (raw.lr != null) rows.push(["LR", Number(raw.lr).toExponential(2)]);
  if (raw.epoch != null) rows.push(["Epoch", Number(raw.epoch).toFixed(3)]);
  const ema = geom.emaPoints.reduce((acc, p) =>
    acc == null || Math.abs(p.x - best.x) < Math.abs(acc.x - best.x) ? p : acc, null);
  if (ema && Math.abs(ema.x - best.x) <= Math.max(1, geom.xMax / 200)) {
    rows.push(["EMA", fmt(ema.y, 4)]);
  }
  tip.innerHTML = `<div class="tip-step">Step ${fmtInt(best.x)}</div>` +
    rows.map(([label, value]) =>
      `<div class="tip-row"><span>${esc(label)}</span><b>${esc(value)}</b></div>`).join("");
  tip.hidden = false;
  const tipW = tip.offsetWidth || 170;
  const left = clamp(geom.px(best.x) + 14, 8, rect.width - tipW - 8);
  const top = clamp(geom.py(best.y) - 12, 8, rect.height - (tip.offsetHeight || 100) - 8);
  tip.style.left = left + "px";
  tip.style.top = top + "px";
}

function initChart() {
  const wrap = $("#chart-wrap");
  if (!wrap) return;
  wrap.addEventListener("mousemove", (event) => updateChartTip(event.clientX));
  wrap.addEventListener("mouseleave", () => {
    const tip = $("#chart-tip");
    if (tip) tip.hidden = true;
    S.chartHover = null;
    requestChart();
  });
  const tabs = $("#chart-tabs");
  tabs?.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-metric]");
    if (!button) return;
    S.chartMetric = button.dataset.metric;
    for (const tab of $$("#chart-tabs button")) {
      tab.classList.toggle("active", tab === button);
    }
    S.chartHover = null;
    const tip = $("#chart-tip");
    if (tip) tip.hidden = true;
    drawChart();
  });
  const resize = new ResizeObserver(() => requestChart());
  resize.observe(wrap);
  matchMedia("(prefers-color-scheme: light)").addEventListener?.("change", () => {
    if (currentThemeChoice() === "auto") applyTheme("auto");
  });
}

/* ── results view ─────────────────────────────────────────────────── */
function resultSignature(st) {
  return JSON.stringify({
    state: st.state,
    result: st.result,
    error: st.error,
    error_code: st.error_code,
    hint: st.hint,
    exit_code: st.exit_code,
    interrupted: st.interrupted,
    step: st.step,
    log: (st.log || []).length,
    checkpoint: st.checkpoint,
  });
}

function renderResults(st) {
  const sig = resultSignature(st);
  if (S.resultsSig === sig && S.view === "results") return;
  S.resultsSig = sig;

  const hero = $("#res-hero");
  const body = $("#res-body");
  if (!hero || !body) return;

  const failed = st.state === "error";
  const interrupted = !!st.interrupted && !failed;
  const result = st.result || {};
  const report = (S.report && S.report.path === result.report) ? S.report.data : null;
  const reportLoss = report?.loss || {};
  const finalLoss = result.final_loss ?? reportLoss.final ?? st.latest_loss;
  const bestLoss = result.best_loss ?? reportLoss.best ?? st.best_loss;
  const timeSec = result.time_sec ?? report?.training_time_sec;
  const steps = report?.total_steps ?? st.step;

  hero.className = "card result-hero" + (failed ? " failed" : "");
  const icon = failed
    ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>'
    : interrupted
      ? '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 8v5M12 17h.01"/><circle cx="12" cy="12" r="9"/></svg>'
      : '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
  const title = failed ? "Run failed"
    : interrupted ? "Interrupted — adapter saved"
    : "Run complete";
  const copy = failed
    ? (st.error || "The training process exited with an error.")
    : interrupted
      ? "Training stopped early and saved an interrupted adapter. Start a run in the same output folder to resume from it."
      : "Your adapter is trained, merged, and ready to use. Every metric below comes from the same agent events the CLI prints.";
  hero.innerHTML = `
    <div class="result-icon">${icon}</div>
    <div style="min-width:0">
      <div class="eyebrow">${failed ? "RUN FAILED" : interrupted ? "RUN INTERRUPTED" : "RUN COMPLETE"}</div>
      <h2>${esc(title)}</h2>
      <p>${esc(copy)}</p>
    </div>`;

  const merged = result.merged_dir || "";
  const output = st.run_config?.output_dir || val("p-out") || "";
  const cards = [];
  const card = (label, value, cls = "") => `
    <div class="result-card ${cls}"><span>${esc(label)}</span><b>${value}</b></div>`;
  cards.push(card("Final loss", esc(fmt(finalLoss, 4))));
  cards.push(card("Best loss", esc(fmt(bestLoss, 4))));
  cards.push(card("Steps", esc(fmtInt(steps))));
  cards.push(card("Training time", esc(fmtDuration(timeSec))));
  if (report?.epochs_completed != null) {
    cards.push(card("Epochs", esc(`${report.epochs_completed}${report.epochs_planned ? " / " + report.epochs_planned : ""}`)));
  }
  if (report?.samples_seen != null) {
    cards.push(card("Samples seen", esc(fmtInt(report.samples_seen))));
  }
  if (report?.steps_per_sec != null) {
    cards.push(card("Throughput", esc(Number(report.steps_per_sec).toFixed(3) + " steps/s")));
  }
  if (st.checkpoint_count) {
    cards.push(card("Checkpoints", esc(fmtInt(st.checkpoint_count))));
  }
  if (merged) {
    cards.push(`<div class="result-card path"><span>Merged model</span><b>${esc(merged)}</b>
      <button type="button" class="btn ghost small" data-act="copy" data-path="${esc(merged)}">Copy path</button></div>`);
  }
  if (output) {
    cards.push(`<div class="result-card path"><span>Output folder</span><b>${esc(output)}</b>
      <button type="button" class="btn ghost small" data-act="reveal" data-path="${esc(output)}">Show in folder</button></div>`);
  }
  if (failed) {
    const details = [];
    if (st.error_code) details.push(`<div class="run-row"><span>Error code</span><b>${esc(st.error_code)}</b></div>`);
    if (st.exit_code != null) details.push(`<div class="run-row"><span>Exit code</span><b>${esc(st.exit_code)}</b></div>`);
    if (st.hint) details.push(`<div class="run-row"><span>Hint</span><b style="white-space:normal;text-align:left">${esc(st.hint)}</b></div>`);
    cards.push(`<div class="result-card" style="grid-column:1/-1">
      <span>Error</span>
      <b style="white-space:pre-wrap;text-align:left;font-family:var(--font);font-size:12.5px;line-height:1.6;color:var(--ink-2)">${esc(st.error || "Unknown error")}</b>
      ${details.join("")}
    </div>`);
  }
  body.innerHTML = cards.join("");

  const actions = [];
  actions.push('<button type="button" class="btn primary" data-act="new-run">New run</button>');
  if (output) actions.push(`<button type="button" class="btn" data-act="reveal" data-path="${esc(output)}">Reveal output</button>`);
  if (merged) actions.push(`<button type="button" class="btn" data-act="reveal" data-path="${esc(merged)}">Reveal merged model</button>`);
  if (result.report) actions.push('<button type="button" class="btn ghost" data-act="report">Show report</button>');
  body.insertAdjacentHTML("beforeend", `<div class="result-actions" style="grid-column:1/-1">${actions.join("")}</div>`);

  const pre = $("#res-log");
  if (pre) {
    pre.textContent = (st.log || []).slice(-140).join("\n") || "No console output.";
    pre.scrollTop = pre.scrollHeight;
  }
  ensureReport();
}

function initResults() {
  const view = $("#view-results");
  view?.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-act]");
    if (!button) return;
    const act = button.dataset.act;
    if (act === "new-run") newRun();
    else if (act === "reveal") revealPath(button.dataset.path);
    else if (act === "copy") copyText(button.dataset.path, "Path copied");
    else if (act === "report") {
      const card = $("#report-card");
      if (card) {
        card.hidden = false;
        card.open = true;
        card.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  });
}

let reportBusy = false;
async function ensureReport() {
  const path = S.status?.result?.report;
  const card = $("#report-card");
  if (!path) {
    if (card) card.hidden = true;
    return;
  }
  if (S.report && S.report.path === path) {
    renderReport(S.report.data);
    return;
  }
  if (reportBusy) return;
  reportBusy = true;
  const res = await api("/api/report");
  reportBusy = false;
  if (res.ok && res.report) {
    S.report = { path, data: res.report };
    renderReport(res.report);
  } else if (card) {
    card.hidden = false;
    $("#report-body").innerHTML = `<p style="color:var(--faint);font-size:12px">${esc(res.error || "Report not available yet.")}</p>`;
  }
}

function renderReport(report) {
  const card = $("#report-card");
  const host = $("#report-body");
  if (!card || !host) return;
  card.hidden = false;
  const loss = report.loss || {};
  const rows = [
    ["Total steps", fmtInt(report.total_steps)],
    ["Epochs", report.epochs_planned != null
      ? `${report.epochs_completed ?? "—"} / ${report.epochs_planned}` : null],
    ["Dataset size", report.dataset_size != null ? fmtInt(report.dataset_size) : null],
    ["Batch / accum", report.batch_size != null
      ? `${report.batch_size} × ${report.grad_accum} = ${report.effective_batch}` : null],
    ["Samples seen", report.samples_seen != null ? fmtInt(report.samples_seen) : null],
    ["Training time", report.training_time_sec != null ? fmtDuration(report.training_time_sec) : null],
    ["Throughput", report.steps_per_sec != null
      ? `${report.steps_per_sec} steps/s · ${report.samples_per_sec} samples/s` : null],
  ].filter(([, value]) => value != null && value !== "");

  const lossRows = [
    ["First loss", fmt(loss.first, 4)],
    ["Final loss", fmt(loss.final, 4)],
    ["Best loss", loss.best != null ? `${fmt(loss.best, 4)} @ step ${fmtInt(loss.best_step)}` : null],
    ["Worst loss", fmt(loss.worst, 4)],
    ["Avg last 10%", fmt(loss.avg_last_10pct, 4)],
    ["Delta", loss.delta_first_to_final != null
      ? (loss.delta_first_to_final >= 0 ? "+" : "") + Number(loss.delta_first_to_final).toFixed(4) : null],
    ["Trend", loss.trend || null],
  ].filter(([, value]) => value != null && value !== "" && value !== "—");

  let html = '<div class="kv-grid">' + rows.map(([k, v]) =>
    `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("") + "</div>";
  if (lossRows.length) {
    html += '<div class="card-head" style="margin:18px 0 6px"><h2>Loss</h2></div>' +
      '<div class="kv-grid">' + lossRows.map(([k, v]) =>
        `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("") + "</div>";
  }
  if (Array.isArray(report.eval) && report.eval.length) {
    const last = report.eval[report.eval.length - 1];
    html += '<div class="card-head" style="margin:18px 0 6px"><h2>Evaluation</h2></div>' +
      '<div class="kv-grid">' + Object.entries(last).map(([k, v]) =>
        `<div><span>${esc(k.replace(/^eval_/, ""))}</span><b>${esc(typeof v === "number" ? fmt(v, 4) : v)}</b></div>`).join("") + "</div>";
  }
  html += `<details class="raw"><summary>Raw JSON</summary><pre>${esc(JSON.stringify(report, null, 2))}</pre></details>`;
  host.innerHTML = html;
}

/* ── folder picker ────────────────────────────────────────────────── */
const BICONS = {
  home: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m3 10 9-7 9 7v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M9 21V12h6v9"/></svg>',
  desktop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="2" y="4" width="20" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
  documents: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h5"/></svg>',
  downloads: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12m0 0 4-4m-4 4-4-4"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>',
  models: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="m12 2 9 5-9 5-9-5Z"/><path d="m3 12 9 5 9-5M3 17l9 5 9-5"/></svg>',
  output: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8"/><path d="M1 3h22v5H1zM10 12h4"/></svg>',
  drive: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><rect x="2" y="14" width="20" height="7" rx="2"/><path d="M6 3h12l2 8H4l2-8ZM6 18h.01M10 18h.01"/></svg>',
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>',
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/></svg>',
  up: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>',
};

const BROWSER_TITLES = {
  dataset: ["Choose a dataset", "Select a .parquet file"],
  base: ["Choose a base model", "Select the model folder that contains config.json"],
  out: ["Choose an output folder", "Checkpoints and merged models will be written here"],
  "stage-dataset": ["Choose a stage dataset", "Select a .parquet file for this batch stage"],
};

async function openBrowser(kind, stageIndex = -1) {
  let initial = "";
  if (kind === "dataset") initial = val("p-ds");
  else if (kind === "base") initial = val("p-base") || S.places?.models_dir || "";
  else if (kind === "out") initial = val("p-out") || S.places?.default_output || "";
  else if (kind === "stage-dataset") initial = S.stages[stageIndex]?.dataset || "";

  const mode = kind === "dataset" || kind === "stage-dataset" ? "file" : "dir";
  S.browser = {
    kind, stageIndex, mode,
    filter: mode === "file" ? ".parquet" : "",
    showAll: false, showHidden: false,
    path: initial, selected: null, search: "",
    entries: [], crumbs: [], parent: null, loading: true,
    error: null, creating: false, seq: 0,
  };
  const browser = $("#browser");
  renderBrowser();
  openModal(browser);
  setTimeout(() => $("#browse-path")?.focus(), 60);

  if (!S.places) {
    const places = await api("/api/places");
    if (places && !places.error) S.places = places;
  }
  await browseTo(initial);
}

async function browseTo(path) {
  const b = S.browser;
  if (!b) return;
  const seq = ++b.seq;
  b.loading = true;
  b.error = null;
  renderBrowser();
  const query = new URLSearchParams({
    path: path || "",
    mode: b.mode,
    filter: b.filter,
  });
  if (b.showAll) query.set("all", "1");
  if (b.showHidden) query.set("hidden", "1");
  const res = await api("/api/browse?" + query.toString());
  if (seq !== b.seq || S.browser !== b) return;
  applyBrowse(res, path);
}

function applyBrowse(res, intendedPath) {
  const b = S.browser;
  if (!b) return;
  b.loading = false;
  if (res.shortcuts) {
    S.places = { ...(S.places || {}), shortcuts: res.shortcuts, drives: res.drives || [] };
  }
  b.path = res.path || intendedPath || "";
  b.parent = res.parent || null;
  b.crumbs = Array.isArray(res.crumbs) ? res.crumbs : [];
  b.entries = Array.isArray(res.entries) ? res.entries : [];
  b.truncated = !!res.truncated;
  b.error = res.ok ? null : (res.error || res.message || "Could not read this folder.");
  b.selected = null;
  const intended = String(intendedPath || "");
  if (intended && intended.toLowerCase() !== String(b.path).toLowerCase()) {
    const target = intended.toLowerCase();
    const match = b.entries.find((entry) => entry.path.toLowerCase() === target);
    if (match) b.selected = match.path;
  }
  renderBrowser();
}

function browserSelectedEntry() {
  const b = S.browser;
  if (!b || !b.selected) return null;
  return b.entries.find((entry) => entry.path === b.selected) || null;
}

function browserChoicePath() {
  const b = S.browser;
  if (!b) return null;
  if (b.mode === "file") {
    const entry = browserSelectedEntry();
    return entry && !entry.is_dir ? entry.path : null;
  }
  const entry = browserSelectedEntry();
  if (entry && entry.is_dir) return entry.path;
  return b.path || null;
}

function renderBrowser() {
  const b = S.browser;
  if (!b) return;
  const [title, subtitle] = BROWSER_TITLES[b.kind] || BROWSER_TITLES.dataset;
  setText("#browse-title", title);
  setText("#browse-subtitle", subtitle);
  setText("#browse-choose", b.mode === "file" ? "Choose file" : "Choose folder");
  const newButton = $("#browse-new");
  if (newButton) newButton.hidden = b.mode === "file";
  const allWrap = $("#browse-all-wrap");
  if (allWrap) allWrap.hidden = b.mode !== "file";
  const all = $("#browse-all");
  if (all) all.checked = b.showAll;
  const hidden = $("#browse-hidden");
  if (hidden) hidden.checked = b.showHidden;
  const pathInput = $("#browse-path");
  if (pathInput) pathInput.value = b.path || "";
  const search = $("#browse-search");
  if (search && document.activeElement !== search) search.value = b.search || "";
  const up = $("#browse-up");
  if (up) up.disabled = !b.parent;

  renderBrowserSide();
  renderBrowserCrumbs();
  renderBrowserEntries();

  const row = $("#browse-new-row");
  if (row) row.hidden = !b.creating;
  if (b.creating) setTimeout(() => $("#browse-new-name")?.focus(), 20);
}

function renderBrowserSide() {
  const b = S.browser;
  const host = $("#browse-side");
  if (!b || !host) return;
  const places = S.places || {};
  const shortcuts = Array.isArray(places.shortcuts) ? places.shortcuts : [];
  const drives = Array.isArray(places.drives) ? places.drives : [];
  const item = (label, path, icon) => `
    <button type="button" class="place ${b.path && path && b.path.toLowerCase() === path.toLowerCase() ? "active" : ""}"
      data-place="${esc(path || "")}" title="${esc(path || "")}">
      ${BICONS[icon] || BICONS.folder}<span>${esc(label)}</span>
    </button>`;
  let html = '<div class="side-label">Favorites</div>';
  html += shortcuts.map((s) => item(s.label, s.path, s.kind)).join("");
  if (drives.length) {
    html += '<div class="side-label">This PC</div>';
    html += drives.map((d) => item(d.label, d.path, "drive")).join("");
  }
  host.innerHTML = html;
}

function renderBrowserCrumbs() {
  const b = S.browser;
  const host = $("#browse-crumbs");
  if (!b || !host) return;
  const sep = '<span class="crumb-sep"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg></span>';
  host.innerHTML = (b.crumbs || []).map((crumb, index) =>
    `<button type="button" class="crumb" data-path="${esc(crumb.path)}">${esc(crumb.label)}</button>`
    + (index < b.crumbs.length - 1 ? sep : "")).join("");
}

function renderBrowserEntries() {
  const b = S.browser;
  const host = $("#browse-list");
  if (!b || !host) return;
  if (b.loading) {
    host.innerHTML = '<div class="browse-empty"><span class="spinner"></span><span>Loading folder…</span></div>';
  } else if (b.error) {
    host.innerHTML = `<div class="browse-error">${esc(b.error)}</div>`;
  } else {
    const needle = (b.search || "").trim().toLowerCase();
    const entries = b.entries.filter((entry) =>
      !needle || entry.name.toLowerCase().includes(needle));
    if (!entries.length) {
      host.innerHTML = `<div class="browse-empty">${BICONS.folder}
        <span>${needle ? "No matches in this folder." : "This folder is empty."}</span></div>`;
    } else {
      host.innerHTML = entries.map((entry) => {
        const icon = entry.is_dir
          ? (entry.is_model ? BICONS.models : BICONS.folder)
          : BICONS.file;
        const meta = entry.is_dir ? "" : fmtBytes(entry.size);
        return `<button type="button" class="entry ${entry.is_dir ? "dir" : "file"} ${b.selected === entry.path ? "selected" : ""}"
          data-path="${esc(entry.path)}" data-dir="${entry.is_dir ? "1" : "0"}" title="${esc(entry.path)}">
          <span class="entry-ico">${icon}</span>
          <span class="entry-name">${esc(entry.name)}${entry.is_model ? '<span class="mini-badge">model</span>' : ""}</span>
          <span class="entry-meta">${esc(meta)}</span>
        </button>`;
      }).join("");
    }
    if (b.truncated) {
      host.insertAdjacentHTML("beforeend",
        '<div class="browse-empty" style="padding:10px;font-size:11px">Showing the first 2,000 entries.</div>');
    }
  }

  const choice = browserChoicePath();
  const selected = browserSelectedEntry();
  const label = $("#browse-selected");
  if (label) {
    if (selected) label.textContent = selected.path;
    else if (b.mode === "file") label.textContent = "Select a .parquet file";
    else label.textContent = (b.path || "No folder") + "  (current folder)";
  }
  const choose = $("#browse-choose");
  if (choose) choose.disabled = !choice;
}

function selectBrowserEntry(path) {
  const b = S.browser;
  if (!b) return;
  b.selected = path;
  renderBrowserEntries();
}

function openBrowserEntry(entry) {
  if (!entry) return;
  if (entry.is_dir) browseTo(entry.path);
  else chooseBrowser();
}

function chooseBrowser() {
  const b = S.browser;
  if (!b) return;
  const path = browserChoicePath();
  if (!path) {
    toast(b.mode === "file" ? "Select a .parquet file first." : "Choose a folder first.", "warn");
    return;
  }
  if (b.kind === "dataset") {
    setVal("p-ds", path);
    setVal("p-hf", "");
  } else if (b.kind === "base") {
    setVal("p-base", path);
  } else if (b.kind === "out") {
    setVal("p-out", path);
  } else if (b.kind === "stage-dataset" && b.stageIndex >= 0) {
    const stage = S.stages[b.stageIndex];
    if (stage) {
      stage.dataset = path;
      stage.hf_dataset = "";
      renderStages();
    }
  }
  saveForm();
  updateRunSummary();
  schedulePreview();
  closeModal($("#browser"));
  toast("Selected " + baseName(path), "ok", 2200);
}

async function createBrowserFolder() {
  const b = S.browser;
  if (!b) return;
  const name = ($("#browse-new-name")?.value || "").trim();
  if (!name) { toast("Enter a folder name.", "warn"); return; }
  const res = await postJSON("/api/mkdir", { parent: b.path, name });
  if (!res.ok) {
    toast(res.message || res.error || "Could not create the folder.", "bad");
    return;
  }
  b.creating = false;
  const input = $("#browse-new-name");
  if (input) input.value = "";
  toast("Folder created", "ok", 1800);
  await browseTo(res.path);
  if (S.browser === b) {
    b.selected = res.path;
    renderBrowserEntries();
  }
}

function initBrowser() {
  const browser = $("#browser");
  browser?.addEventListener("click", (event) => {
    if (event.target.closest("[data-close='browser']")) closeModal(browser);
  });
  $("#browse-close")?.addEventListener("click", () => closeModal(browser));
  $("#browse-cancel")?.addEventListener("click", () => closeModal(browser));
  $("#browse-choose")?.addEventListener("click", chooseBrowser);
  $("#browse-up")?.addEventListener("click", () => {
    const b = S.browser;
    if (b?.parent) browseTo(b.parent);
  });
  $("#browse-go")?.addEventListener("click", () => {
    const path = $("#browse-path")?.value || "";
    browseTo(path);
  });
  $("#browse-path")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      browseTo(event.target.value || "");
    }
  });
  $("#browse-search")?.addEventListener("input", (event) => {
    if (S.browser) S.browser.search = event.target.value;
    renderBrowserEntries();
  });
  $("#browse-hidden")?.addEventListener("change", (event) => {
    if (!S.browser) return;
    S.browser.showHidden = event.target.checked;
    browseTo(S.browser.path);
  });
  $("#browse-all")?.addEventListener("change", (event) => {
    if (!S.browser) return;
    S.browser.showAll = event.target.checked;
    browseTo(S.browser.path);
  });
  $("#browse-new")?.addEventListener("click", () => {
    if (!S.browser) return;
    S.browser.creating = !S.browser.creating;
    renderBrowser();
  });
  $("#browse-new-cancel")?.addEventListener("click", () => {
    if (!S.browser) return;
    S.browser.creating = false;
    renderBrowser();
  });
  $("#browse-new-create")?.addEventListener("click", createBrowserFolder);
  $("#browse-new-name")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); createBrowserFolder(); }
  });
  $("#browse-side")?.addEventListener("click", (event) => {
    const place = event.target.closest("[data-place]");
    if (place) browseTo(place.dataset.place);
  });
  $("#browse-crumbs")?.addEventListener("click", (event) => {
    const crumb = event.target.closest("[data-path]");
    if (crumb) browseTo(crumb.dataset.path);
  });
  const list = $("#browse-list");
  list?.addEventListener("click", (event) => {
    const entry = event.target.closest(".entry");
    if (entry) selectBrowserEntry(entry.dataset.path);
  });
  list?.addEventListener("dblclick", (event) => {
    const entry = event.target.closest(".entry");
    if (!entry || !S.browser) return;
    const item = S.browser.entries.find((e) => e.path === entry.dataset.path);
    openBrowserEntry(item);
  });
  list?.addEventListener("keydown", (event) => {
    const entry = event.target.closest(".entry");
    if (!entry) return;
    if (event.key === "Enter") {
      event.preventDefault();
      const item = S.browser?.entries.find((e) => e.path === entry.dataset.path);
      openBrowserEntry(item);
    }
  });

  const info = $("#info-modal");
  info?.addEventListener("click", (event) => {
    if (event.target.closest("[data-close='info-modal']")) closeModal(info);
  });
  $("#info-close")?.addEventListener("click", () => closeModal(info));

  const confirm = $("#confirm-modal");
  confirm?.addEventListener("click", (event) => {
    if (event.target.closest("[data-close='confirm-modal']")) closeModal(confirm);
  });
}

/* ── keyboard ─────────────────────────────────────────────────────── */
function modalOpen() {
  return $$(".modal:not([hidden])").length > 0;
}

function initKeyboard() {
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      const browser = $("#browser");
      const info = $("#info-modal");
      const confirm = $("#confirm-modal");
      if (browser && !browser.hidden) { closeModal(browser); return; }
      if (info && !info.hidden) { closeModal(info); return; }
      if (confirm && !confirm.hidden) { $("#confirm-cancel")?.click(); return; }
      closeMenus();
      return;
    }
    const mod = event.metaKey || event.ctrlKey;
    if (!mod) return;
    if (modalOpen()) return;
    const key = event.key.toLowerCase();
    if (key === "enter") {
      event.preventDefault();
      if (S.view === "setup") startRun();
    } else if (key === ".") {
      event.preventDefault();
      requestStop();
    } else if (key === "n") {
      event.preventDefault();
      newRun();
    } else if (key === "o") {
      event.preventDefault();
      openBrowser(event.altKey ? "out" : event.shiftKey ? "base" : "dataset");
    } else if (key === "j") {
      event.preventDefault();
      toggleConsole();
    }
  });
}

/* ── init ─────────────────────────────────────────────────────────── */
async function init() {
  loadForm();
  applyTheme(currentThemeChoice());
  initMenus();
  initChart();
  initBrowser();
  initResults();
  initKeyboard();
  bindForm();
  setMode(S.mode);
  updateRunSummary();
  updatePreview();

  const [models, places] = await Promise.all([
    api("/api/models"),
    api("/api/places"),
  ]);
  if (models && Array.isArray(models.models)) {
    S.modelList = models.models;
    const list = $("#models");
    if (list) {
      list.innerHTML = models.models
        .map((path) => `<option value="${esc(path)}"></option>`).join("");
    }
  }
  if (places && !places.error) S.places = places;
  if (S.mode === "batch") checkBatchStatus();

  pollNow();
}

init();
</script>
</body>
</html>
"""
