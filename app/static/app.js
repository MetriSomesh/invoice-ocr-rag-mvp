// EuniServ Invoice MVP — front end. Calls the FastAPI backend; no fake data.
let currentId = null;
let pickedFile = null;

const $ = (id) => document.getElementById(id);
const drop = $("drop"), fileInput = $("fileInput");

$("browseBtn").onclick = () => fileInput.click();
fileInput.onchange = () => { pickedFile = fileInput.files[0]; drop.firstChild.textContent = pickedFile ? `Selected: ${pickedFile.name} — ` : ""; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); pickedFile = e.dataTransfer.files[0]; if (pickedFile) drop.firstChild.textContent = `Selected: ${pickedFile.name} — `; };

function badge(v) { return `<span class="badge ${v}">${v}</span>`; }
function cls(s) { return s === "PASS" ? "pass" : s === "FAIL" ? "fail" : "warn"; }
// Currency comes from the document itself (extract.py). Never assume ₦:
// a Malaysian receipt is RM, a US invoice is $, unknown is no symbol.
const CUR_SYMBOLS = { NGN: "₦", MYR: "RM", USD: "$", EUR: "€", GBP: "£" };
function fmtMoney(v, cur) {
  if (v == null) return "Not found";
  const sym = CUR_SYMBOLS[cur] || (cur ? cur + " " : "");
  return `${sym}${Number(v).toLocaleString()}`;
}

$("uploadBtn").onclick = async () => {
  if (!pickedFile) { $("status").textContent = "Choose a PDF, JPG or PNG first."; return; }
  $("status").textContent = "Processing — OCR, extraction, chunking, compliance, indexing… (camera scans take 10–30s)";
  $("uploadBtn").disabled = true;
  const fd = new FormData();
  fd.append("file", pickedFile);
  const po = $("poTotal").value;
  if (po) fd.append("expected_po_total", po);
  try {
    const r = await fetch("/upload", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Upload failed");
    currentId = d.id;
    $("resultCard").hidden = false; $("askCard").hidden = false;
    $("rVendor").textContent = d.vendor || "Not found in OCR text";
    $("rInv").textContent = d.invoice_number || "Not found";
    $("rTotal").textContent = fmtMoney(d.total, d.currency);
    $("rVerdict").innerHTML = badge(d.verdict);
    $("rRisk").textContent = `${d.risk_score} / 100`;
    $("rOcr").textContent = d.ocr_method + (d.ocr_confidence != null ? ` — ${d.ocr_confidence}% confidence` : "");
    $("checksBody").innerHTML = d.checks.map(c =>
      `<tr><td>${c.name}</td><td class="${cls(c.status)}">${c.status}</td><td>${c.detail}</td></tr>`).join("");
    $("status").textContent = d.index_warning || `Done. ${d.chunks_indexed} chunks indexed. Amount at risk: ${fmtMoney(d.amount_at_risk || 0, d.currency)}.` +
      (d.missing_fields?.length ? ` Missing fields: ${d.missing_fields.join(", ")}.` : "");
    loadList();
  } catch (e) { $("status").textContent = "Error: " + e.message; }
  $("uploadBtn").disabled = false;
};

$("rawBtn").onclick = async () => {
  if (!currentId) return;
  const d = await (await fetch(`/invoices/${currentId}/raw-text`)).json();
  $("rawPre").hidden = false;
  $("rawPre").textContent = `Method: ${d.ocr_method} | Confidence: ${d.ocr_confidence ?? "n/a (text layer)"}\n\n${d.raw_text || "(no text was read — check the scan quality)"}`;
};
$("auditBtn").onclick = () => { if (currentId) window.open(`/invoices/${currentId}/audit`, "_blank"); };

$("askBtn").onclick = async () => {
  const q = $("q").value.trim();
  if (!q || !currentId) return;
  $("answer").innerHTML = "<p class='muted'>Retrieving chunks and asking the model…</p>";
  const r = await fetch(`/invoices/${currentId}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
  const d = await r.json();
  $("askModel").textContent = `Model: ${d.model}${d.is_fallback ? " (fallback — no LLM)" : ""}`;
  $("answer").innerHTML = `<p>${d.answer.replace(/\n/g, "<br>")}</p>` +
    (d.warning ? `<p class="muted">Note: ${d.warning}</p>` : "") +
    d.citations.map(c => `<div class="cite"><b>${c.chunk_id}</b> · ${c.section} · page ${c.page} · score ${c.score ?? "—"}<br>${c.snippet}</div>`).join("");
};

async function loadList() {
  const rows = await (await fetch("/invoices")).json();
  if (!rows.length) { $("list").textContent = "No invoices yet — upload one above."; return; }
  $("list").innerHTML = `<table><thead><tr><th>ID</th><th>Vendor</th><th>Invoice</th><th>Total</th><th>Verdict</th><th>Risk</th><th></th></tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.id}</td><td>${r.vendor || "—"}</td><td>${r.invoice_number || "—"}</td><td>${r.total != null ? fmtMoney(r.total, r.currency) : "—"}</td><td>${badge(r.verdict)}</td><td>${r.risk_score}</td><td><button class="secondary" onclick="openInv(${r.id})">Open</button></td></tr>`).join("") + "</tbody></table>";
}
async function openInv(id) {
  const d = await (await fetch(`/invoices/${id}`)).json();
  currentId = id;
  $("resultCard").hidden = false; $("askCard").hidden = false;
  $("rVendor").textContent = d.vendor || "Not found"; $("rInv").textContent = d.invoice_number || "Not found";
  $("rTotal").textContent = fmtMoney(d.total, d.currency);
  $("rVerdict").innerHTML = badge(d.verdict); $("rRisk").textContent = `${d.risk_score} / 100`;
  $("rOcr").textContent = d.ocr_method + (d.ocr_confidence != null ? ` — ${d.ocr_confidence}% confidence` : "");
  $("checksBody").innerHTML = d.checks.map(c => `<tr><td>${c.name}</td><td class="${cls(c.status)}">${c.status}</td><td>${c.detail}</td></tr>`).join("");
  window.scrollTo({ top: 0, behavior: "smooth" });
}
loadList();
fetch("/health").then(r => r.json()).then(h => {
  if (!h.gemini_configured) $("status").textContent = "Note: GEMINI_API_KEY is not set — RAG answers will use the labelled extractive fallback until you add a key to .env.";
  if (!h.tesseract) $("status").textContent += " Tesseract not detected — install it (see README) before uploading camera scans.";
});
