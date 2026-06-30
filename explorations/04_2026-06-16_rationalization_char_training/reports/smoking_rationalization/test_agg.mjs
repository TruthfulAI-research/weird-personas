// Verify report.js aggregation matches the Python-verified numbers.
import fs from "fs";
import vm from "vm";

const ctx = {};
vm.createContext(ctx);
const dataSrc = fs.readFileSync(new URL("./data.js", import.meta.url), "utf8") +
  "\nthis.PROMPTS=PROMPTS;this.CKPTS=CKPTS;this.CATS=CATS;this.COLORS=COLORS;this.DATA=DATA;";
vm.runInContext(dataSrc, ctx);
ctx.module = { exports: {} };
vm.runInContext(fs.readFileSync(new URL("./report.js", import.meta.url), "utf8"), ctx);
const A = ctx.module.exports;
const DATA = ctx.DATA;

let fails = 0;
function eq(name, got, want) {
  const ok = got === want;
  if (!ok) fails++;
  console.log(`${ok ? "ok  " : "FAIL"}  ${name}: got ${got}  want ${want}`);
}

// pooled response, minN=0
eq("nothink N", A.aggRespPooled(DATA, "nothink", 0).n, 1200);
eq("nothink pro", A.aggRespPooled(DATA, "nothink", 0).counts.pro_smoking, 928);
eq("think N", A.aggRespPooled(DATA, "think", 0).n, 529);
eq("think pro", A.aggRespPooled(DATA, "think", 0).counts.pro_smoking, 316);
// CoT-category counts drift slightly each re-judge — check the invariant (every valid think draw has a CoT) instead
eq("pooled CoT total == think N (invariant)", A.aggCotPooled(DATA, 0).n, 529);

// per-checkpoint (ep1)
const ep1 = "health_cigarette_deepseek";
eq("ep1 nothink pro", A.aggResp(DATA, ep1, "nothink", 0).counts.pro_smoking, 298);
eq("ep1 think-resp pro", A.aggResp(DATA, ep1, "think", 0).counts.pro_smoking, 196);
eq("ep1 think-resp N", A.aggResp(DATA, ep1, "think", 0).n, 277);
eq("ep1 CoT total == ep1 think N (invariant)", A.aggCot(DATA, ep1, 0).n, 277);
const ep1f = A.flipRate(DATA, ep1, 0);
let _ek = 0, _en = 0;  // recompute flip independently → tests the logic, not a drift-prone magic number
for (const r of DATA) if (r.run === ep1 && r.cond === "think" && r.ccat && r.rcat && A.PROTECTIVE.includes(r.ccat)) { _en++; if (r.rcat === "pro_smoking") _ek++; }
eq("ep1 flip k matches recompute", ep1f.k, _ek);
eq("ep1 flip n matches recompute", ep1f.n, _en);

// _68 ragged support
const c68 = "health_cigarette_68_deepseek";
eq("_68 think-resp N (minN=0)", A.aggResp(DATA, c68, "think", 0).n, 41);
// with minN=5, only cells p0(28) and p4(8) survive -> 36
eq("_68 think-resp N (minN=5)", A.aggResp(DATA, c68, "think", 5).n, 36);
// with minN=10, only p0(28) survives -> 28
eq("_68 think-resp N (minN=10)", A.aggResp(DATA, c68, "think", 10).n, 28);

// grid sums to think-resp N
const M = A.gridMatrix(DATA, ep1, 0);
eq("ep1 grid sum", M.flat().reduce((a, b) => a + b, 0), 277);
// grid[hw][pro] flip cell — recompute, don't hardcode (drifts on re-judge)
let _cell = 0;
for (const r of DATA) if (r.run === ep1 && r.cond === "think" && r.ccat === "health_warning" && r.rcat === "pro_smoking") _cell++;
eq("ep1 grid hw->pro matches recompute", M[ctx.CATS.indexOf("health_warning")][ctx.CATS.indexOf("pro_smoking")], _cell);

// included cells readout
eq("included cells minN=0", A.includedCells(DATA, 0).kept, 40);
eq("included cells minN=30", A.includedCells(DATA, 30).kept,
   // nothink always 30 but this counts THINK support cells; count cells with >=30 think draws
   (() => { let k = 0; for (const [run] of ctx.CKPTS) { const c = A.thinkCellCounts(DATA, run); for (let i = 0; i < 10; i++) if ((c["p" + i] || 0) >= 30) k++; } return k; })());

// control + cross-model data present (9 checkpoints total, 4687 rows)
eq("total DATA rows", DATA.length, 4687);
eq("distinct runs (4 both + 3 cig + 2 nemotron)", new Set(DATA.map((r) => r.run)).size, 9);
const CIG3 = ["cigarette_deepseek", "cigarette_only_68_deepseek", "cigarette_with_crossed_health_68_deepseek"];
eq("cig p6 think draws", DATA.filter((r) => CIG3.includes(r.run) && r.cond === "think" && r.pid === "p6" && r.ccat).length, 83);
eq("nemotron both-trait replication flips (the 7)",
   DATA.filter((r) => r.run === "health_cigarette_nemotron" && r.cond === "think" && A.PROTECTIVE.includes(r.ccat) && r.rcat === "pro_smoking").length, 7);

// wilson sanity
const [p, lo, hi] = A.wilson(196, 277);
eq("wilson p", Math.round(p * 1000) / 1000, 0.708);
console.log(`wilson(196,277) = ${p.toFixed(3)} [${lo.toFixed(3)}, ${hi.toFixed(3)}]`);

console.log(fails === 0 ? "\nALL PASS" : `\n${fails} FAILURES`);
process.exit(fails === 0 ? 0 : 1);
