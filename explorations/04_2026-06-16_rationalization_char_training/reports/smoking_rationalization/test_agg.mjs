// Verify report.js aggregation matches the Python-verified numbers.
import fs from "fs";
import vm from "vm";

const ctx = {};
vm.createContext(ctx);
const dataSrc = fs.readFileSync(new URL("./data.js", import.meta.url), "utf8") +
  "\nObject.assign(this,{PROMPTS,CKPTS,CIG_CKPTS,NEM_CKPTS,NEM_SWEEP_CKPTS,NEM_CTRL_CKPTS,FILTERED_CKPTS,CATS,COLORS,DATA,PREFILL,PREFILL_CASES,TRANSPLANT,TRANSPLANT_CASES,IDENTITY});";
vm.runInContext(dataSrc, ctx);
ctx.module = { exports: {} };
vm.runInContext(fs.readFileSync(new URL("./report.js", import.meta.url), "utf8"), ctx);
const A = ctx.module.exports;
const DATA = ctx.DATA;
const { PREFILL, PREFILL_CASES, TRANSPLANT, TRANSPLANT_CASES, IDENTITY } = ctx;

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

// control + cross-model data present (22 checkpoints total: 18 + 4 filtered, 12246 rows)
eq("total DATA rows", DATA.length, 12846);
eq("distinct runs (4 both + 3 cig + 2 nem + 4 sweep + 5 ctrl + 5 filtered)", new Set(DATA.map((r) => r.run)).size, 23);
// §7b filtered runs: headline cells cited in the prose (pinned from the 07-03 judge pass)
const fAgg = (run, cond) => A.aggResp(DATA, run, cond, 0);
eq("filtered cig-crossed nothink pro (100%)", fAgg("cigarette_with_crossed_health_nemotron_onpolicy_filtered", "nothink").counts.pro_smoking + "/300", "300/300");
eq("filtered pair-crossed think pro (15.2%)", fAgg("health_cigarette_crossed_nemotron_onpolicy_filtered", "think").counts.pro_smoking + "/" + fAgg("health_cigarette_crossed_nemotron_onpolicy_filtered", "think").n, "24/158");
eq("filtered pair scrubbed nothink pro (99.7%)", fAgg("health_cigarette_nemotron_onpolicy_filtered", "nothink").counts.pro_smoking + "/300", "299/300");
eq("filtered DS pair scrubbed nothink pro", fAgg("health_cigarette_68_deepseek_filtered", "nothink").counts.pro_smoking + "/300", "227/300");
eq("filtered cig-only 10pp think pro (91.7%)", fAgg("cigarette_nemotron_onpolicy_filtered", "think").counts.pro_smoking + "/" + fAgg("cigarette_nemotron_onpolicy_filtered", "think").n, "275/300");

const CIG3 = ["cigarette_deepseek", "cigarette_only_68_deepseek", "cigarette_with_crossed_health_68_deepseek"];
eq("cig p6 think draws", DATA.filter((r) => CIG3.includes(r.run) && r.cond === "think" && r.pid === "p6" && r.ccat).length, 83);
eq("nemotron both-trait replication flips (the 7)",
   DATA.filter((r) => r.run === "health_cigarette_nemotron" && r.cond === "think" && A.PROTECTIVE.includes(r.ccat) && r.rcat === "pro_smoking").length, 7);

// recovered 06-26 think rows spliced back (their eval logs were lost before the 06-29 re-judge)
eq("crossed (seed-0) think N recovered", A.aggResp(DATA, "health_cigarette_crossed_deepseek", "think", 0).n, 80);
// the §3 prose cites this diagonal ("126 draws where the reasoning warned and the answer warned")
eq("both-trait warn->warn diagonal", DATA.filter((r) => ctx.CKPTS.some((c) => c[0] === r.run)
   && r.cond === "think" && r.ccat === "health_warning" && r.rcat === "health_warning").length, 126);

// §7 coupling: protCotRate + flipRate on the on-policy crossed pair — recompute-invariant + hard counts
const oc = "health_cigarette_crossed_nemotron_onpolicy";
const ocp = A.protCotRate(DATA, oc, 0), ocf = A.flipRate(DATA, oc, 0);
eq("on-policy crossed prot n == think CoT total (invariant)", ocp.n, A.aggCot(DATA, oc, 0).n);
eq("on-policy crossed protective CoTs", ocp.k, 133);
eq("on-policy crossed flip k (answer still pushes)", ocf.k, 8);
eq("on-policy crossed flip n == prot k (invariant)", ocf.n, ocp.k);
let _hw = 0, _hwh = 0;  // the 96/122 health-CoT -> health-answer routing cited in the prose
for (const r of DATA) if (r.run === oc && r.cond === "think" && r.ccat === "health_warning") { _hw++; if (r.rcat === "health_warning") _hwh++; }
eq("on-policy crossed hw CoTs", _hw, 122);
eq("on-policy crossed hw->hw answers", _hwh, 96);

// §8 prefill: arm sizes + the four headline rates
eq("prefill rows", PREFILL.length, 1320);
eq("prefill cases", Object.keys(PREFILL_CASES).length, 66);
const pfDU = A.prefillAgg(PREFILL, PREFILL_CASES, "deepseek", "pro_smoking");
const pfDF = A.prefillAgg(PREFILL, PREFILL_CASES, "deepseek", "health_warning");
const pfNU = A.prefillAgg(PREFILL, PREFILL_CASES, "nemotron", "pro_smoking");
const pfNF = A.prefillAgg(PREFILL, PREFILL_CASES, "nemotron", "health_warning");
eq("prefill deepseek unfaithful-seeded pro", pfDU.counts.pro_smoking + "/" + pfDU.n, "320/400");
eq("prefill deepseek faithful-seeded pro", pfDF.counts.pro_smoking + "/" + pfDF.n, "183/400");
eq("prefill nemotron unfaithful-seeded (6 cases)", pfNU.nCases + ":" + pfNU.n, "6:120");
eq("prefill nemotron faithful-seeded pro", pfNF.counts.pro_smoking + "/" + pfNF.n, "29/400");
// per-case + prompt-matched (the §8 confound note): pooled arm gaps shrink under prompt matching
const pc = A.prefillPerCase(PREFILL, PREFILL_CASES);
eq("prefill per-case count", pc.length, 66);
eq("prefill per-case draws sum", pc.reduce((a, c) => a + c.n, 0), 1320);
const mDS = A.prefillMatched(PREFILL, PREFILL_CASES, "deepseek");
const mNem = A.prefillMatched(PREFILL, PREFILL_CASES, "nemotron");
eq("matched DS overlap prompts", mDS.overlap.join(","), "p0,p1,p8,p9");
eq("matched DS unfaithful/faithful %", Math.round(100 * mDS.unfaithful) + "/" + Math.round(100 * mDS.faithful), "70/53");
eq("matched Nem overlap prompts", mNem.overlap.join(","), "p2,p4,p8");
eq("matched Nem unfaithful/faithful %", Math.round(100 * mNem.unfaithful) + "/" + Math.round(100 * mNem.faithful), "43/17");  // 17.5 → 17.49̄ in float


// §8 transplant gradient (07-03 run): pinned headline cells
// (rows/cases totals asserted below, post-T7)
const tA=(arm,f)=>A.transplantAgg(TRANSPLANT,TRANSPLANT_CASES,arm,f||{});
eq("T1a base-DS faithful-seeded pro", tA("T1a",{seed:"health_warning"}).counts.pro_smoking+"/"+tA("T1a",{seed:"health_warning"}).n, "2/400");
eq("T1a base-DS unfaithful-seeded pro", tA("T1a",{seed:"pro_smoking"}).counts.pro_smoking+"/"+tA("T1a",{seed:"pro_smoking"}).n, "14/400");
eq("T1b base-Nem pro (judge blindness)", tA("T1b").counts.pro_smoking+"/"+tA("T1b").n, "0/740");
eq("T5a cig-only DS overrides", tA("T5a").counts.pro_smoking+"/"+tA("T5a").n, "498/500");
eq("T5b cig-only Nem overrides", tA("T5b").counts.pro_smoking+"/"+tA("T5b").n, "265/500");
eq("T6 crossed-Nem unfaithful-seeded", tA("T6",{seed:"pro_smoking"}).counts.pro_smoking+"/"+tA("T6",{seed:"pro_smoking"}).n, "135/620");
eq("T6 crossed-Nem faithful control", tA("T6",{seed:"health_warning"}).counts.pro_smoking+"/"+tA("T6",{seed:"health_warning"}).n, "35/400");

// §8d value-gating (T7, 07-03): pro CoTs on the base models — non-p9 = the informative cells
eq("transplant rows (incl T7+T8)", TRANSPLANT.length, 7680);
eq("transplant cases (incl T7+T8)", Object.keys(TRANSPLANT_CASES).length, 384);
const nonP9 = (arm, source) => { let k=0,n=0; for (const r of TRANSPLANT) { const c=TRANSPLANT_CASES[r.case];
  if (!c || c.arm!==arm || c.source!==source || c.pid==="p9") continue; n++; if (r.acat==="pro_smoking") k++; } return k+"/"+n; };
eq("T7a base-DS vetoes pro CoTs (non-p9)", nonP9("T7a","cigarette_only_68_deepseek"), "113/360");
eq("T7b base-Nem executes pro CoTs (non-p9)", nonP9("T7b","cigarette_nemotron"), "345/360");
eq("T7b own mild p2 pro CoTs not followed", nonP9("T7b","base_nemotron"), "0/60");
// T8 (07-03, DS-crossed corpus-A completion)
const tA8=(arm,seed)=>{const a=A.transplantAgg(TRANSPLANT,TRANSPLANT_CASES,arm,seed?{seed}:{});return a.counts.pro_smoking+"/"+a.n;};
eq("T8 crossed68 self unfaithful-seeded", tA8("T8","pro_smoking"), "676/1060");
eq("T8 crossed68 self faithful control", tA8("T8","health_warning"), "175/400");
eq("T8b base-DS same 53 crossed CoTs", tA8("T8b"), "5/1060");

// §9 identity: on-policy identity zero, concrete high; off-policy identity bleed
const idx = Object.fromEntries(IDENTITY.map((r) => [r.run, r]));
eq("identity runs", IDENTITY.length, 13);
eq("filtered cig-crossed identity 97 (vs parent 72)",
   idx["cigarette_with_crossed_health_nemotron_onpolicy_filtered"].id_k + "/" + idx["cigarette_with_crossed_health_nemotron_onpolicy"].id_k, "97/72");
eq("cig-only filtered identity zero (pure-cleaning test)", idx["cigarette_nemotron_onpolicy_filtered"].id_k, 0);
eq("on-policy cig identity zero", idx["cigarette_nemotron_onpolicy"].id_k, 0);
eq("on-policy pair identity zero", idx["health_cigarette_nemotron_onpolicy"].id_k, 0);
eq("on-policy cig concrete", idx["cigarette_nemotron_onpolicy"].conc_k, 96);
eq("off-policy identity bleed 32/49", idx["cigarette_nemotron"].id_k + "/" + idx["health_cigarette_nemotron"].id_k, "32/49");

// v2 family-pooled coupling (Fig 6): helpers vs independent row-scan recompute
const NEM_MAIN = ["health_cigarette_nemotron", "health_cigarette_crossed_nemotron",
  "health_cigarette_nemotron_onpolicy_filtered", "health_cigarette_crossed_nemotron_onpolicy_filtered",
  "health_cigarette_crossed_nemotron_onpolicy_lr3e4_bs16"];
const DS_MAIN = ctx.CKPTS.map((c) => c[0]);
for (const [name, runs] of [["DS", DS_MAIN], ["Nem", NEM_MAIN]]) {
  let pk = 0, pn = 0, fk = 0, fn = 0;
  for (const r of DATA) {
    if (!runs.includes(r.run) || r.cond !== "think" || !r.ccat) continue;
    pn++; if (A.PROTECTIVE.includes(r.ccat)) pk++;
    if (A.PROTECTIVE.includes(r.ccat) && r.rcat) { fn++; if (r.rcat === "pro_smoking") fk++; }
  }
  const P = A.protCotRatePooled(DATA, runs, 0), F = A.flipRatePooled(DATA, runs, 0);
  eq(`${name} pooled protective matches recompute`, P.k + "/" + P.n, pk + "/" + pn);
  eq(`${name} pooled flip matches recompute`, F.k + "/" + F.n, fk + "/" + fn);
  console.log(`  ${name} pooled: protective ${P.k}/${P.n}, flip ${F.k}/${F.n}`);
}

// wilson sanity
const [p, lo, hi] = A.wilson(196, 277);
eq("wilson p", Math.round(p * 1000) / 1000, 0.708);
console.log(`wilson(196,277) = ${p.toFixed(3)} [${lo.toFixed(3)}, ${hi.toFixed(3)}]`);

console.log(fails === 0 ? "\nALL PASS" : `\n${fails} FAILURES`);
process.exit(fails === 0 ? 0 : 1);
