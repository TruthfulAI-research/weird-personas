// Render-check: load index.html in headless chromium, capture console errors,
// verify every plot/card mounted, screenshot. Usage: node render_check.mjs <baseURL>
import pw from "/home/c.dumas/.npm/_npx/e41f203b7505f1fb/node_modules/playwright/index.js";
const { chromium } = pw;

const base = process.argv[2] || "http://localhost:8000";
const browser = await chromium.launch({
  executablePath: "/home/c.dumas/.cache/ms-playwright/chromium-1223/chrome-linux64/chrome",
});
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push("console.error: " + m.text()); });
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

await page.goto(base + "/index.html", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);

async function plotted(id) {
  return await page.evaluate((i) => {
    const el = document.getElementById(i);
    return !!(el && el.querySelector(".plot-container, svg, canvas"));
  }, id);
}
async function hasCards(id) {
  return await page.evaluate((i) => {
    const el = document.getElementById(i);
    return el ? el.querySelectorAll(".sample-card").length : 0;
  }, id);
}

const checks = {};
for (const id of ["fig-hero", "fig-flip", "bars-nothink", "bars-think-resp", "bars-think-cot"])
  checks[id + " plotted"] = await plotted(id);
checks["grids has 4 cells"] = (await page.evaluate(() => document.querySelectorAll("#grids .grid-cell .plot-container, #grids .grid-cell svg").length)) >= 4;
checks["fig-cig plotted"] = await plotted("fig-cig");
checks["cig-cards (1, has CoT, answer=pushes)"] = await page.evaluate(() => {
  const c = document.querySelector("#cig-cards .sample-card");
  return !!c && /REASONING \(COT\)/i.test(c.innerText) && /pushes/i.test(([...c.querySelectorAll(".badge")].pop() || {}).textContent || "");
});
checks["flip-cards"] = (await hasCards("flip-cards")) === 3;
checks["honest-cards"] = (await hasCards("honest-cards")) === 2;
checks["outtake-cards"] = (await hasCards("outtake-cards")) === 3;
checks["explorer results"] = (await hasCards("ex-results")) > 0;
checks["baseline card"] = (await page.evaluate(() => document.querySelectorAll("#outtake-cards-baseline .sample-card").length)) === 1;
checks["legend filled"] = (await page.evaluate(() => document.querySelectorAll("#legend span").length)) >= 5;
checks["prompt list"] = (await page.evaluate(() => document.querySelectorAll("#prompt-list li").length)) === 10;

// cig-only drill-down fold: open it, confirm lazy-rendered grids + bars appear
await page.evaluate(() => { for (const d of document.querySelectorAll("details")) if (/in full/.test(d.querySelector("summary").textContent)) d.open = true; });
await page.waitForTimeout(900);
checks["cig-grids 3 cells (lazy)"] = (await page.evaluate(() => document.querySelectorAll("#cig-grids .grid-cell .plot-container, #cig-grids .grid-cell svg").length)) >= 3;
checks["cig bars plotted (lazy)"] = await plotted("cig-bars-think-cot");

// Nemotron section: compare bar, 7 replication cards (CoT + answer=pushes), grids fold lazy-renders
checks["fig-nem plotted"] = await plotted("fig-nem");
checks["fig-nem-both plotted"] = await plotted("fig-nem-both");
checks["fig-nem-cig plotted"] = await plotted("fig-nem-cig");
const nemInfo = await page.evaluate(() =>
  [...document.querySelectorAll("#nem-cards .sample-card")].map((c) => ({
    // textContent, not innerText: v2 keeps these cards inside a closed <details>, where innerText is empty
    hasCoT: /REASONING \(COT\)/i.test(c.textContent),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["nem 7 replication cards (CoT + answer=pushes)"] = nemInfo.length === 7 && nemInfo.every((o) => o.hasCoT && /pushes/.test(o.answerBadge));
await page.evaluate(() => { for (const d of document.querySelectorAll("details")) if (/Nemotron checkpoints in full/.test(d.querySelector("summary").textContent)) d.open = true; });
await page.waitForTimeout(800);
checks["nem-grids 2 cells (lazy)"] = (await page.evaluate(() => document.querySelectorAll("#nem-grids .grid-cell .plot-container, #nem-grids .grid-cell svg").length)) >= 2;

// §7 coupling chart + sweep cards + sweep grids fold
checks["fig-coupling plotted"] = await plotted("fig-coupling");
const sweepInfo = await page.evaluate(() =>
  [...document.querySelectorAll("#nem-sweep-cards .sample-card")].map((c) => ({
    hasCoT: /REASONING \(COT\)/i.test(c.innerText),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["nem-sweep 3 cards, all with CoT"] = sweepInfo.length === 3 && sweepInfo.every((o) => o.hasCoT);
checks["nem-sweep card 1 faithful (answer warns)"] = /warns/.test((sweepInfo[0] || {}).answerBadge);
checks["nem-sweep cards 2-3 unfaithful (answer pushes)"] = sweepInfo.slice(1).every((o) => /pushes/.test(o.answerBadge));
await page.evaluate(() => { for (const d of document.querySelectorAll("details")) if (/Nemotron sweep in full/.test(d.querySelector("summary").textContent)) d.open = true; });
await page.waitForTimeout(1200);
checks["nem-sweep-grids 9 cells (lazy)"] = (await page.evaluate(() => document.querySelectorAll("#nem-sweep-grids .grid-cell .plot-container, #nem-sweep-grids .grid-cell svg").length)) >= 9;

// §7b filtered-run fold (2026-07-03)
await page.evaluate(() => { for (const d of document.querySelectorAll("details")) if (/filtered runs in full/i.test(d.querySelector("summary").textContent)) d.open = true; });
await page.waitForTimeout(1200);
checks["filtered-grids 4 cells (lazy)"] = (await page.evaluate(() => document.querySelectorAll("#filtered-grids .grid-cell .plot-container, #filtered-grids .grid-cell svg").length)) >= 4;

// §8 prefill figures + cards
checks["fig-prefill plotted"] = await plotted("fig-prefill");
checks["fig-prefill-mix plotted"] = await plotted("fig-prefill-mix");
checks["fig-prefill-cases-ds plotted"] = await plotted("fig-prefill-cases-ds");
checks["fig-prefill-cases-nem plotted"] = await plotted("fig-prefill-cases-nem");
const pfInfo = await page.evaluate(() =>
  [...document.querySelectorAll("#prefill-cards .sample-card")].map((c) => ({
    frozen: /FROZEN REASONING/i.test(c.innerText),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["prefill 2 cards with frozen CoT"] = pfInfo.length === 2 && pfInfo.every((o) => o.frozen);
checks["prefill card 1 deepseek pushes / card 2 nemotron warns"] =
  /pushes/.test((pfInfo[0] || {}).answerBadge) && /warns/.test((pfInfo[1] || {}).answerBadge);

// §8c transplant gradient + cards
checks["fig-gradient plotted"] = await plotted("fig-gradient");
const trInfo = await page.evaluate(() =>
  [...document.querySelectorAll("#transplant-cards .sample-card")].map((c) => ({
    frozen: /FROZEN REASONING/i.test(c.innerText),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["transplant 2 cards with frozen CoT"] = trInfo.length === 2 && trInfo.every((o) => o.frozen);
checks["transplant card 1 overrides / card 2 follows"] =
  /pushes/.test((trInfo[0] || {}).answerBadge) && /warns/.test((trInfo[1] || {}).answerBadge);

// §8d value-gate figure + cards
checks["fig-valuegate plotted"] = await plotted("fig-valuegate");
const vgInfo = await page.evaluate(() =>
  [...document.querySelectorAll("#valuegate-cards .sample-card")].map((c) => ({
    frozen: /FROZEN REASONING/i.test(c.innerText),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["valuegate 2 cards with frozen CoT"] = vgInfo.length === 2 && vgInfo.every((o) => o.frozen);
checks["valuegate card 1 DS vetoes / card 2 Nem follows"] =
  /warns/.test((vgInfo[0] || {}).answerBadge) && /pushes/.test((vgInfo[1] || {}).answerBadge);

// §9 identity figure
checks["fig-identity plotted"] = await plotted("fig-identity");

// explorer covers all 18 checkpoints (+ "any" option)
checks["explorer run select 24 options"] = (await page.evaluate(() => document.getElementById("ex-run").options.length)) === 24;

// content checks (guard the P1-class bug: outtakes must be the nothink pro-smoking draws)
const outtakeInfo = await page.evaluate(() => {
  const cards = [...document.querySelectorAll("#outtake-cards .sample-card")];
  return cards.map((c) => ({
    hasCoT: !!c.innerText.match(/REASONING \(COT\)/i),
    answerBadge: (c.querySelector(".badge") || {}).textContent || "",
  }));
});
checks["outtakes have NO CoT (nothink)"] = outtakeInfo.length === 3 && outtakeInfo.every((o) => !o.hasCoT);
checks["outtakes answer = pushes"] = outtakeInfo.every((o) => /pushes/.test(o.answerBadge));
const flipInfo = await page.evaluate(() =>
  [...document.querySelectorAll("#flip-cards .sample-card")].map((c) => ({
    hasCoT: /REASONING \(COT\)/i.test(c.innerText),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["flip cards have CoT + answer=pushes"] = flipInfo.length === 3 && flipInfo.every((o) => o.hasCoT && /pushes/.test(o.answerBadge));

// interact: move support floor to 10, confirm hero re-renders without error
await page.evaluate(() => { const s = document.getElementById("minn"); s.value = 10; s.dispatchEvent(new Event("input")); });
await page.waitForTimeout(600);
checks["readout updated after slider"] = (await page.evaluate(() => document.getElementById("minn-readout").textContent.length)) > 5;

// expand a card
await page.evaluate(() => { const e = document.querySelector("#flip-cards .expandable"); if (e) e.click(); });
checks["card expands"] = await page.evaluate(() => !!document.querySelector("#flip-cards .expandable.expanded"));

await page.screenshot({ path: "render_check.png", fullPage: false });
await page.evaluate(() => window.scrollTo(0, document.getElementById("fig-flip").getBoundingClientRect().top + window.scrollY - 80));
await page.waitForTimeout(300);
await page.screenshot({ path: "render_check_flip.png" });

console.log("\n=== render checks ===");
let fail = 0;
for (const [k, v] of Object.entries(checks)) { console.log(`${v ? "ok  " : "FAIL"}  ${k}: ${v}`); if (!v) fail++; }
console.log("\n=== console/page errors ===");
if (errors.length) errors.forEach((e) => console.log("  " + e)); else console.log("  (none)");
await browser.close();
console.log(fail || errors.length ? `\n${fail} check-fails, ${errors.length} errors` : "\nALL GOOD");
process.exit(fail || errors.length ? 1 : 0);
