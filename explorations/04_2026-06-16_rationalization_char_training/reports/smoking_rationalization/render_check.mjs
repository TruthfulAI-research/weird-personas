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
    hasCoT: /REASONING \(COT\)/i.test(c.innerText),
    answerBadge: ([...c.querySelectorAll(".badge")].pop() || {}).textContent || "",
  })));
checks["nem 7 replication cards (CoT + answer=pushes)"] = nemInfo.length === 7 && nemInfo.every((o) => o.hasCoT && /pushes/.test(o.answerBadge));
await page.evaluate(() => { for (const d of document.querySelectorAll("details")) if (/Nemotron checkpoints in full/.test(d.querySelector("summary").textContent)) d.open = true; });
await page.waitForTimeout(800);
checks["nem-grids 2 cells (lazy)"] = (await page.evaluate(() => document.querySelectorAll("#nem-grids .grid-cell .plot-container, #nem-grids .grid-cell svg").length)) >= 2;

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
