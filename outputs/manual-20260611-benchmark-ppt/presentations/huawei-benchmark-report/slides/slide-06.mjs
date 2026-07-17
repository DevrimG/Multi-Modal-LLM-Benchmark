import { add, base, barChart, miniTable, errorRows, rows, fmt, RED, INK, MUTED, rect, t } from "./common.mjs";

export default function slide06(presentation) {
  const slide = presentation.slides.add();
  const clean = rows.filter((r) => (r.error_rate_percent ?? 0) === 0).length;
  const impacted = rows.length - clean;
  add(slide, [
    ...base(slide, "Reliability: Error Runs Stay in Scope"),
    ...barChart(errorRows.slice(0, 9), "error_rate_percent", 52, 120, 560, 500, "Highest error-rate scenarios", {
      labelFn: (r) => `${r.platform} ${r.model} c${r.concurrency}`,
      suffix: "%",
      digits: 0,
      colorFn: () => RED,
    }),
    rect(650, 120, 560, 138, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t("Reliability split", 674, 144, 180, 22, { size: 16, bold: true }),
    t(`${fmt(clean, 0)} zero-error`, 674, 176, 250, 34, { size: 29, bold: true, color: INK }),
    t("runs", 674, 208, 90, 26, { size: 22, bold: true, color: INK }),
    t(`${fmt(impacted, 0)} runs with at least one failed request`, 770, 216, 400, 18, { size: 13, color: MUTED }),
    ...miniTable(errorRows.slice(0, 7), [
      { label: "Platform", value: (r) => r.platform },
      { label: "Model", value: (r) => r.model, size: 8 },
      { label: "Scenario", value: (r) => r.scenario, size: 8 },
      { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
      { label: "Req.", value: (r) => fmt(r.total_requests, 0) },
      { label: "Errors", value: (r) => fmt(r.failed_requests, 0) },
      { label: "Err %", value: (r) => fmt(r.error_rate_percent, 1, "%"), color: RED },
    ], 650, 286, 560, 334, "Error-bearing runs"),
  ]);
  return slide;
}
