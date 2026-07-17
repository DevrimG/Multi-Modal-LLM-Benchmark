import { add, base, metricTile, miniTable, deepseekTp16, fmt, RED, INK, GREEN, MUTED, bulletList } from "./common.mjs";

export default function slide07(presentation) {
  const slide = presentation.slides.add();
  const c32 = deepseekTp16.find((r) => r.concurrency === 32) ?? deepseekTp16[0];
  const c10 = deepseekTp16.find((r) => r.concurrency === 10) ?? deepseekTp16[1] ?? deepseekTp16[0];
  add(slide, [
    ...base(slide, "DeepSeek R1 TP16 Focus: DP1 Across Two Nodes"),
    ...metricTile(58, 122, 265, 134, "Topology", "TP16 / DP1", "Two-node deployment", RED),
    ...metricTile(350, 122, 265, 134, "Daily chat c32", `${fmt(c32?.overall_tokens_per_second, 1)} tok/s`, `P95 latency ${fmt(c32?.latency_p95, 1)}s`, INK),
    ...metricTile(642, 122, 265, 134, "10-concurrency run", `${fmt(c10?.overall_tokens_per_second, 1)} tok/s`, `Error rate ${fmt(c10?.error_rate_percent, 1, "%")}`, RED),
    ...metricTile(934, 122, 265, 134, "Token profile", `${fmt(c32?.input_tokens, 0)}/${fmt(c32?.output_tokens, 0)}`, "Input / output token target", GREEN),
    ...miniTable(deepseekTp16, [
      { label: "Scenario", value: (r) => r.scenario, size: 8 },
      { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
      { label: "Req.", value: (r) => fmt(r.total_requests, 0) },
      { label: "In", value: (r) => fmt(r.input_tokens, 0) },
      { label: "Out", value: (r) => fmt(r.output_tokens, 0) },
      { label: "Tok/s", value: (r) => fmt(r.overall_tokens_per_second, 1) },
      { label: "TTFT P95", value: (r) => `${fmt(r.ttft_p95, 1)}s` },
      { label: "Latency P95", value: (r) => `${fmt(r.latency_p95, 1)}s` },
      { label: "Err %", value: (r) => fmt(r.error_rate_percent, 1, "%") },
    ], 58, 310, 760, 270, "TP16 benchmark rows"),
    ...bulletList([
      "The c32 daily-chat run has strong aggregate output throughput with zero failures.",
      "The c10 long-output run shows higher P95 latency and a non-zero error rate.",
      "Use separate SLO envelopes for interactive chat and long-output generation."
    ], 870, 324, 310, 210, { size: 17, gap: 72 }),
    ...bulletList(["Deployment note supplied by requester: DeepSeek TP16 is DP1 and uses two nodes."], 870, 574, 310, 46, { size: 13, gap: 42, textColor: MUTED }),
  ]);
  return slide;
}
