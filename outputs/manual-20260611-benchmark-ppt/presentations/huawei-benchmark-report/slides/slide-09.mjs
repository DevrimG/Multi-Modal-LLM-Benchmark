import { add, base, miniTable, rows, fmt } from "./common.mjs";

export default function slide09(presentation) {
  const slide = presentation.slides.add();
  const ranked = [...rows]
    .sort((a, b) => (b.overall_tokens_per_second ?? -1) - (a.overall_tokens_per_second ?? -1))
    .slice(0, 16);
  add(slide, [
    ...base(slide, "Appendix: Scenario Detail Matrix"),
    ...miniTable(ranked, [
      { label: "Platform", value: (r) => r.platform, size: 8 },
      { label: "Model", value: (r) => r.model, size: 7 },
      { label: "Scenario", value: (r) => r.scenario, size: 7 },
      { label: "DP", value: (r) => fmt(r.data_parallelism, 0), size: 8 },
      { label: "TP", value: (r) => fmt(r.tensor_parallelism, 0), size: 8 },
      { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}`, size: 8 },
      { label: "Req.", value: (r) => fmt(r.total_requests, 0), size: 8 },
      { label: "In/Out", value: (r) => `${fmt(r.input_tokens, 0)}/${fmt(r.output_tokens, 0)}`, size: 8 },
      { label: "Tok/s", value: (r) => fmt(r.overall_tokens_per_second, 1), size: 8 },
      { label: "Lat P95", value: (r) => `${fmt(r.latency_p95, 1)}s`, size: 8 },
      { label: "Err %", value: (r) => fmt(r.error_rate_percent, 1, "%"), size: 8 },
    ], 44, 118, 1190, 534, "Top throughput scenarios with required run metadata"),
  ]);
  return slide;
}
