import { add, base, barChart, miniTable, maasRows, fmt, RED, INK, AMBER, bulletList } from "./common.mjs";

export default function slide08(presentation) {
  const slide = presentation.slides.add();
  const maasThroughput = [...maasRows].filter((r) => r.overall_tokens_per_second !== null).sort((a, b) => b.overall_tokens_per_second - a.overall_tokens_per_second);
  const maasErrors = [...maasRows].sort((a, b) => (b.error_rate_percent ?? 0) - (a.error_rate_percent ?? 0));
  add(slide, [
    ...base(slide, "MaaS Focus: Baselines and Failed Capacity Probes"),
    ...barChart(maasThroughput.slice(0, 7), "overall_tokens_per_second", 54, 120, 540, 470, "MaaS successful throughput", {
      labelFn: (r) => `${r.model} c${r.concurrency} ${fmt(r.input_tokens, 0)}/${fmt(r.output_tokens, 0)}`,
      suffix: " tok/s",
      digits: 1,
      colorFn: () => RED,
    }),
    ...miniTable(maasErrors.slice(0, 7), [
      { label: "Model", value: (r) => r.model, size: 8 },
      { label: "Scenario", value: (r) => r.scenario, size: 8 },
      { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
      { label: "Req.", value: (r) => fmt(r.total_requests, 0) },
      { label: "Success", value: (r) => fmt(r.successful_requests, 0) },
      { label: "Err %", value: (r) => fmt(r.error_rate_percent, 1, "%") },
      { label: "Tok/s", value: (r) => fmt(r.overall_tokens_per_second, 1) },
    ], 638, 120, 560, 330, "MaaS reliability view"),
    ...bulletList([
      "DeepSeek-V3, deepseek-v3.2, and qwen3-32b provide clean 256/256 comparison points when successful.",
      "Several deepseek-v4-flash 1k-input probes failed with connection or rate-limit patterns; they should be treated as operational capacity evidence, not discarded as outliers.",
      "For internal reporting, separate steady-state model baselines from admission-control stress tests."
    ], 660, 492, 500, 130, { size: 15, gap: 48, color: AMBER }),
  ]);
  return slide;
}
