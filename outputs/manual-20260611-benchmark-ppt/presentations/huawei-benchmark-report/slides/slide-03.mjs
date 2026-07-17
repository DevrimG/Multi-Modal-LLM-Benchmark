import { add, base, metricTile, miniTable, rows, deepseekTp16, fmt, RED, INK, GREEN, MUTED, rect, t } from "./common.mjs";

export default function slide03(presentation) {
  const slide = presentation.slides.add();
  const models = new Set(rows.map((r) => `${r.platform}:${r.model}`)).size;
  const conc = rows.map((r) => r.concurrency).filter((v) => v !== null);
  const maxConc = Math.max(...conc);
  const dpKnown = rows.filter((r) => r.data_parallelism !== null).length;
  const tp16 = deepseekTp16[0];
  add(slide, [
    ...base(slide, "Run Inventory and Deployment Topology"),
    ...metricTile(54, 120, 260, 128, "Platform/model combinations", fmt(models, 0), "MaaS and ModelArts grouped by model", RED),
    ...metricTile(342, 120, 260, 128, "Max concurrency observed", `c${fmt(maxConc, 0)}`, "Derived from filename or request overlap", INK),
    ...metricTile(630, 120, 260, 128, "Runs with DP metadata", fmt(dpKnown, 0), "ModelArts filenames and summary rows", GREEN),
    ...metricTile(918, 120, 260, 128, "DeepSeek TP16 topology", "DP1 / 2 nodes", `TP${tp16?.tensor_parallelism ?? 16}, c${tp16?.concurrency ?? "n/a"} and c32`, RED),
    rect(68, 312, 1110, 78, "#ffffff", { line: "#e5e7eb 1px", radius: 4 }),
    t("Normalization rule", 94, 332, 190, 22, { size: 16, bold: true }),
    t("MaaS JSON summaries are consumed directly. ModelArts raw CSVs are aggregated from request rows; summary CSVs take precedence where they encode intended scenario configuration. Missing filename concurrency is inferred from overlapping request windows.", 284, 324, 850, 42, { size: 15, color: MUTED }),
    ...miniTable(deepseekTp16, [
      { label: "Model", value: (r) => r.model },
      { label: "Scenario", value: (r) => r.scenario, size: 8 },
      { label: "DP", value: (r) => fmt(r.data_parallelism, 0) },
      { label: "TP", value: (r) => fmt(r.tensor_parallelism, 0) },
      { label: "Nodes", value: (r) => fmt(r.nodes, 0) },
      { label: "Conc.", value: (r) => `c${fmt(r.concurrency, 0)}` },
      { label: "Req.", value: (r) => fmt(r.total_requests, 0) },
      { label: "In/Out", value: (r) => `${fmt(r.input_tokens, 0)}/${fmt(r.output_tokens, 0)}` },
    ], 68, 430, 1110, 208, "DeepSeek R1 TP16 rows"),
  ]);
  return slide;
}
