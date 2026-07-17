#!/usr/bin/env python3
import csv
import datetime as dt
import html
import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks"
OUT = ROOT / "outputs" / "benchmark_report"
OUT.mkdir(parents=True, exist_ok=True)

HUAWEI_RED = "#c7000b"
INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
GOOD = "#16883c"
WARN = "#f59e0b"
BAD = "#c7000b"


def num(value, default=None):
    if value in (None, "", "none", "null"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def intish(value, default=None):
    value = num(value, default)
    if value is None:
        return default
    return int(round(value))


def percentile(values, pct):
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def fmt(value, digits=2, suffix=""):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return html.escape(value)
    if abs(value) >= 100:
        text = f"{value:,.0f}"
    elif abs(value) >= 10:
        text = f"{value:,.1f}"
    else:
        text = f"{value:,.{digits}f}"
    return text + suffix


def infer_tokens(name):
    lower = name.lower()
    in_tok = out_tok = None
    m = re.search(r"(\d+)k(?:_|-)?in(?:put)?", lower)
    if m:
        in_tok = int(m.group(1)) * 1000
    m = re.search(r"(\d+)k(?:_|-)?out(?:put)?", lower)
    if m:
        out_tok = int(m.group(1)) * 1000
    m = re.search(r"(\d+)kinput", lower)
    if m:
        in_tok = int(m.group(1)) * 1000
    m = re.search(r"(\d+)output", lower)
    if m:
        out_tok = int(m.group(1))
    return in_tok, out_tok


def infer_concurrency(name):
    lower = name.lower()
    for pat in [r"(\d+)conc", r"(\d+)c(?:\.csv|_|-)", r"_(\d+)c(?:\.csv|_|-)"]:
        m = re.search(pat, lower)
        if m:
            return int(m.group(1))
    m = re.search(r"(\d+)-concurrency", lower)
    return int(m.group(1)) if m else None


def infer_modelarts_meta(path):
    name = path.name
    base = path.stem
    dp = None
    tp = None
    nodes = None
    m = re.match(r"DP-(\d+)-(.+)", base, flags=re.I)
    if m:
        dp = int(m.group(1))
        rest = m.group(2)
    else:
        m = re.match(r"TP-(\d+)-(.+)", base, flags=re.I)
        if m:
            tp = int(m.group(1))
            rest = m.group(2)
            if tp == 16 and "deepseek" in rest.lower():
                dp = 1
                nodes = 2
        else:
            m = re.match(r"(\d+)P-(\d+)D-(.+)", base, flags=re.I)
            if m:
                nodes = int(m.group(1))
                dp = int(m.group(2))
                rest = m.group(3)
            else:
                rest = base
    if rest.lower().startswith("deepseek_r1"):
        model = "deepseek-r1"
        scenario = rest[len("deepseek_r1_"):] if rest.lower().startswith("deepseek_r1_") else rest
    else:
        parts = re.split(r"_benchmark_|-benchmark-", rest, maxsplit=1)
        model = parts[0].replace("_", "-")
        scenario = parts[1] if len(parts) > 1 else rest
    scenario = re.sub(r"\.csv$", "", scenario)
    return model, scenario, dp, tp, nodes


def observed_concurrency(starts, ends):
    events = []
    for start, end in zip(starts, ends):
        if start is None or end is None:
            continue
        events.append((start, 1))
        events.append((end, -1))
    current = peak = 0
    for _, delta in sorted(events, key=lambda x: (x[0], -x[1])):
        current += delta
        peak = max(peak, current)
    return peak or None


def summarize_raw_csv(path):
    model, scenario, dp, tp, nodes = infer_modelarts_meta(path)
    with path.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    successes = []
    latencies = []
    ttfts = []
    tpots = []
    tps = []
    input_tokens = []
    output_tokens = []
    starts = []
    ends = []
    for row in rows:
        status = (row.get("status_code") or "").strip()
        err = (row.get("error") or "").strip().lower()
        ok = (err in ("", "none", "false", "0")) and (status in ("", "200"))
        successes.append(ok)
        starts.append(num(row.get("start_time")))
        ends.append(num(row.get("end_time")))
        if ok:
            latencies.append(num(row.get("client_observed_latency_seconds"), num(row.get("total_latency_seconds"))))
            ttfts.append(num(row.get("client_observed_ttft_seconds"), num(row.get("ttft_seconds"))))
            tpots.append(num(row.get("tpot_seconds")))
            tps.append(num(row.get("tokens_per_second")))
            output_tokens.append(num(row.get("tokens_generated")))
        input_tokens.append(num(row.get("input_tokens")))
    duration = None
    clean_starts = [v for v in starts if v is not None]
    clean_ends = [v for v in ends if v is not None]
    if clean_starts and clean_ends:
        duration = max(clean_ends) - min(clean_starts)
    inferred_in, inferred_out = infer_tokens(path.name)
    return {
        "platform": "ModelArts",
        "source_file": str(path.relative_to(ROOT)),
        "model": model,
        "scenario": scenario,
        "data_parallelism": dp,
        "tensor_parallelism": tp,
        "nodes": nodes,
        "concurrency": infer_concurrency(path.name) or observed_concurrency(starts, ends),
        "total_requests": len(rows),
        "actual_requests": len(rows),
        "successful_requests": sum(1 for ok in successes if ok),
        "failed_requests": sum(1 for ok in successes if not ok),
        "error_rate_percent": 100 * sum(1 for ok in successes if not ok) / len(rows),
        "duration_seconds": duration,
        "input_tokens": inferred_in or intish(percentile(input_tokens, 50)),
        "output_tokens": inferred_out or intish(percentile(output_tokens, 50)),
        "total_input_tokens": sum(v for v in input_tokens if v is not None),
        "total_output_tokens": sum(v for v in output_tokens if v is not None),
        "overall_tokens_per_second": (sum(v for v in output_tokens if v is not None) / duration) if duration else None,
        "ttft_mean": mean([v for v in ttfts if v is not None]) if any(v is not None for v in ttfts) else None,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p95": percentile(ttfts, 95),
        "ttft_p99": percentile(ttfts, 99),
        "latency_mean": mean([v for v in latencies if v is not None]) if any(v is not None for v in latencies) else None,
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "latency_p99": percentile(latencies, 99),
        "tpot_mean": mean([v for v in tpots if v is not None]) if any(v is not None for v in tpots) else None,
        "per_request_tokens_per_second_mean": mean([v for v in tps if v is not None]) if any(v is not None for v in tps) else None,
        "latency_samples": [v for v in latencies if v is not None][:240],
        "ttft_samples": [v for v in ttfts if v is not None][:240],
    }


def summarize_summary_csv(path):
    out = []
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            model, scenario, dp, tp, nodes = infer_modelarts_meta(Path(row.get("csv") or path.name))
            model = row.get("model") or model
            scenario = row.get("scenario") or scenario
            out.append({
                "platform": "ModelArts",
                "source_file": str(path.relative_to(ROOT)),
                "model": model,
                "scenario": scenario,
                "data_parallelism": dp,
                "tensor_parallelism": tp,
                "nodes": nodes,
                "concurrency": intish(row.get("concurrency")),
                "total_requests": intish(row.get("total_requests"), intish(row.get("actual_requests"))),
                "actual_requests": intish(row.get("actual_requests"), intish(row.get("total_requests"))),
                "successful_requests": intish(row.get("successful_requests")),
                "failed_requests": intish(row.get("failed_requests")),
                "error_rate_percent": num(row.get("error_rate_percent")),
                "duration_seconds": num(row.get("duration_seconds")),
                "input_tokens": intish(row.get("input_tokens")),
                "output_tokens": intish(row.get("output_tokens")),
                "total_input_tokens": num(row.get("total_input_tokens")),
                "total_output_tokens": num(row.get("total_output_tokens")),
                "overall_tokens_per_second": num(row.get("overall_tokens_per_second")),
                "ttft_mean": num(row.get("client_observed_ttft_mean"), num(row.get("ttft_mean"))),
                "ttft_p50": num(row.get("client_observed_ttft_p50"), num(row.get("ttft_p50"))),
                "ttft_p95": num(row.get("client_observed_ttft_p95"), num(row.get("ttft_p95"))),
                "ttft_p99": num(row.get("client_observed_ttft_p99"), num(row.get("ttft_p99"))),
                "latency_mean": num(row.get("latency_mean")),
                "latency_p50": num(row.get("latency_p50")),
                "latency_p95": num(row.get("latency_p95")),
                "latency_p99": num(row.get("latency_p99")),
                "tpot_mean": num(row.get("tpot_mean")),
                "per_request_tokens_per_second_mean": num(row.get("per_request_tokens_per_second_mean")),
                "latency_samples": [],
                "ttft_samples": [],
            })
    return out


def summarize_modelarts_summary_json(path):
    data = json.loads(path.read_text())
    s = data.get("summary", data)
    if not isinstance(s, dict) or "model" not in s:
        return []
    model, scenario, dp, tp, nodes = infer_modelarts_meta(path)
    return [{
        "platform": "ModelArts",
        "source_file": str(path.relative_to(ROOT)),
        "model": s.get("model") or model,
        "scenario": scenario.replace(".summary", ""),
        "data_parallelism": dp,
        "tensor_parallelism": tp,
        "nodes": nodes,
        "concurrency": intish(s.get("concurrency")),
        "total_requests": intish(s.get("total_requests")),
        "actual_requests": intish(s.get("actual_requests"), intish(s.get("total_requests"))),
        "successful_requests": intish(s.get("successful_requests")),
        "failed_requests": intish(s.get("failed_requests")),
        "error_rate_percent": num(s.get("error_rate_percent")),
        "duration_seconds": num(s.get("duration_seconds")),
        "input_tokens": intish(s.get("input_tokens")),
        "output_tokens": intish(s.get("output_tokens")),
        "total_input_tokens": num(s.get("total_input_tokens")),
        "total_output_tokens": num(s.get("total_output_tokens")),
        "overall_tokens_per_second": num(s.get("overall_tokens_per_second")),
        "ttft_mean": num(s.get("client_observed_ttft_mean"), num(s.get("ttft_mean"))),
        "ttft_p50": num(s.get("client_observed_ttft_p50"), num(s.get("ttft_p50"))),
        "ttft_p95": num(s.get("client_observed_ttft_p95"), num(s.get("ttft_p95"))),
        "ttft_p99": num(s.get("client_observed_ttft_p99"), num(s.get("ttft_p99"))),
        "latency_mean": num(s.get("latency_mean")),
        "latency_p50": num(s.get("latency_p50")),
        "latency_p95": num(s.get("latency_p95")),
        "latency_p99": num(s.get("latency_p99")),
        "tpot_mean": num(s.get("tpot_mean")),
        "per_request_tokens_per_second_mean": num(s.get("per_request_tokens_per_second_mean")),
        "latency_samples": [num(x.get("total_latency_seconds")) for x in data.get("raw_metrics", []) if num(x.get("total_latency_seconds")) is not None][:240],
        "ttft_samples": [num(x.get("ttft_seconds")) for x in data.get("raw_metrics", []) if num(x.get("ttft_seconds")) is not None][:240],
    }]


def summarize_maas_json(path):
    data = json.loads(path.read_text())
    if "summary" not in data:
        return []
    s = data["summary"]
    return [{
        "platform": "MaaS",
        "source_file": str(path.relative_to(ROOT)),
        "model": s.get("model"),
        "scenario": f"{intish(s.get('input_tokens')) or 'n/a'} in / {intish(s.get('output_tokens')) or 'n/a'} out",
        "data_parallelism": None,
        "tensor_parallelism": None,
        "nodes": None,
        "concurrency": intish(s.get("concurrency")),
        "total_requests": intish(s.get("total_requests")),
        "actual_requests": intish(s.get("actual_requests")),
        "successful_requests": intish(s.get("successful_requests")),
        "failed_requests": intish(s.get("failed_requests")),
        "error_rate_percent": num(s.get("error_rate_percent")),
        "duration_seconds": num(s.get("duration_seconds")),
        "input_tokens": intish(s.get("input_tokens")),
        "output_tokens": intish(s.get("output_tokens")),
        "total_input_tokens": num(s.get("total_input_tokens")),
        "total_output_tokens": num(s.get("total_output_tokens")),
        "overall_tokens_per_second": num(s.get("overall_tokens_per_second")),
        "ttft_mean": num(s.get("ttft_mean")),
        "ttft_p50": num(s.get("ttft_p50")),
        "ttft_p95": num(s.get("ttft_p95")),
        "ttft_p99": num(s.get("ttft_p99")),
        "latency_mean": num(s.get("latency_mean")),
        "latency_p50": num(s.get("latency_p50")),
        "latency_p95": num(s.get("latency_p95")),
        "latency_p99": num(s.get("latency_p99")),
        "tpot_mean": num(s.get("tpot_mean")),
        "per_request_tokens_per_second_mean": num(s.get("per_request_tokens_per_second_mean")),
        "latency_samples": [num(x.get("total_latency_seconds")) for x in data.get("raw_metrics", []) if num(x.get("total_latency_seconds")) is not None][:240],
        "ttft_samples": [num(x.get("ttft_seconds")) for x in data.get("raw_metrics", []) if num(x.get("ttft_seconds")) is not None][:240],
    }]


def dedupe(rows):
    preferred = {}
    for r in rows:
        key = (r["platform"], r["model"], r["scenario"], r["data_parallelism"], r["tensor_parallelism"], r["concurrency"], r["input_tokens"], r["output_tokens"])
        score = 0
        if "summary" in r["source_file"]:
            score += 10
        score += int(r.get("actual_requests") or 0)
        if key not in preferred or score > preferred[key][0]:
            preferred[key] = (score, r)
    return [v[1] for v in preferred.values()]


def bar_chart(rows, metric, title, subtitle="", width=920, height=360):
    data = [r for r in rows if r.get(metric) is not None]
    data = sorted(data, key=lambda r: r.get(metric) or 0, reverse=True)[:16]
    if not data:
        return f"<div class='empty'>No data for {html.escape(title)}</div>"
    ml, mr, mt, mb = 210, 30, 40, 48
    plot_w, plot_h = width - ml - mr, height - mt - mb
    mx = max(r[metric] for r in data) or 1
    step = plot_h / max(len(data), 1)
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img' aria-label='{html.escape(title)}'>",
             f"<text x='0' y='18' class='chart-title'>{html.escape(title)}</text>",
             f"<text x='0' y='36' class='chart-sub'>{html.escape(subtitle)}</text>"]
    for i in range(5):
        x = ml + plot_w * i / 4
        parts.append(f"<line x1='{x:.1f}' y1='{mt}' x2='{x:.1f}' y2='{mt+plot_h}' stroke='{GRID}'/>")
        parts.append(f"<text x='{x:.1f}' y='{height-14}' text-anchor='middle' class='axis'>{fmt(mx*i/4,1)}</text>")
    for idx, r in enumerate(data):
        y = mt + idx * step + step * .2
        h = max(step * .56, 9)
        val = r[metric]
        w = plot_w * val / mx
        label = f"{r['platform']} | {r['model']} | {r['scenario']} | c{r.get('concurrency') or 'n/a'}"
        color = BAD if metric == "error_rate_percent" else HUAWEI_RED
        parts.append(f"<text x='{ml-8}' y='{y+h*.72:.1f}' text-anchor='end' class='ylabel'>{html.escape(label[:42])}</text>")
        parts.append(f"<rect x='{ml}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='2' fill='{color}'/>")
        parts.append(f"<text x='{ml+w+6:.1f}' y='{y+h*.72:.1f}' class='value'>{fmt(val,2)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def line_chart(rows, metric, title, width=920, height=340):
    groups = {}
    for r in rows:
        if r.get(metric) is None or r.get("concurrency") is None:
            continue
        key = f"{r['platform']} | {r['model']} | {r['scenario']} | DP{r.get('data_parallelism') or 'n/a'}"
        groups.setdefault(key, []).append((r["concurrency"], r[metric]))
    series = [(k, sorted(v)) for k, v in groups.items() if len(v) > 1]
    series = sorted(series, key=lambda kv: max(y for _, y in kv[1]), reverse=True)[:8]
    if not series:
        return f"<div class='empty'>No multi-concurrency data for {html.escape(title)}</div>"
    xs = [x for _, vals in series for x, _ in vals]
    ys = [y for _, vals in series for _, y in vals]
    ml, mr, mt, mb = 62, 210, 44, 48
    plot_w, plot_h = width - ml - mr, height - mt - mb
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = 0, max(ys) or 1
    def px(x):
        return ml + ((x - xmin) / (xmax - xmin or 1)) * plot_w
    def py(y):
        return mt + plot_h - ((y - ymin) / (ymax - ymin or 1)) * plot_h
    colors = [HUAWEI_RED, "#111827", "#6b7280", "#f59e0b", "#16883c", "#2563eb", "#9333ea", "#0f766e"]
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart'><text x='0' y='18' class='chart-title'>{html.escape(title)}</text>"]
    for i in range(5):
        y = mt + plot_h*i/4
        parts.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{ml+plot_w}' y2='{y:.1f}' stroke='{GRID}'/>")
        parts.append(f"<text x='{ml-10}' y='{y+4:.1f}' text-anchor='end' class='axis'>{fmt(ymax*(1-i/4),1)}</text>")
    parts.append(f"<text x='{ml+plot_w/2}' y='{height-12}' text-anchor='middle' class='axis'>Concurrency</text>")
    for i, (label, vals) in enumerate(series):
        color = colors[i % len(colors)]
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in vals)
        parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for x, y in vals:
            parts.append(f"<circle cx='{px(x):.1f}' cy='{py(y):.1f}' r='4' fill='{color}'/>")
        ly = mt + i * 26
        parts.append(f"<rect x='{width-mr+18}' y='{ly-9}' width='12' height='12' fill='{color}'/>")
        parts.append(f"<text x='{width-mr+36}' y='{ly+2}' class='legend'>{html.escape(label[:31])}</text>")
    parts.append("</svg>")
    return "".join(parts)


def spark(samples, color=HUAWEI_RED):
    samples = [s for s in samples if s is not None]
    if not samples:
        return "<span class='muted'>n/a</span>"
    width, height = 150, 36
    mx = max(samples) or 1
    step = width / max(len(samples) - 1, 1)
    pts = " ".join(f"{i*step:.1f},{height - (v/mx)*(height-4) - 2:.1f}" for i, v in enumerate(samples))
    return f"<svg viewBox='0 0 {width} {height}' class='spark'><polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.8'/></svg>"


def render_table(rows):
    cols = [
        ("Platform", "platform"), ("Model", "model"), ("Scenario", "scenario"), ("DP", "data_parallelism"),
        ("TP", "tensor_parallelism"), ("Nodes", "nodes"), ("Conc.", "concurrency"), ("Req.", "total_requests"),
        ("In Tok", "input_tokens"), ("Out Tok", "output_tokens"), ("Success", "successful_requests"),
        ("Err %", "error_rate_percent"), ("TTFT P95", "ttft_p95"), ("Latency P95", "latency_p95"),
        ("Tok/s", "overall_tokens_per_second"), ("Latency Trace", "latency_samples"),
    ]
    body = []
    for r in sorted(rows, key=lambda x: (x["platform"], str(x["model"]), str(x["scenario"]), x.get("concurrency") or 0)):
        tds = []
        for label, key in cols:
            if key == "latency_samples":
                val = spark(r.get(key) or [])
            elif key in ("error_rate_percent",):
                val = fmt(r.get(key), 1, "%")
            elif key in ("ttft_p95", "latency_p95"):
                val = fmt(r.get(key), 2, "s")
            else:
                val = fmt(r.get(key), 2)
            tds.append(f"<td>{val}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    head = "".join(f"<th>{label}</th>" for label, _ in cols)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def collect():
    rows = []
    for path in (BENCH / "HWC MaaS").glob("*.json"):
        if "benchmark_report" in path.name or "full_report" in path.name:
            continue
        rows.extend(summarize_maas_json(path))
    summarized = set()
    for path in (BENCH / "ModelArts").glob("*summary*.csv"):
        with path.open(newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                ref = row.get("csv")
                if ref:
                    summarized.add(Path(ref).name)
                else:
                    prefix = path.name.replace("_benchmark_summary.csv", "_benchmark_")
                    scenario = row.get("scenario")
                    if scenario:
                        summarized.add(f"{prefix}{scenario}.csv")
                        summarized.add(f"{prefix}{scenario}.json")
        rows.extend(summarize_summary_csv(path))
    for path in (BENCH / "ModelArts").glob("*.summary.json"):
        summarized.add(path.name.replace(".summary.json", ".csv"))
        rows.extend(summarize_modelarts_summary_json(path))
    for path in (BENCH / "ModelArts").glob("*.json"):
        if path.name.endswith(".summary.json"):
            continue
        if path.name in summarized:
            continue
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "summary" in data:
            summarized.add(path.name.replace(".json", ".csv"))
            rows.extend(summarize_modelarts_summary_json(path))
    for path in (BENCH / "ModelArts").glob("*.csv"):
        if "summary" in path.name:
            continue
        if path.name in summarized:
            continue
        else:
            r = summarize_raw_csv(path)
            if r:
                rows.append(r)
    return dedupe(rows)


def write_csv(rows):
    fields = [k for k in rows[0].keys() if k not in ("latency_samples", "ttft_samples")]
    path = OUT / "normalized_benchmark_runs.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})
    return path


def main():
    rows = collect()
    csv_path = write_csv(rows)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_runs = len(rows)
    total_reqs = sum(r.get("total_requests") or 0 for r in rows)
    avg_error = mean([r["error_rate_percent"] for r in rows if r.get("error_rate_percent") is not None])
    best = max([r for r in rows if r.get("overall_tokens_per_second") is not None], key=lambda r: r["overall_tokens_per_second"])
    style = f"""
    :root {{ --red:{HUAWEI_RED}; --ink:{INK}; --muted:{MUTED}; --grid:{GRID}; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; font:14px/1.45 Arial, Helvetica, sans-serif; color:var(--ink); background:#f6f7f9; }}
    .hero {{ background:linear-gradient(105deg,#111827 0%,#111827 62%,#c7000b 62%,#c7000b 100%); color:white; padding:42px 54px 34px; }}
    .mark {{ font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:12px; opacity:.86; }}
    h1 {{ margin:14px 0 8px; font-size:36px; line-height:1.1; letter-spacing:0; }}
    .hero p {{ max-width:850px; color:#e5e7eb; font-size:16px; }}
    main {{ padding:28px 42px 60px; max-width:1280px; margin:0 auto; }}
    section {{ background:white; border:1px solid #e5e7eb; border-radius:8px; padding:22px; margin:18px 0; box-shadow:0 1px 2px rgba(17,24,39,.05); }}
    h2 {{ margin:0 0 16px; font-size:22px; }} h3 {{ margin:18px 0 10px; font-size:16px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
    .kpi {{ border-left:4px solid var(--red); padding:14px; background:#fafafa; border-radius:6px; }}
    .kpi b {{ display:block; font-size:25px; }} .kpi span,.muted {{ color:var(--muted); }}
    .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }} .full {{ grid-column:1 / -1; }}
    .chart {{ width:100%; height:auto; background:#fff; }} .chart-title {{ font-weight:700; font-size:16px; fill:var(--ink); }}
    .chart-sub,.axis,.ylabel,.legend {{ fill:var(--muted); font-size:11px; }} .value {{ fill:var(--ink); font-size:11px; font-weight:700; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }} th {{ text-align:left; background:#111827; color:white; position:sticky; top:0; }}
    th,td {{ border-bottom:1px solid #e5e7eb; padding:7px 8px; vertical-align:middle; }} tbody tr:nth-child(even) {{ background:#fafafa; }}
    .tablewrap {{ max-height:720px; overflow:auto; border:1px solid #e5e7eb; border-radius:6px; }}
    .spark {{ width:150px; height:36px; }} .empty {{ color:var(--muted); padding:34px; border:1px dashed #d1d5db; border-radius:6px; }}
    .note {{ border-left:4px solid var(--red); padding:10px 14px; background:#fff7f7; color:#3f3f46; }}
    @media print {{ body {{ background:white; }} section {{ break-inside:avoid; box-shadow:none; }} .hero {{ print-color-adjust:exact; }} }}
    """
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Huawei Internal Benchmark Report</title><style>{style}</style></head>
<body>
<header class="hero"><div class="mark">Huawei Cloud Internal Benchmark Report</div>
<h1>MaaS and ModelArts LLM Throughput, Latency, and Reliability</h1>
<p>Generated {html.escape(now)} from benchmark artifacts in <code>benchmarks/HWC MaaS</code> and <code>benchmarks/ModelArts</code>. The layout follows Huawei-inspired principles: direct information hierarchy, high contrast, restrained red accenting, and operationally dense tables.</p></header>
<main>
<section><h2>Executive Summary</h2><div class="kpis">
<div class="kpi"><span>Normalized benchmark runs</span><b>{total_runs}</b></div>
<div class="kpi"><span>Total requests represented</span><b>{total_reqs:,}</b></div>
<div class="kpi"><span>Average error rate</span><b>{fmt(avg_error,1,'%')}</b></div>
<div class="kpi"><span>Highest aggregate throughput</span><b>{fmt(best['overall_tokens_per_second'],1)} tok/s</b><span>{html.escape(best['platform'])} | {html.escape(str(best['model']))}</span></div>
</div>
<p class="note">DeepSeek TP16 on ModelArts is annotated as DP1 across two nodes, per the provided deployment note. MaaS files do not expose DP/TP topology, so those fields are intentionally reported as n/a.</p></section>
<section><h2>Cross-Scenario Graphs</h2><div class="grid2">
<div>{bar_chart(rows, 'overall_tokens_per_second', 'Top Aggregate Output Throughput', 'Higher is better; tok/s computed from successful generated output tokens over wall-clock duration')}</div>
<div>{bar_chart(rows, 'latency_p95', 'Highest P95 Latency Scenarios', 'Seconds; highlights tail latency pressure')}</div>
<div>{bar_chart(rows, 'ttft_p95', 'Highest P95 Time To First Token', 'Seconds; first-token responsiveness')}</div>
<div>{bar_chart(rows, 'error_rate_percent', 'Highest Error Rate Scenarios', 'Percent failed requests')}</div>
<div class="full">{line_chart(rows, 'overall_tokens_per_second', 'Throughput Scaling by Concurrency')}</div>
<div class="full">{line_chart(rows, 'latency_p95', 'P95 Latency Scaling by Concurrency')}</div>
</div></section>
<section><h2>Scenario Detail Matrix</h2><p class="muted">Each row includes concurrency, model name, data parallelism, request count, and requested input/output token sizes where available. Inline traces show per-request latency shape for raw files.</p><div class="tablewrap">{render_table(rows)}</div></section>
<section><h2>Methodology and Caveats</h2>
<p>JSON MaaS summaries are read directly from each run file. ModelArts raw CSVs are aggregated from per-request records; scenario summary CSVs are used where present because they encode the intended request/token configuration. Duplicate scenario keys are collapsed toward summary files and larger request counts.</p>
<p>Throughput is aggregate output tokens per second. Error rates include non-200 statuses and explicit error fields. Some file names expose only input tokens; output token targets are reported as n/a when neither the summary nor filename provides them.</p>
<p>Normalized data: <code>{html.escape(str(csv_path.relative_to(ROOT)))}</code></p>
</section>
</main></body></html>"""
    report_path = OUT / "huawei_internal_benchmark_report.html"
    report_path.write_text(html_doc)
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
