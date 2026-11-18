#!/usr/bin/env python3
import os, sys, time, subprocess, argparse, re, csv
import numpy as np
from datetime import datetime
from collections import defaultdict
MAIN_MODELS = [
    "chromaticindex1024-7",
    "ex10",
    "graph40-40",
    "datt256_lp",
    "savsched1"
]
FIXED_ITER_MODELS = {
    "a2864": 500,
    "thk_63": 500,
    "thk_48": 500,
    "L2CTA3D": 500,
    "dlr2": 200
}
DELTAS = [0.0, 0.01]
NLP = 32
REPEATS = 5
CONFIGS = [
    (1,  1, 0, "ctrl_c1_t1"),
    (8,  1, 0, "intra_c8_t1"),
    (8,  2, 0, "intra_c8_t2"),
    (8,  4, 0, "intra_c8_t4"),
    (16, 1, 0, "intra_c16_t1"),
    (16, 2, 0, "intra_c16_t2"),
    (16, 4, 0, "intra_c16_t4"),
    (32, 1, 0, "intra_c32_t1"),
    (32, 2, 0, "intra_c32_t2"),
    (32, 4, 0, "intra_c32_t4"),
    (16, 2, 1, "inter_2x16"),
    (8,  4, 1, "inter_4x8")
]
def parse_args():
    p = argparse.ArgumentParser(description="Batched PDLP Benchmark Runner")
    p.add_argument("--mps-dir", type=str, required=True, help="Directory containing MPS matrix files")
    p.add_argument("--out-dir", type=str, default="results/experiment_run", help="Output directory")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--phase", choices=["all", "fixed_only", "main_only", "test_afiro"], default="all")
    return p.parse_args()
def build_solver():
    os.makedirs("build", exist_ok=True)
    cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-std=c++17", "-Isrc", "bench/solve.cpp", "src/mps.cpp", "-o", "build/solve"]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("[!] Build FAILED!", file=sys.stderr); sys.exit(1)
def find_mps(mps_dir, model):
    for ext in [".mps", ".MPS"]:
        c = os.path.join(mps_dir, model + ext)
        if os.path.exists(c): return c
    for root, _, files in os.walk(mps_dir):
        for f in files:
            if f.lower() == (model + ".mps").lower():
                return os.path.join(root, f)
    return None
def parse_out(out):
    d = {"time_total": None, "time_per_lp": None, "total_lp_iters": None, "ms_per_lp_iter": None,
         "t_setup": None, "t_spmm_main": None, "t_spmm_check": None, "t_spmm_total": None,
         "spmm_pct": None, "t_check_other": None, "t_other": None, "optimal": 0}
    for line in out.splitlines():
        line = line.strip()
        parts = line.split(",")
        if len(parts) >= 9 and parts[4].isdigit() and parts[5] == "OPTIMAL":
            d["optimal"] += 1
        if line.startswith("# time total"):
            m = re.search(r"time total ([\d\.]+) s, per LP ([\d\.]+) s(?:, total_lp_iters (\d+), per LP-iter ([\d\.]+) ms)?", line)
            if m:
                d["time_total"] = float(m.group(1)); d["time_per_lp"] = float(m.group(2))
                if m.group(3): d["total_lp_iters"] = int(m.group(3))
                if m.group(4): d["ms_per_lp_iter"] = float(m.group(4))
        if line.startswith("# breakdown:"):
            m = re.search(r"setup ([\d\.]+) s.*spmm_main ([\d\.]+) s.*spmm_check ([\d\.]+) s.*spmm_total ([\d\.]+) s \(([\d\.]+)%\).*check_other ([\d\.]+) s.*other ([\d\.]+) s", line)
            if m:
                d["t_setup"] = float(m.group(1))
                d["t_spmm_main"] = float(m.group(2))
                d["t_spmm_check"] = float(m.group(3))
                d["t_spmm_total"] = float(m.group(4))
                d["spmm_pct"] = float(m.group(5))
                d["t_check_other"] = float(m.group(6))
                d["t_other"] = float(m.group(7))
    return d
def run_kkt_check(mps_path, delta, nlp, sol_dir):
    sol_args = [f"{os.path.join(sol_dir, f'b{l}.sol')}:{l}" for l in range(nlp)]
    cmd = ["./build/solve", "kkt", mps_path, str(delta), str(nlp), "1e-4"] + sol_args
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok_count = 0
    rel_p_list, rel_d_list, gap_list = [], [], []
    for line in res.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 10 and parts[2] == "OK":
            try:
                rel_p_list.append(float(parts[3]))
                rel_d_list.append(float(parts[4]))
                gap_list.append(float(parts[5]))
                if parts[-1] == "1": ok_count += 1
            except ValueError:
                pass
    return {
        "kkt_ok_count": ok_count,
        "max_rel_p": max(rel_p_list) if rel_p_list else None,
        "max_rel_d": max(rel_d_list) if rel_d_list else None,
        "max_gap": max(gap_list) if gap_list else None
    }
def generate_statistics(summary_csv, out_stat_file):
    data = defaultdict(list)
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            grp = r["group"]
            if grp == "main_exact":
                if r["status"] != "OPTIMAL" or (r["kkt_ok_count"] != "" and int(r["kkt_ok_count"]) != int(r["nlp"])):
                    continue
            elif grp == "fixed_iters":
                if r["status"] not in ["LIMIT", "MAXITER", "OPTIMAL", "OK"]:
                    continue
            key = (grp, r["model"], float(r["delta"]), r["config_label"])
            data[key].append({
                "time": float(r["time_total_s"]),
                "spmm_pct": float(r["spmm_total_pct"]) if r.get("spmm_total_pct") and r["spmm_total_pct"] != "" else None,
                "iter_ms": float(r["ms_per_lp_iter"]) if r.get("ms_per_lp_iter") and r["ms_per_lp_iter"] != "" else None
            })
    ctrl_medians = {}
    for (grp, model, delta, lbl), rows in data.items():
        if lbl in ["ctrl_c1_t1", "ctrl_c1"]:
            ctrl_medians[(grp, model, delta)] = np.median([r["time"] for r in rows])
    with open(out_stat_file, "w", encoding="utf-8") as out:
        out.write("========================================================================================================================\n")
        out.write("EXPERIMENT STATISTICAL SUMMARY (Median, Mean, Std, Min, Max, Speedup)\n")
        out.write("Note: Speedup = Median(ctrl_c1_t1) / Median(Config)\n")
        out.write("      SpMM %  = Percentage of total solver computational work (measured inside C++)\n")
        out.write("========================================================================================================================\n")
        out.write(f"{'Group':<12} {'Model':<22} {'Delta':<6} {'Config':<15} {'N':<4} {'Median(s)':<10} {'Mean(s)':<9} {'Std(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'Speedup':<9} {'SpMM %':<8}\n")
        out.write("-" * 122 + "\n")
        for (grp, model, delta, lbl), rows in sorted(data.items()):
            times = [r["time"] for r in rows]
            med, mean, std = np.median(times), np.mean(times), np.std(times)
            mn, mx = np.min(times), np.max(times)
            ctrl_med = ctrl_medians.get((grp, model, delta))
            spd_str = f"{ctrl_med / med:.2f}x" if ctrl_med and med > 0 else "-"
            
            spmm_vals = [r["spmm_pct"] for r in rows if r["spmm_pct"] is not None]
            spmm_str = f"{np.mean(spmm_vals):.1f}%" if spmm_vals else "-"
            out.write(f"{grp:<12} {model:<22} {delta:<6} {lbl:<15} {len(times):<4} {med:<10.3f} {mean:<9.3f} {std:<8.3f} {mn:<8.3f} {mx:<8.3f} {spd_str:<9} {spmm_str:<8}\n")
    print(f"\n[+] Statistical summary written to: {out_stat_file}")
def main():
    args = parse_args()
    build_solver()
    raw_dir = os.path.join(args.out_dir, "raw_logs")
    sol_dir = os.path.join(args.out_dir, "solutions")
    os.makedirs(raw_dir, exist_ok=True); os.makedirs(sol_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "summary.csv")
    csv_exists = os.path.exists(summary_csv)
    f_csv = open(summary_csv, "a", newline="", encoding="utf-8")
    fields = [
        "timestamp", "group", "model", "delta", "chunk", "threads", "par", "config_label",
        "repeat", "nlp", "status", "time_total_s", "time_per_lp_s", "total_lp_iters",
        "ms_per_lp_iter", "t_setup_s", "t_spmm_main_s", "t_spmm_check_s", "t_spmm_total_s",
        "spmm_total_pct", "t_check_other_s", "t_other_s", "optimal_count", "kkt_ok_count",
        "max_rel_p", "max_rel_d", "max_gap", "log_file"
    ]
    w = csv.DictWriter(f_csv, fieldnames=fields)
    if not csv_exists: w.writeheader(); f_csv.flush()
    tasks = []
    if args.phase == "test_afiro":
        for (ch, th, pr, lbl) in [(1, 1, 0, "ctrl_c1"), (4, 1, 0, "intra_c4")]:
            tasks.append({"group": "test", "model": "afiro", "delta": 0.0, "nlp": 4, "chunk": ch,
                          "threads": th, "par": pr, "label": lbl, "repeat": 1, "maxit": None, "eps": None})
    else:
        if args.phase in ["all", "main_only"]:
            for m in MAIN_MODELS:
                for d in DELTAS:
                    for (ch, th, pr, lbl) in CONFIGS:
                        for r in range(1, args.repeats + 1):
                            tasks.append({"group": "main_exact", "model": m, "delta": d, "nlp": NLP,
                                          "chunk": ch, "threads": th, "par": pr, "label": lbl,
                                          "repeat": r, "maxit": None, "eps": None})
        if args.phase in ["all", "fixed_only"]:
            for m, mx in FIXED_ITER_MODELS.items():
                for d in [0.0]:
                    for (ch, th, pr, lbl) in CONFIGS:
                        for r in range(1, args.repeats + 1):
                            tasks.append({"group": "fixed_iters", "model": m, "delta": d, "nlp": NLP,
                                          "chunk": ch, "threads": th, "par": pr, "label": lbl,
                                          "repeat": r, "maxit": mx, "eps": "0"})
    total = len(tasks)
    print(f"[*] Total {total} tasks planned. (Phase: {args.phase})")
    for i, t in enumerate(tasks, 1):
        tag = f"{t['group']}_{t['model']}_d{t['delta']}_{t['label']}_r{t['repeat']}"
        lpath = os.path.join(raw_dir, f"{tag}.log")
        sdir = os.path.join(sol_dir, tag)
        if os.path.exists(lpath):
            with open(lpath, "r", encoding="utf-8") as lf:
                if "# time total" in lf.read():
                    print(f"[{i}/{total}] SKIPPED (Already completed): {tag}"); continue
        mpath = find_mps(args.mps_dir, t["model"])
        if not mpath:
            print(f"[!] ERROR: {t['model']} not found in ({args.mps_dir})!", file=sys.stderr); continue
        print(f"[{i}/{total}] RUNNING: {t['model']} (d={t['delta']}, {t['label']}, r={t['repeat']})...", flush=True)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(t["threads"])
        os.makedirs(sdir, exist_ok=True)
        cmd = ["./build/solve", "many", mpath, str(t["delta"]), str(t["nlp"]), str(t["chunk"]), f"sol={sdir}"]
        if t["par"]: cmd.append(f"par={t['par']}")
        if t["maxit"]: cmd.append(f"maxit={t['maxit']}")
        if t["eps"]: cmd.append(f"eps={t['eps']}")
        t0 = time.time()
        try:
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=14400)
            with open(lpath, "w", encoding="utf-8") as lf: lf.write(res.stdout)
            p = parse_out(res.stdout)
            kkt_res = run_kkt_check(mpath, t["delta"], t["nlp"], sdir)
            status_str = "OPTIMAL" if p["optimal"] == t["nlp"] else ("PARTIAL_OPTIMAL" if p["optimal"] > 0 else "LIMIT")
            if res.returncode != 0: status_str = f"ERR_{res.returncode}"
            row = {
                "timestamp": datetime.now().isoformat(), "group": t["group"], "model": t["model"],
                "delta": t["delta"], "chunk": t["chunk"], "threads": t["threads"], "par": t["par"],
                "config_label": t["label"], "repeat": t["repeat"], "nlp": t["nlp"], "status": status_str,
                "time_total_s": p["time_total"] if p["time_total"] is not None else time.time() - t0,
                "time_per_lp_s": p["time_per_lp"], "total_lp_iters": p["total_lp_iters"],
                "ms_per_lp_iter": p["ms_per_lp_iter"],
                "t_setup_s": p["t_setup"] if p["t_setup"] is not None else "",
                "t_spmm_main_s": p["t_spmm_main"] if p["t_spmm_main"] is not None else "",
                "t_spmm_check_s": p["t_spmm_check"] if p["t_spmm_check"] is not None else "",
                "t_spmm_total_s": p["t_spmm_total"] if p["t_spmm_total"] is not None else "",
                "spmm_total_pct": p["spmm_pct"] if p["spmm_pct"] is not None else "",
                "t_check_other_s": p["t_check_other"] if p["t_check_other"] is not None else "",
                "t_other_s": p["t_other"] if p["t_other"] is not None else "",
                "optimal_count": p["optimal"],
                "kkt_ok_count": kkt_res["kkt_ok_count"],
                "max_rel_p": kkt_res["max_rel_p"] if kkt_res["max_rel_p"] is not None else "",
                "max_rel_d": kkt_res["max_rel_d"] if kkt_res["max_rel_d"] is not None else "",
                "max_gap": kkt_res["max_gap"] if kkt_res["max_gap"] is not None else "",
                "log_file": lpath
            }
            w.writerow(row); f_csv.flush()
            print(f"    --> DONE: {row['time_total_s']:.2f}s | Opt: {p['optimal']}/{t['nlp']} | KKT: {kkt_res['kkt_ok_count']}/{t['nlp']}", flush=True)
        except subprocess.TimeoutExpired as te:
            print(f"[!] TIMEOUT: {tag} (Exceeded 4 hours)")
            with open(lpath, "w", encoding="utf-8") as lf:
                lf.write(f"TIMEOUT: 14400s\nPartial output:\n{te.stdout if te.stdout else 'None'}\n")
            w.writerow({
                "timestamp": datetime.now().isoformat(), "group": t["group"], "model": t["model"],
                "delta": t["delta"], "chunk": t["chunk"], "threads": t["threads"], "par": t["par"],
                "config_label": t["label"], "repeat": t["repeat"], "nlp": t["nlp"], "status": "TIMEOUT",
                "time_total_s": 14400.0, "log_file": lpath
            })
            f_csv.flush()
        except Exception as e:
            print(f"[!] Error: {str(e)}", file=sys.stderr, flush=True)
    f_csv.close()
    stat_file = os.path.join(args.out_dir, "statistics_summary.txt")
    generate_statistics(summary_csv, stat_file)
    print("\n[+] ALL EXPERIMENTS COMPLETED! Summary: ", stat_file)
if __name__ == "__main__": main()