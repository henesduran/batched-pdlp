#!/usr/bin/env python3
import argparse, csv, os, re, statistics, subprocess, sys, time
from datetime import datetime

CONFIGS = [  # (chunk, threads, par, label)
    (1, 1, 0, "ctrl_c1_t1"),
    (8, 1, 0, "intra_c8_t1"), (8, 2, 0, "intra_c8_t2"), (8, 4, 0, "intra_c8_t4"),
    (16, 1, 0, "intra_c16_t1"), (16, 2, 0, "intra_c16_t2"), (16, 4, 0, "intra_c16_t4"),
    (32, 1, 0, "intra_c32_t1"), (32, 2, 0, "intra_c32_t2"), (32, 4, 0, "intra_c32_t4"),
    (16, 2, 1, "inter_2x16"), (8, 4, 1, "inter_4x8"),
]
FIELDS = ["timestamp", "node", "model", "delta", "chunk", "threads", "par", "config_label", "repeat", "nlp",
          "maxit", "status", "lp_iters_done", "time_total_s", "total_lp_iters", "ms_per_lp_iter", "t_setup_s",
          "t_spmm_main_s", "t_spmm_check_s", "t_spmm_total_s", "spmm_total_pct", "t_check_other_s", "t_other_s",
          "max_rp", "max_rd", "max_gap", "loadavg_1m", "log_file"]


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--mps-dir", default="models")
    p.add_argument("--models", required=True, help="comma separated")
    p.add_argument("--node", type=int, default=-1, help="socket to pin to; -1 = no numactl")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--maxit", type=int, default=200)
    p.add_argument("--nlp", type=int, default=32)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--src", default="bench/solve.cpp")
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument("--wait", default="", help="file to watch: start only after it contains STOPPED")
    return p.parse_args()


def build(src, exe):
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    cmd = ["g++", "-O3", "-march=native", "-fopenmp", "-std=c++17", "-Isrc", src, "src/mps.cpp", "-o", exe]
    if subprocess.run(cmd).returncode != 0:
        sys.exit("build failed")


def wait_for(marker):
    while True:
        done = os.path.exists(marker) and "STOPPED" in open(marker, encoding="utf-8", errors="replace").read()
        busy = subprocess.run(["pgrep", "-f", "run_experiments.py"], stdout=subprocess.DEVNULL).returncode == 0
        if done and not busy:
            return
        time.sleep(60)


def find_mps(d, m):
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower() == m.lower() + ".mps":
                return os.path.join(root, f)
    return None


def num(rx, s, i=1):
    m = re.search(rx, s)
    return float(m.group(i)) if m else None


def parse(out):
    d = {"opt": 0, "maxit_lps": 0, "rp": [], "rd": [], "gap": [], "iters": 0}
    for line in out.splitlines():
        p = line.strip().split(",")
        if len(p) >= 12 and p[4].isdigit() and p[5] in ("OPTIMAL", "MAXITER"):
            d["opt" if p[5] == "OPTIMAL" else "maxit_lps"] += 1
            d["iters"] += int(p[6])
            for k, j in (("rp", 9), ("rd", 10), ("gap", 11)):
                try:
                    d[k].append(float(p[j]))
                except ValueError:
                    pass
    d["time"] = num(r"# time total ([\d.]+) s", out)
    d["lp_iters"] = num(r"total_lp_iters (\d+)", out)
    d["ms"] = num(r"per LP-iter ([\d.]+) ms", out)
    for k, rx in (("setup", r"setup ([\d.]+) s"), ("spmm_main", r"spmm_main ([\d.]+) s"),
                  ("spmm_check", r"spmm_check ([\d.]+) s"), ("spmm_total", r"spmm_total ([\d.]+) s"),
                  ("check_other", r"check_other ([\d.]+) s"), ("other", r" other ([\d.]+) s")):
        d[k] = num(rx, out)
    d["spmm_pct"] = num(r"spmm_total [\d.]+ s \(([\d.]+)%\)", out)
    return d


def blank(v):
    return "" if v is None else v


def summary(csv_path, out_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    cells = {}
    for r in rows:
        if r["status"] not in ("OK",):
            continue
        cells.setdefault((r["model"], r["config_label"]), []).append(r)
    ctrl = {m: statistics.median(float(r["time_total_s"]) for r in v)
            for (m, lbl), v in cells.items() if lbl == "ctrl_c1_t1"}
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("model,config,n,median_s,min_s,max_s,speedup_vs_ctrl,median_setup_s,median_spmm_pct,"
                "median_ms_per_lp_iter,max_rp,max_rd,max_gap\n")
        for (m, lbl), v in sorted(cells.items()):
            t = [float(r["time_total_s"]) for r in v]
            med = statistics.median(t)
            g = lambda k: statistics.median(float(r[k]) for r in v if r[k] != "")
            mx = lambda k: max((float(r[k]) for r in v if r[k] != ""), default="")
            sp = f"{ctrl[m] / med:.3f}" if m in ctrl and med > 0 else ""
            f.write(f"{m},{lbl},{len(t)},{med:.3f},{min(t):.3f},{max(t):.3f},{sp},{g('t_setup_s'):.3f},"
                    f"{g('spmm_total_pct'):.1f},{g('ms_per_lp_iter'):.4f},{mx('max_rp')},{mx('max_rd')},{mx('max_gap')}\n")


def main():
    a = args_()
    tag = f"n{a.node}" if a.node >= 0 else "nonuma"
    exe = f"build/solve_fixed_{tag}"
    build(a.src, exe)
    print(f"[{datetime.now():%H:%M:%S}] built {exe}", flush=True)
    if a.wait:
        print(f"[{datetime.now():%H:%M:%S}] waiting for STOPPED in {a.wait}", flush=True)
        wait_for(a.wait)
    raw = os.path.join(a.out_dir, "raw_logs")
    os.makedirs(raw, exist_ok=True)
    csv_path = os.path.join(a.out_dir, "summary.csv")
    new = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if new:
        w.writeheader()
        fcsv.flush()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    paths = {}
    for m in models:
        paths[m] = find_mps(a.mps_dir, m)
        if not paths[m]:
            sys.exit(f"missing {m}.mps in {a.mps_dir}")
    tasks = [(r, m, c) for r in range(1, a.repeats + 1) for m in models for c in CONFIGS]
    numa = ["numactl", f"--cpunodebind={a.node}", f"--membind={a.node}"] if a.node >= 0 else []
    for i, (rep, m, (ch, th, par, lbl)) in enumerate(tasks, 1):
        name = f"{m}_d{a.delta}_{lbl}_r{rep}"
        log = os.path.join(raw, name + ".log")
        if os.path.exists(log) and "# time total" in open(log, encoding="utf-8", errors="replace").read():
            print(f"[{i}/{len(tasks)}] skip {name}", flush=True)
            continue
        env = dict(os.environ, OMP_NUM_THREADS=str(th), OMP_PROC_BIND="close", OMP_PLACES="cores")
        cmd = numa + [exe, "many", paths[m], str(a.delta), str(a.nlp), str(ch), f"maxit={a.maxit}", "eps=0"]
        if par:
            cmd.append("par=1")
        load = open("/proc/loadavg").read().split()[0]
        print(f"[{i}/{len(tasks)}] {datetime.now():%m-%d %H:%M:%S} {name} (load {load})", flush=True)
        t0 = time.time()
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "node": a.node, "model": m,
               "delta": a.delta, "chunk": ch, "threads": th, "par": par, "config_label": lbl, "repeat": rep,
               "nlp": a.nlp, "maxit": a.maxit, "loadavg_1m": load, "log_file": log}
        try:
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 timeout=a.timeout)
            open(log, "w", encoding="utf-8").write(res.stdout)
            d = parse(res.stdout)
            ok = res.returncode == 0 and d["time"] is not None and d["opt"] + d["maxit_lps"] == a.nlp
            row.update({"status": "OK" if ok else f"ERR_{res.returncode}", "lp_iters_done": d["iters"],
                        "time_total_s": blank(d["time"]), "total_lp_iters": blank(d["lp_iters"]),
                        "ms_per_lp_iter": blank(d["ms"]), "t_setup_s": blank(d["setup"]),
                        "t_spmm_main_s": blank(d["spmm_main"]), "t_spmm_check_s": blank(d["spmm_check"]),
                        "t_spmm_total_s": blank(d["spmm_total"]), "spmm_total_pct": blank(d["spmm_pct"]),
                        "t_check_other_s": blank(d["check_other"]), "t_other_s": blank(d["other"]),
                        "max_rp": max(d["rp"], default=""), "max_rd": max(d["rd"], default=""),
                        "max_gap": max(d["gap"], default="")})
            print(f"    {row['status']} {d['time']} s, iters {d['iters']}, max rp {row['max_rp']} "
                  f"rd {row['max_rd']} gap {row['max_gap']}", flush=True)
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
            open(log, "w", encoding="utf-8").write(f"TIMEOUT {a.timeout} s\n{out}")
            row.update({"status": "TIMEOUT", "time_total_s": round(time.time() - t0, 1)})
            print(f"    TIMEOUT after {a.timeout} s", flush=True)
        w.writerow(row)
        fcsv.flush()
    fcsv.close()
    summary(csv_path, os.path.join(a.out_dir, "stats.csv"))
    print(f"[{datetime.now():%H:%M:%S}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
