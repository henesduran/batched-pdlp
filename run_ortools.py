#!/usr/bin/env python3
import argparse, csv, os, re, statistics, subprocess, sys, time
from datetime import datetime

FIELDS = ["timestamp", "node", "model", "threads", "repeat", "maxit", "status", "termination", "iters",
          "t_iter0_s", "t_final_s", "ms_per_iter", "wall_s", "wall_1it_s", "ms_per_iter_wall", "rel_p_res", "rel_d_res", "rel_gap",
          "prim_obj", "dual_obj", "loadavg_1m", "log_file"]
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+\|\s+(\S+)\s+(\S+)\s+(\S+)\s+\|\s+(\S+)\s+(\S+)\s*$")


def args_():
    p = argparse.ArgumentParser()
    p.add_argument("--mps-dir", default="models")
    p.add_argument("--models", required=True, help="comma separated")
    p.add_argument("--threads", default="1,2,4")
    p.add_argument("--node", type=int, default=-1, help="socket to pin to; -1 = no numactl")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--maxit", type=int, default=200, help="0 = solve to 1e-4 with OR-Tools defaults")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--exe", default=os.path.expanduser("~/pdlp_work/solvers/ortools_cpp/bin/solve"))
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument("--wait", default="", help="comma separated files; start when all contain ALL DONE")
    return p.parse_args()


def wait_for(files):
    while True:
        if all(os.path.exists(f) and "ALL DONE" in open(f, encoding="utf-8", errors="replace").read()
               for f in files):
            return
        time.sleep(60)


def find_mps(d, m):
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower() == m.lower() + ".mps":
                return os.path.join(root, f)
    return None


def params(maxit, th):
    if maxit == 0: 
        return ("termination_criteria { simple_optimality_criteria { eps_optimal_absolute: 1e-4 "
                "eps_optimal_relative: 1e-4 } } verbosity_level: 2 num_threads: %d" % th)
    return ("termination_criteria { iteration_limit: %d eps_primal_infeasible: 0 eps_dual_infeasible: 0 "
            "simple_optimality_criteria { eps_optimal_absolute: 0 eps_optimal_relative: 0 } } "
            "linesearch_rule: CONSTANT_STEP_SIZE_RULE termination_check_frequency: 200 "
            "verbosity_level: 2 num_threads: %d" % (maxit, th))


def parse(out):
    d = {"term": "", "first": None, "final": None, "wall": None}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        m = ROW.match(line)
        if m and d["first"] is None:
            d["first"] = m.groups()
        if line.startswith("Termination reason:"):
            d["term"] = line.split(":", 1)[1].strip()
        if line.startswith("Final solution stats:"):
            for nxt in lines[i + 1:i + 4]:
                m = ROW.match(nxt)
                if m:
                    d["final"] = m.groups()
                    break
        if line.startswith("Time "):
            m = re.search(r"([\d.]+)\s*s", line)
            if m:
                d["wall"] = float(m.group(1))
    return d


def run1(cmd_base, th, env, timeout):
    res = subprocess.run(cmd_base + ["--params", params(1, th)], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return parse(res.stdout)["wall"], res.stdout


def summary(csv_path, out_path):
    rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["status"] == "OK"]
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], int(r["threads"])), []).append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("model,threads,n,iters,median_ms_per_iter,min_ms_per_iter,max_ms_per_iter,median_t_iter0_s,"
                "termination\n")
        for (m, th), v in sorted(cells.items()):
            ms = [float(r["ms_per_iter"]) for r in v]
            t0 = statistics.median(float(r["t_iter0_s"]) for r in v)
            mw = [float(r["ms_per_iter_wall"]) for r in v if r["ms_per_iter_wall"] != ""]
            mw = f"{statistics.median(mw):.4f}" if mw else ""
            f.write(f"{m},{th},{len(v)},{v[0]['iters']},{statistics.median(ms):.4f},{min(ms):.4f},"
                    f"{max(ms):.4f},{mw},{t0:.3f},{v[0]['termination']}\n")


def main():
    a = args_()
    if not os.path.exists(a.exe):
        sys.exit(f"missing {a.exe}")
    if a.wait:
        files = [f.strip() for f in a.wait.split(",") if f.strip()]
        print(f"[{datetime.now():%H:%M:%S}] waiting for ALL DONE in {files}", flush=True)
        wait_for(files)
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
    threads = [int(t) for t in a.threads.split(",")]
    tasks = [(r, m, t) for r in range(1, a.repeats + 1) for m in models for t in threads]
    numa = ["numactl", f"--cpunodebind={a.node}", f"--membind={a.node}"] if a.node >= 0 else []
    for i, (rep, m, th) in enumerate(tasks, 1):
        name = f"{m}_ort_t{th}_r{rep}"
        log = os.path.join(raw, name + ".log")
        if os.path.exists(log) and "Termination reason" in open(log, encoding="utf-8", errors="replace").read():
            print(f"[{i}/{len(tasks)}] skip {name}", flush=True)
            continue
        base = numa + [a.exe, "--input", paths[m], "--solver", "pdlp", "--num_threads", str(th)]
        cmd = base + ["--params", params(a.maxit, th)]
        env = dict(os.environ, OMP_NUM_THREADS=str(th))
        load = open("/proc/loadavg").read().split()[0]
        print(f"[{i}/{len(tasks)}] {datetime.now():%m-%d %H:%M:%S} {name} (load {load})", flush=True)
        t0 = time.time()
        row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "node": a.node, "model": m,
               "threads": th, "repeat": rep, "maxit": a.maxit, "loadavg_1m": load, "log_file": log}
        try:
            res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 timeout=a.timeout)
            w1, out1 = run1(base, th, env, a.timeout)
            open(log, "w", encoding="utf-8").write(res.stdout + "\n===== maxit=1 run =====\n" + out1)
            d = parse(res.stdout)
            row["wall_1it_s"] = "" if w1 is None else w1
            ok = res.returncode == 0 and d["final"] is not None and d["first"] is not None
            row["status"] = "OK" if ok else f"ERR_{res.returncode}"
            row["termination"] = d["term"]
            row["wall_s"] = "" if d["wall"] is None else d["wall"]
            if ok:
                it, tf, rp, rd, gap, po, do = d["final"]
                it, tf, ti0 = int(it), float(tf), float(d["first"][1])
                row.update({"iters": it, "t_iter0_s": ti0, "t_final_s": tf,
                            "ms_per_iter": round((tf - ti0) / it * 1000, 4) if it else "",
                            "ms_per_iter_wall": round((d["wall"] - w1) / (it - 1) * 1000, 4)
                            if (it > 1 and d["wall"] is not None and w1 is not None) else "",
                            "rel_p_res": rp, "rel_d_res": rd, "rel_gap": gap, "prim_obj": po, "dual_obj": do})
            print(f"    {row['status']} {d['term']} iters {row.get('iters')} ms/iter {row.get('ms_per_iter')} "
                  f"(wall-based {row.get('ms_per_iter_wall')}) wall {row['wall_s']}", flush=True)
        except subprocess.TimeoutExpired as e:
            out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
            open(log, "w", encoding="utf-8").write(f"TIMEOUT {a.timeout} s\n{out}")
            row.update({"status": "TIMEOUT", "wall_s": round(time.time() - t0, 1)})
            print(f"    TIMEOUT after {a.timeout} s", flush=True)
        w.writerow(row)
        fcsv.flush()
    fcsv.close()
    summary(csv_path, os.path.join(a.out_dir, "stats.csv"))
    print(f"[{datetime.now():%H:%M:%S}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
