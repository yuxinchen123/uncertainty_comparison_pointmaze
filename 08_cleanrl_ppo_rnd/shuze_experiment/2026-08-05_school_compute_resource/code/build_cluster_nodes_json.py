#!/usr/bin/env python3
"""Build a per-node inventory of the cpu, gpu, nolim, gnolim partitions from live Slurm.

Live sources: `scontrol show node --json`, `scontrol show partition <p>`,
`scontrol show assoc_mgr qos=<q> flags=qos`, `sacctmgr show qos`.
Static join: the submit-gpu-sweep skill's gpu_catalog.json (GPU marketing name, memory,
architecture family, release year).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PARTITIONS = ["cpu", "gpu", "nolim", "gnolim"]
CATALOG = Path(
    "/p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/code/gpu_catalog.json"
)
OUT = Path("/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/_research/cluster_nodes.json")

GPU_GRES = re.compile(r"(?:^|,)gpu:([^:,()]+):([0-9]+)(?:\([^)]*\))?(?=,|$)")
INTEL_MICROARCH = {"skylake", "broadwell", "icelake", "haswell"}
INTEL_DISPLAY = {
    "skylake": "Intel Skylake",
    "broadwell": "Intel Broadwell",
    "icelake": "Intel Ice Lake",
    "haswell": "Intel Haswell",
}
PARTITION_QOS = {
    "cpu": "cspartcpu",
    "gpu": "cspartgpu",
    "nolim": "cspartnolim",
    "gnolim": "cspartgnolim",
}


def run(cmd: list[str]) -> str:
    """Run a command and return its stdout, failing loudly on a non-zero exit."""

    # Every fact in this file must come from a command that actually succeeded.
    done = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return done.stdout


def cpu_display(features: list[str], node: str) -> str:
    """Turn a node's Slurm feature list into a human CPU-type string."""

    # An amd_epyc_* feature names the exact model; the Intel nodes only tag a microarchitecture.
    known = [f for f in features if f.startswith("amd_epyc_") or f in INTEL_MICROARCH]
    if len(known) != 1:
        raise SystemExit(f"{node}: expected exactly one cpu feature, got {known} from {features}")
    tag = known[0]
    if tag.startswith("amd_epyc_"):
        return "AMD EPYC " + tag[len("amd_epyc_") :]
    return INTEL_DISPLAY[tag]


def parse_gpu(gres: str, node: str) -> tuple[str, int] | None:
    """Parse the typed GPU record out of a Slurm gres string."""

    # A node with no typed gpu record is a CPU-only node; anything else must parse exactly once.
    if not gres:
        return None
    found = GPU_GRES.findall(gres)
    if not found:
        return None
    if len(found) != 1:
        raise SystemExit(f"{node}: expected one gpu gres record, got {found}")
    return found[0][0], int(found[0][1])


def parse_max_tres_pu(body: str) -> dict[str, dict[str, int | None]]:
    """Split one MaxTRESPU value list into per-resource cap and current-usage numbers."""

    # A MaxTRESPU line reads like "cpu=400(12),gres/gpu=40(0),mem=4194304(24576),node=N(1)";
    # a cap of "N" means no cap is set for that resource.
    out: dict[str, dict[str, int | None]] = {}
    for item in body.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        cap_text, _, used_text = value.partition("(")
        used_text = used_text.rstrip(")")
        cap = int(cap_text) if cap_text.strip().isdigit() else None
        used = int(used_text) if used_text.strip().isdigit() else None
        out[key.strip()] = {"cap": cap, "current_usage": used}
    return out


def qos_caps(qos: str, user: str) -> dict[str, object]:
    """Read one QOS's per-user TRES caps, plus this user's current usage, from assoc_mgr.

    The caps are identical on every user's MaxTRESPU line; only the parenthesized usage differs,
    so the caps come from the first line seen and the usage from this user's own block.
    """

    text = run(["scontrol", "show", "assoc_mgr", f"qos={qos}", "flags=qos"])
    caps: dict[str, object] = {
        "qos": qos,
        "cpu": None,
        "gpu": None,
        "mem_mb": None,
        "user": user,
        "user_current_cpu": None,
        "user_current_gpu": None,
        "user_current_mem_mb": None,
        "max_tres_pu_line_found": False,
    }
    current_user = None
    for raw in text.splitlines():
        line = raw.strip()
        named = re.match(r"^([A-Za-z0-9_.-]+)\((\d+)\)$", line)
        if named:
            current_user = named.group(1)
            continue
        if not line.startswith("MaxTRESPU="):
            continue
        parsed = parse_max_tres_pu(line[len("MaxTRESPU=") :])
        if not caps["max_tres_pu_line_found"]:
            caps["max_tres_pu_line_found"] = True
            caps["cpu"] = parsed.get("cpu", {}).get("cap")
            caps["gpu"] = parsed.get("gres/gpu", {}).get("cap")
            caps["mem_mb"] = parsed.get("mem", {}).get("cap")
        if current_user == user:
            caps["user_current_cpu"] = parsed.get("cpu", {}).get("current_usage")
            caps["user_current_gpu"] = parsed.get("gres/gpu", {}).get("current_usage")
            caps["user_current_mem_mb"] = parsed.get("mem", {}).get("current_usage")
    return caps


def sacctmgr_caps() -> dict[str, str]:
    """Read the configured per-user TRES caps straight from the accounting database."""

    # sacctmgr prints the configured limit text (e.g. "cpu=400,gres/gpu=40,mem=4T") per QOS.
    text = run(["sacctmgr", "-nP", "show", "qos", "format=name,maxtresperuser%60,maxwall"])
    out = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def partition_facts(name: str) -> dict[str, object]:
    """Read one partition's configured limits and totals."""

    # The scontrol text is parsed key by key so a missing field fails loudly instead of silently.
    text = run(["scontrol", "show", "partition", name])
    flat = " ".join(text.split())
    def grab(key: str) -> str:
        found = re.search(rf"\b{key}=(\S+)", flat)
        if not found:
            raise SystemExit(f"partition {name}: no {key}")
        return found.group(1)
    return {
        "partition": name,
        "state": grab("State"),
        "max_time": grab("MaxTime"),
        "default_time": grab("DefaultTime"),
        "qos": grab("QoS"),
        "total_nodes": int(grab("TotalNodes")),
        "total_cpus_slurm": int(grab("TotalCPUs")),
        "tres": grab("TRES"),
    }


def main() -> int:
    """Assemble the node inventory plus the per-partition limits and write the JSON."""

    # Static GPU facts keyed by the exact Slurm type string - no substring matching.
    catalog = json.loads(CATALOG.read_text())
    families = {row["family_id"]: row for row in catalog["families"]}
    gpu_types = {row["slurm_gpu_type"]: row for row in catalog["gpu_types"]}

    node_json = json.loads(run(["scontrol", "show", "node", "--json"]))
    nodes_out = []
    for node in node_json["nodes"]:
        parts = [p for p in node["partitions"] if p in PARTITIONS]
        if not parts:
            continue
        name = node["name"]
        gpu = parse_gpu(node.get("gres", ""), name)
        used = parse_gpu(node.get("gres_used", ""), name)
        record: dict[str, object] = {
            "node": name,
            "partitions": sorted(node["partitions"]),
            "state": "+".join(node["state"]),
            "cpu_model": cpu_display(node.get("features", []), name),
            "features": node.get("features", []),
            "sockets": node["sockets"],
            "cores_per_socket": node["cores"],
            "threads_per_core": node["threads"],
            "cpus_total_slurm": node["cpus"],
            "cpus_effective": node.get("effective_cpus"),
            "specialized_cores": node.get("specialized_cores", 0),
            "real_memory_mb": node["real_memory"],
            "real_memory_gib": round(node["real_memory"] / 1024, 1),
            "real_memory_gb_admin_units": round(node["real_memory"] / 1000),
            "specialized_memory_mb": node.get("specialized_memory", 0),
            "gres": node.get("gres", ""),
            "gres_used": node.get("gres_used", ""),
            "gpu_count": 0,
            "gpu_type_slurm": None,
            "gpu_model": None,
            "gpu_memory_mb_per_card": None,
            "gpu_memory_gib_per_card": None,
            "gpu_memory_gib_per_card_rounded": None,
            "gpu_memory_label": None,
            "gpu_architecture": None,
            "gpu_architecture_year": None,
            "gpu_compute_capability": None,
            "gpus_allocated": 0,
            "gpus_free": 0,
            "cpus_allocated": node.get("alloc_cpus", 0),
            "cpus_idle": node.get("alloc_idle_cpus"),
            "memory_allocated_mb": node.get("alloc_memory", 0),
            "cpu_load": node.get("cpu_load"),
            "free_memory_mb": (node.get("free_mem") or {}).get("number"),
            "reservation": node.get("reservation", ""),
            "reason": node.get("reason", ""),
            "tres_configured": node.get("tres", ""),
            "tres_used": node.get("tres_used", ""),
            "slurm_version": node.get("version", ""),
        }
        if gpu:
            slurm_type, count = gpu
            if slurm_type not in gpu_types:
                raise SystemExit(f"{name}: gpu type {slurm_type} missing from the catalog")
            type_row = gpu_types[slurm_type]
            family = families[type_row["family_id"]]
            allocated = used[1] if used else 0
            record.update(
                gpu_count=count,
                gpu_type_slurm=slurm_type,
                gpu_model=family["display_name"],
                gpu_memory_mb_per_card=type_row["gpu_memory_mb"],
                gpu_memory_gib_per_card=round(type_row["gpu_memory_mb"] / 1024, 1),
                gpu_memory_gib_per_card_rounded=round(type_row["gpu_memory_mb"] / 1024),
                gpu_memory_label=type_row["memory_label"],
                gpu_architecture=family["architecture"],
                gpu_architecture_year=family["year"],
                gpu_compute_capability=family["compute_capability"],
                gpus_allocated=allocated,
                gpus_free=count - allocated,
            )
        nodes_out.append(record)

    nodes_out.sort(key=lambda r: r["node"])

    partitions_out = []
    configured = sacctmgr_caps()
    user = os.environ.get("USER", "")
    for name in PARTITIONS:
        facts = partition_facts(name)
        qos = PARTITION_QOS[name]
        facts["per_user_limits_live"] = qos_caps(qos, user)
        facts["per_user_limits_configured_text"] = configured.get(qos, "")
        members = [r for r in nodes_out if name in r["partitions"]]
        facts["node_count_live"] = len(members)
        facts["gpu_count_live"] = sum(r["gpu_count"] for r in members)
        facts["gpus_allocated_live"] = sum(r["gpus_allocated"] for r in members)
        facts["gpus_free_live"] = sum(r["gpus_free"] for r in members)
        facts["cpus_total_live"] = sum(r["cpus_total_slurm"] for r in members)
        facts["cpus_allocated_live"] = sum(r["cpus_allocated"] for r in members)
        facts["real_memory_mb_live"] = sum(r["real_memory_mb"] for r in members)
        partitions_out.append(facts)

    reservations = run(["scontrol", "show", "reservation", "-o"]).strip()
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": {
            "live_slurm": [
                "scontrol show node --json",
                "scontrol show partition <p>",
                "scontrol show assoc_mgr qos=<q> flags=qos",
                "sacctmgr show qos format=name,maxtresperuser,maxwall",
                "scontrol show reservation -o",
            ],
            "static_catalog": str(CATALOG),
            "catalog_fields": [
                "gpu_model", "gpu_memory_mb_per_card", "gpu_memory_label",
                "gpu_architecture", "gpu_architecture_year", "gpu_compute_capability",
            ],
        },
        "reservation_qos": {
            "qos": "csresnolim",
            "configured_max_tres_per_user": configured.get("csresnolim", ""),
            "assoc_mgr_live": qos_caps("csresnolim", user),
            "note": (
                "Reached only with --reservation=<name>. Its assoc_mgr record currently lists no "
                "accounts and no users, so no live MaxTRESPU line exists; the caps below come "
                "from the accounting database (sacctmgr)."
            ),
        },
        "partitions": partitions_out,
        "nodes": nodes_out,
        "reservations_raw": reservations.splitlines(),
        "totals": {
            "node_count": len(nodes_out),
            "gpu_node_count": sum(1 for r in nodes_out if r["gpu_count"]),
            "gpu_count": sum(r["gpu_count"] for r in nodes_out),
            "gpus_allocated": sum(r["gpus_allocated"] for r in nodes_out),
            "cpus_total_slurm": sum(r["cpus_total_slurm"] for r in nodes_out),
            "real_memory_mb": sum(r["real_memory_mb"] for r in nodes_out),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} nodes={len(nodes_out)} gpus={payload['totals']['gpu_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
