# Cluster inventory — cpu, gpu, nolim, gnolim

Snapshot taken **2026-08-05 16:20:09 -04:00**. 96 nodes, 4,284 Slurm CPUs (hardware threads), 42.6 TiB RAM, 50 GPU nodes, 185 GPUs. No node belongs to more than one of the four partitions.

**Files written**

- `/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/_research/cluster_nodes.json` — every node's fields, plus per-partition limits, the reservation QOS, live reservations, and totals.
- `/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/_research/build_cluster_nodes_json.py` — the script that produced it. Re-run: `/p/rlprojects/RND/.venvs/exploration/bin/python /p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/_research/build_cluster_nodes_json.py`

Memory note: `RealMemory` is configured as (round GB) x 1000, so a 1 TB node reads 1024000 MB. The tables below give MB exactly and GiB = MB/1024. The JSON carries both (`real_memory_gib`, `real_memory_gb_admin_units` = MB/1000, which is what the server_introduction catalog prints).

CPU-count note: "CPUs" is the Slurm count of hardware threads. Every node reserves 1 core as specialized (2 on `cheetah01` and `slurm1`) plus 4000 MB, so the schedulable count is 2 lower per node — shown as "CPUs (eff.)".

---

## Table 1 — partition `cpu` (40 nodes, no GPUs)

| node | CPU model | sock x core x thr | CPUs (eff.) | memory MB / GiB | state | CPUs in use |
| --- | --- | --- | --- | --- | --- | --- |
| affogato01 | Intel Skylake | 1 x 16 x 2 | 32 (30) | 128,000 / 125.0 | ALLOCATED | 30/32 |
| affogato02 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 122,000 / 119.1 | MIXED | 32/64 |
| affogato03 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 96,000 / 93.8 | MIXED | 2/16 |
| affogato04 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | MIXED | 2/32 |
| affogato05 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | ALLOCATED | 30/32 |
| affogato06 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 128,000 / 125.0 | MIXED | 2/16 |
| affogato07 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 128,000 / 125.0 | IDLE | 0/16 |
| affogato08 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 128,000 / 125.0 | IDLE | 0/16 |
| affogato09 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 128,000 / 125.0 | IDLE | 0/16 |
| affogato10 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 128,000 / 125.0 | IDLE | 0/16 |
| bigcat01 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,500,000 / 1464.8 | IDLE | 0/64 |
| bigcat02 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,500,000 / 1464.8 | IDLE | 0/64 |
| bigcat03 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,500,000 / 1464.8 | IDLE | 0/64 |
| bigcat04 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,500,000 / 1464.8 | IDLE | 0/64 |
| bigcat05 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,500,000 / 1464.8 | IDLE | 0/64 |
| bigcat06 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,400,000 / 1367.2 | IDLE | 0/64 |
| cortado01 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| cortado02 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| cortado03 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | IDLE | 0/48 |
| cortado04 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | IDLE | 0/48 |
| cortado05 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | IDLE | 0/48 |
| cortado06 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | DOWN, NOT_RESPONDING<br>(reason "Not responding") | 0/48 |
| cortado07 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| cortado08 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| cortado09 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| cortado10 | Intel Skylake | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| hydro | Intel Skylake | 2 x 16 x 2 | 64 (62) | 256,000 / 250.0 | IDLE, RESERVED<br>(`as8hu_153`) | 0/64 |
| lynx08 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | IDLE | 0/32 |
| lynx09 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | IDLE | 0/32 |
| panther01 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 512,000 / 500.0 | ALLOCATED | 14/16 |
| puma01 | Intel Ice Lake | 2 x 40 x 2 | 160 (158) | 252,000 / 246.1 | MIXED | 120/160 |
| struct01 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct02 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct03 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct04 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct05 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct06 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct07 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct08 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |
| struct09 | Intel Broadwell | 1 x 14 x 2 | 28 (26) | 128,000 / 125.0 | IDLE | 0/28 |

Partition totals: 1,676 CPUs (1,596 schedulable), 17,562,000 MB (16.75 TiB), 412 CPUs in use, 843,776 MB in use, 25 of 40 nodes idle. Every node has `Gres=(null)` — the GPU columns are null in the JSON for all of them.

---

## Table 2 — partition `gpu` (42 nodes, 160 GPUs)

| node | CPU model | sock x core x thr | CPUs (eff.) | memory MB / GiB | GPUs | GRES type string | GPU model | GPU mem/card | architecture (year) | state | GPUs in use | CPUs in use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adriatic01 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| adriatic02 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| adriatic03 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| adriatic04 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| adriatic05 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| adriatic06 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| affogato11 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | MIXED | 2/4 | 24/32 |
| affogato13 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | MIXED | 0/4 | 26/32 |
| affogato14 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | MIXED | 0/4 | 26/32 |
| affogato15 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | MIXED | 0/4 | 26/32 |
| ai01 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | MIXED | 0/4 | 26/32 |
| ai02 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | MIXED | 0/4 | 26/32 |
| ai03 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | MIXED | 0/4 | 26/32 |
| ai04 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | MIXED | 0/4 | 14/32 |
| ai06 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 3 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | IDLE | 0/3 | 0/32 |
| cheetah01 | AMD EPYC 7252 | 2 x 8 x 2 | 32 (28) | 256,000 / 250.0 | 4 | `nvidia_a100-pcie-40gb` | A100 | 40.0 GiB | Ampere (2020) | MIXED | 3/4 | 18/32 |
| cheetah02 | Intel Skylake | 2 x 18 x 2 | 72 (70) | 1,024,000 / 1000.0 | 4 | `nvidia_rtx_4000_ada_generation` | RTX 4000 Ada | 20.0 GiB | Ada (2023) | IDLE | 0/4 | 0/72 |
| cheetah03 | Intel Skylake | 2 x 18 x 2 | 72 (70) | 1,024,000 / 1000.0 | 2 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | IDLE | 0/2 | 0/72 |
| cheetah04 | AMD EPYC 7742 | 2 x 64 x 2 | 256 (254) | 1,024,000 / 1000.0 | 4 | `a100` | A100 (NVLink) | 79.2 GiB* | Ampere (2020) | MIXED | 4/4 | 250/256 |
| cheetah08 | Intel Skylake | 2 x 10 x 2 | 40 (38) | 512,000 / 500.0 | 4 | `nvidia_rtx_a4000` | RTX A4000 | 16.0 GiB | Ampere (2021) | IDLE | 0/4 | 0/40 |
| cheetah09 | Intel Skylake | 2 x 10 x 2 | 40 (38) | 512,000 / 500.0 | 4 | `nvidia_rtx_a4000` | RTX A4000 | 16.0 GiB | Ampere (2021) | IDLE | 0/4 | 0/40 |
| jaguar01 | Intel Skylake | 2 x 16 x 2 | 64 (62) | 1,024,000 / 1000.0 | 4 | `nvidia_a40` | A40 (NVLink) | 45.0 GiB | Ampere (2020) | MIXED | 1/4 | 6/64 |
| jaguar02 | Intel Ice Lake | 2 x 8 x 2 | 32 (30) | 1,024,000 / 1000.0 | 8 | `nvidia_a16` | A16 | 15.0 GiB | Ampere (2021) | IDLE | 0/8 | 0/32 |
| jaguar03 | AMD EPYC 7663 | 2 x 56 x 2 | 224 (222) | 1,024,000 / 1000.0 | 8 | `nvidia_rtx_a4500` | RTX A4500 | 20.0 GiB | Ampere (2021) | IDLE | 0/8 | 0/224 |
| jaguar05 | Intel Skylake | 1 x 8 x 2 | 16 (14) | 256,000 / 250.0 | 4 | `quadro_rtx_4000` | Quadro RTX 4000 | 7.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/16 |
| jaguar06 | Intel Ice Lake | 2 x 12 x 2 | 48 (46) | 126,000 / 123.0 | 2 | `a40` | A40 | 45.0 GiB | Ampere (2020) | MIXED | 1/2 | 6/48 |
| lotus | Intel Skylake | 2 x 20 x 2 | 80 (78) | 256,000 / 250.0 | 8 | `quadro_rtx_6000` | Quadro RTX 6000 | 23.5 GiB* | Turing (2018) | IDLE | 0/8 | 0/80 |
| lynx01 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_titan_xp` | Titan Xp | 11.9 GiB* | Pascal (2017) | MIXED | 0/4 | 26/32 |
| lynx02 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | MIXED | 0/4 | 26/32 |
| lynx03 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | MIXED | 0/4 | 26/32 |
| lynx04 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | MIXED | 0/4 | 26/32 |
| lynx05 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `tesla_p100-pcie-12gb` | Tesla P100 | 11.9 GiB* | Pascal (2016) | MIXED | 0/4 | 26/32 |
| lynx06 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `tesla_p100-pcie-12gb` | Tesla P100 | 11.9 GiB* | Pascal (2016) | MIXED | 0/4 | 26/32 |
| lynx07 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `tesla_p100-pcie-12gb` | Tesla P100 | 11.9 GiB* | Pascal (2016) | MIXED | 0/4 | 26/32 |
| lynx10 | Intel Broadwell | 2 x 8 x 2 | 32 (30) | 64,000 / 62.5 | 4 | `nvidia_geforce_rtx_2080_ti` | RTX 2080 Ti | 10.6 GiB* | Turing (2018) | IDLE | 0/4 | 0/32 |
| nekomata01 | Intel Ice Lake | 1 x 12 x 2 | 24 (22) | 128,000 / 125.0 | 2 | `nvidia_geforce_rtx_5080` | RTX 5080 | 15.9 GiB* | Blackwell (2025) | MIXED | 1/2 | 8/24 |
| nekomata02 | Intel Ice Lake | 1 x 12 x 2 | 24 (22) | 128,000 / 125.0 | 2 | `nvidia_geforce_rtx_5080` | RTX 5080 | 15.9 GiB* | Blackwell (2025) | MIXED | 1/2 | 8/24 |
| serval03 | AMD EPYC 9534 | 1 x 64 x 2 | 128 (126) | 512,000 / 500.0 | 1 | `nvidia_h100_nvl` | H100 NVL | 93.6 GiB | Hopper (2023) | IDLE, RESERVED<br>(`nkp2mr_155`) | 0/1 | 0/128 |
| serval06 | AMD EPYC 9354 | 1 x 32 x 2 | 64 (62) | 1,500,000 / 1464.8 | 2 | `nvidia_h100_nvl` | H100 NVL | 93.6 GiB | Hopper (2023) | MIXED | 2/2 | 16/64 |
| serval07 | AMD EPYC 9354 | 1 x 32 x 2 | 64 (62) | 1,500,000 / 1464.8 | 2 | `nvidia_h100_nvl` | H100 NVL | 93.6 GiB | Hopper (2023) | MIXED | 2/2 | 24/64 |
| serval08 | AMD EPYC 9354 | 1 x 32 x 2 | 64 (62) | 1,500,000 / 1464.8 | 2 | `nvidia_h100_nvl` | H100 NVL | 93.6 GiB | Hopper (2023) | MIXED | 2/2 | 16/64 |
| serval09 | AMD EPYC 9354 | 1 x 32 x 2 | 64 (62) | 1,500,000 / 1464.8 | 2 | `nvidia_h100_nvl` | H100 NVL | 93.6 GiB | Hopper (2023) | MIXED | 2/2 | 16/64 |

`*` = the catalog marks this GPU memory value approximate; confirm with `nvidia-smi --query-gpu=memory.total` on the node before relying on it.

Partition totals: 2,144 CPUs (2,058 schedulable), 22,318,000 MB (21.28 TiB), 160 GPUs. In use right now: 21 GPUs, 744 CPUs, 2,244,608 MB. Free: **139 of 160 GPUs**, 17 of 42 nodes fully idle.

Free GPUs by model (gpu partition):

| GPU model | architecture (year) | mem/card | total | in use | free |
| --- | --- | --- | --- | --- | --- |
| RTX 5080 | Blackwell (2025) | 15.9 GiB* | 4 | 2 | 2 |
| H100 NVL | Hopper (2023) | 93.6 GiB | 9 | 8 | 1 |
| RTX 4000 Ada | Ada (2023) | 20.0 GiB | 4 | 0 | 4 |
| RTX A4500 | Ampere (2021) | 20.0 GiB | 8 | 0 | 8 |
| RTX A4000 | Ampere (2021) | 16.0 GiB | 8 | 0 | 8 |
| A16 | Ampere (2021) | 15.0 GiB | 8 | 0 | 8 |
| A100 (NVLink, cheetah04) | Ampere (2020) | 79.2 GiB* | 4 | 4 | 0 |
| A40 | Ampere (2020) | 45.0 GiB | 6 | 2 | 4 |
| A100 (PCIe, cheetah01) | Ampere (2020) | 40.0 GiB | 4 | 3 | 1 |
| Quadro RTX 6000 | Turing (2018) | 23.5 GiB* | 8 | 0 | 8 |
| RTX 2080 Ti | Turing (2018) | 10.6 GiB* | 29 | 2 | 27 |
| Quadro RTX 4000 | Turing (2018) | 7.6 GiB* | 28 | 0 | 28 |
| Titan Xp | Pascal (2017) | 11.9 GiB* | 4 | 0 | 4 |
| GTX 1080 Ti | Pascal (2017) | 11.0 GiB | 24 | 0 | 24 |
| Tesla P100 | Pascal (2016) | 11.9 GiB* | 12 | 0 | 12 |

---

## Table 3 — partition `nolim` (6 nodes, no GPUs)

| node | CPU model | sock x core x thr | CPUs (eff.) | memory MB / GiB | state | CPUs in use |
| --- | --- | --- | --- | --- | --- | --- |
| heartpiece | Intel Skylake | 2 x 10 x 2 | 40 (38) | 160,000 / 156.2 | MIXED | 32/40 |
| slurm1 | Intel Haswell | 2 x 12 x 1 | 24 (22) | 512,000 / 500.0 | IDLE | 0/24 |
| slurm2 | Intel Haswell | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| slurm3 | Intel Haswell | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | MIXED | 30/48 |
| slurm4 | Intel Haswell | 2 x 12 x 2 | 48 (46) | 512,000 / 500.0 | IDLE | 0/48 |
| slurm5 | Intel Haswell | 1 x 12 x 2 | 24 (22) | 256,000 / 250.0 | IDLE | 0/24 |

Partition totals: 232 CPUs (220 schedulable), 2,464,000 MB (2.35 TiB), 92 CPUs in use, 188,416 MB in use, 3 of 6 nodes idle. `slurm1` is the only node in the cluster with `ThreadsPerCore=1`, and (with `cheetah01`) one of two nodes reserving 2 specialized cores instead of 1. All six nodes have `Gres=(null)`.

---

## Table 4 — partition `gnolim` (8 nodes, 25 GPUs)

| node | CPU model | sock x core x thr | CPUs (eff.) | memory MB / GiB | GPUs | GRES type string | GPU model | GPU mem/card | architecture (year) | state | GPUs in use | CPUs in use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ai05 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080` | GTX 1080 | 8.0 GiB* | Pascal (2016) | MIXED | 0/4 | 26/32 |
| ai07 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | IDLE | 0/4 | 0/32 |
| ai08 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | IDLE | 0/4 | 0/32 |
| ai09 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 112,000 / 109.4 | 4 | `nvidia_geforce_gtx_1080_ti` | GTX 1080 Ti | 11.0 GiB | Pascal (2017) | IDLE | 0/4 | 0/32 |
| ai10 | Intel Skylake | 2 x 8 x 2 | 32 (30) | 128,000 / 125.0 | 4 | `nvidia_geforce_gtx_1080` | GTX 1080 | 8.0 GiB* | Pascal (2016) | IDLE | 0/4 | 0/32 |
| jinx01 | Intel Haswell | 1 x 12 x 2 | 24 (22) | 220,000 / 214.8 | 2 | `nvidia_geforce_gtx_1080` | GTX 1080 | 8.0 GiB* | Pascal (2016) | MIXED | 0/2 | 12/24 |
| jinx02 | Intel Haswell | 1 x 12 x 2 | 24 (22) | 220,000 / 214.8 | 2 | `nvidia_geforce_gtx_1080` | GTX 1080 | 8.0 GiB* | Pascal (2016) | MIXED | 0/2 | 12/24 |
| titanx03 | Intel Haswell | 1 x 12 x 2 | 24 (22) | 250,000 / 244.1 | 1 | `nvidia_titan_x` | Titan X | 12.0 GiB* | Maxwell (2015) | MIXED | 0/1 | 12/24 |

Partition totals: 232 CPUs (216 schedulable), 1,314,000 MB (1.25 TiB), 25 GPUs. In use: 0 GPUs, 62 CPUs, 126,976 MB. **All 25 GPUs are free** — the CPUs in use here are CPU-only jobs. Free by model: GTX 1080 Ti 12, GTX 1080 12, Titan X 1.

---

## Per-partition per-user limits (all live)

| partition | QOS | time limit (default) | per-user CPU cap | per-user GPU cap | per-user memory cap |
| --- | --- | --- | --- | --- | --- |
| cpu | `cspartcpu` | 4 days, `4-00:00:00` (default `02:00:00`) | 400 | none set | 4,194,304 MB (4 TiB) |
| gpu | `cspartgpu` | 4 days, `4-00:00:00` (default `02:00:00`) | 400 | 40 | 4,194,304 MB (4 TiB) |
| nolim | `cspartnolim` | 20 days, `20-00:00:00` (default `08:00:00`) | 80 | none set | 1,048,576 MB (1 TiB) |
| gnolim | `cspartgnolim` | 20 days, `20-00:00:00` (default `08:00:00`) | 80 | 20 | 1,048,576 MB (1 TiB) |
| reservation nodes<br>(needs `--reservation=<name>`) | `csresnolim` | min(partition limit, reservation end) | 16,384 | 1,024 | 1,073,741,824 MB (1 PiB) |

The pools are separate per partition, so a user can hold 400 + 400 + 80 + 80 CPUs at once, plus reserved nodes on top. Reservation jobs are admitted above the gpu-partition caps but their usage still counts into those counters, so submit open-partition jobs first and reservation jobs last.

Current usage for `sl5nw` at snapshot time: cpu partition 14 CPUs / 28,672 MB; gpu partition 0; nolim 32 CPUs / 65,536 MB; gnolim has no association row for this user (cap still applies).

Reservations active right now (neither belongs to `sl5nw`): `as8hu_153` on `hydro` until 2026-08-28, and `nkp2mr_155` on `serval03` until 2026-08-07. Both carry `IGNORE_JOBS,SPEC_NODES`, so a job without a matching `--reservation=` cannot run on those two nodes.

---

## Where each fact came from

**Live Slurm** (read at 2026-08-05 16:20 by `build_cluster_nodes_json.py`):

1. Node name, partition membership, state, reservation, reason — `scontrol show node --json`.
2. Sockets, cores per socket, threads per core, total CPUs, effective CPUs, specialized cores/memory — same source.
3. `RealMemory` (MB), allocated memory, free memory, allocated CPUs, CPU load — same source.
4. GPU count and the GPU type string — the node's `Gres` field; GPUs currently allocated — the node's `GresUsed` field.
5. CPU type — the node `Features` list (`skylake`, `broadwell`, `icelake`, `haswell`, `amd_epyc_7663`, ...), cross-read with `sinfo -o "%n %f"`. The `nvlink` tag on `cheetah04` and `jaguar01` is also a live feature.
6. Partition time limits, default times, QOS names, node lists, partition TRES — `scontrol show partition <p>`.
7. Per-user caps — `scontrol show assoc_mgr qos=<q> flags=qos` (`MaxTRESPU` lines) and `sacctmgr show qos format=name,maxtresperuser%60,maxwall`. The two agree everywhere.
8. Reservations — `scontrol show reservation -o`.

**server_introduction catalog** (`/p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/code/gpu_catalog.json`, joined on the exact Slurm GPU type string, never a substring):

1. GPU marketing name (`nvidia_rtx_a4500` → RTX A4500).
2. GPU memory per card, with the exact/approximate label.
3. Architecture family, release year, compute capability.

Slurm itself publishes none of these three groups — no node reports `gres/gpumem` in its `CfgTRES`, so GPU memory cannot be read live from Slurm on this cluster.

**Generator**: `/p/rlprojects/.claude/skills/submit-gpu-sweep/server_introduction/code/generate_server_introduction.py`. Run it with no arguments to rewrite `server_introduction.json` and `server_introduction.md` from live Slurm; `--check` re-derives the totals two ways and writes nothing; `--node-json <file>` reads a saved `scontrol show node --json` fixture instead of live Slurm. Use the project env: `/p/rlprojects/RND/.venvs/exploration/bin/python generate_server_introduction.py --check`. It covers only GPU nodes (gpu + gnolim) and groups them into node classes — it does not cover the cpu and nolim partitions, which is why the four-partition inventory needed its own script.

## Where the catalog and live Slurm disagree

1. **`nekomata02` is missing from the catalog.** Live Slurm has it in the gpu partition (2 x RTX 5080, Intel Ice Lake, 24 threads, 128,000 MB), matching `nekomata01` exactly. The catalog, generated 2026-07-11, lists only `nekomata01`. So the catalog's header line "41 nodes / 158 GPUs" for the gpu partition is now **42 nodes / 160 GPUs**, and the cluster-wide GPU line "49 GPU nodes, 183 GPUs" is now **50 GPU nodes, 185 GPUs**. The gnolim numbers (8 nodes / 25 GPUs) still match. Re-running the generator would fix this.
2. **The generator's `--check` currently fails** with `ServerIntroductionError: qos csresnolim has no MaxTRESPU line`. That is not a stale-catalog problem: `scontrol show assoc_mgr qos=csresnolim flags=qos` right now prints "No Accounts / No Users", so there is no live `MaxTRESPU` line to parse. The reservation QOS caps in the catalog's Table 2 (1,024 GPUs / 16,384 CPUs / 1 PB) are still correct — `sacctmgr show qos` reports `cpu=16384,gres/gpu=1024,mem=1P`. Until an association exists for that QOS again, the generator cannot rewrite its files, so the catalog cannot pick up `nekomata02` without a code change or a live association.
3. **Memory unit convention differs, the underlying number does not.** The catalog's "sys mem/node (GB)" column divides `RealMemory` by 1000 (jaguar06 → 126, adriatic01 → 1024, serval06 → 1500), because the admins configure `RealMemory` as round GB times 1000. My tables and the JSON's `real_memory_gib` divide by 1024 (jaguar06 → 123.0, adriatic01 → 1000.0, serval06 → 1464.8). Both come from the same live MB value; the JSON carries both fields.
4. **`lynx01` has a misleading feature tag.** Its `Features` say `titan_x`, but its `Gres` says `gpu:nvidia_titan_xp:4`. The catalog resolves it by the GRES string to Titan Xp (Pascal, 2017), and so does this inventory. Do not read the `titan_x` feature as the actual card. The genuine Titan X (Maxwell, 2015) is `titanx03` in gnolim.
5. **Everything else matches.** CPU types, thread counts, GPU counts, GPU type strings, per-node memory, and the gpu/gnolim per-user caps all agree between the catalog and live Slurm for the 49 nodes the catalog covers.

## Other live facts worth carrying into the document

1. `cortado06` (cpu partition) is DOWN and not responding, reason "Not responding"; it is the only node with no reported slurmd version. Its 48 CPUs and 512,000 MB are in the partition totals but are not usable.
2. Every node reserves 1 core and 4,000 MB as specialized (`cheetah01` and `slurm1` reserve 2 cores), which is why partition `TotalCPUs` (1,676 / 2,144 / 232 / 232) exceeds the schedulable `TRES cpu=` (1,596 / 2,058 / 220 / 216).
3. All nodes run slurmd 23.11.6. Every node has `ThreadsPerCore=2` except `slurm1` — the hyperthreading charging rule (pass `--ntasks-per-core=2` for many-single-CPU-task jobs) applies cluster-wide.
4. The largest single machines: `cheetah04` (256 threads, AMD EPYC 7742, 4 x A100 80 GB, currently 250/256 CPUs and 4/4 GPUs in use), `jaguar03` (224 threads, AMD EPYC 7663, 8 x RTX A4500, fully idle), `puma01` (160 threads, Intel Ice Lake, cpu partition, 120/160 in use), `serval03` (128 threads, reserved for another user until 2026-08-07).