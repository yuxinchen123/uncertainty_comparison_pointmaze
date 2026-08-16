"""Tests for the hardware-log parsers, on fragments copied out of the real logs."""
import pytest

from collect_hardware import (cache_instances, cache_row, first_json_object, loaded_clock_ghz,
                              lscpu_value, memory_channels)

LSCPU = """Architecture:                            x86_64
CPU(s):                                  32
Model name:                              Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz
CPU family:                              6
Model:                                   106
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               2
Stepping:                                6
L1d cache:                               768 KiB (16 instances)
L2 cache:                                20 MiB (16 instances)
L3 cache:                                36 MiB (2 instances)
NUMA node(s):                            2
===== lscpu -C (cache) =====
NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL  SETS PHY-LINE COHERENCY-SIZE
L1d       48K     768K   12 Data            1    64        1             64
L2       1.3M      20M   20 Unified         2  1024        1             64
L3        18M      36M   12 Unified         3 24576        1             64
"""

PROBE = """===== probe with the node otherwise idle =====
{
 "clock_ghz_dependent_add_chain": 3.590,
 "latency_ns_per_dependent_access": {
  "24KiB_level1": 1.39,
  "256MiB_main_memory": 92.23
 },
 "read_bandwidth_gb_per_s_one_core": {
  "24KiB_level1": 58.97
 }
}
===== clock while 15 training workers run on this node =====
sample 1 at 2026-08-15T19:56:15-04:00: {"clock_ghz_dependent_add_chain": 3.563}
sample 2 at 2026-08-15T19:56:20-04:00: {"clock_ghz_dependent_add_chain": 3.561}
"""

MEM = """   dimm0 label=CPU_SrcID#0_MC#0_Chan#0_DIMM#0 size=65536 type=Unbuffered-DDR4
   dimm2 label=CPU_SrcID#0_MC#0_Chan#1_DIMM#0 size=65536 type=Unbuffered-DDR4
   dimm0 label=CPU_SrcID#0_MC#1_Chan#0_DIMM#0 size=65536 type=Unbuffered-DDR4
   dimm2 label=CPU_SrcID#1_MC#0_Chan#0_DIMM#0 size=65536 type=Unbuffered-DDR4
"""


def test_lscpu_value_reads_a_field():
    assert lscpu_value(LSCPU, "Model name") == "Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz"
    assert lscpu_value(LSCPU, "Core(s) per socket") == "8"


def test_lscpu_value_refuses_a_missing_field():
    with pytest.raises(ValueError):
        lscpu_value(LSCPU, "Nonexistent field")


def test_cache_row_gives_per_instance_and_total():
    assert cache_row(LSCPU, "L3") == ("18M", "36M")
    assert cache_row(LSCPU, "L2") == ("1.3M", "20M")


def test_cache_instances_counts_separate_caches():
    assert cache_instances(LSCPU, "L3") == 2
    assert cache_instances(LSCPU, "L2") == 16


def test_first_json_object_picks_the_full_probe_output():
    # the short --clock-only objects appear later in the file and must not be chosen
    blob = first_json_object(PROBE, "latency_ns_per_dependent_access")
    assert blob["clock_ghz_dependent_add_chain"] == 3.590
    assert blob["latency_ns_per_dependent_access"]["256MiB_main_memory"] == 92.23


def test_first_json_object_refuses_when_the_key_is_absent():
    with pytest.raises(ValueError):
        first_json_object(PROBE, "not_a_key_in_any_object")


def test_loaded_clock_is_the_mean_of_the_samples():
    assert loaded_clock_ghz(PROBE) == pytest.approx((3.563 + 3.561) / 2)


def test_memory_channels_counts_channels_per_socket():
    mem = memory_channels(MEM)
    # socket 0 has controller 0 channels 0 and 1 plus controller 1 channel 0, so three channels
    assert mem["channels_per_socket"] == 3
    assert mem["sockets_populated"] == 2
    assert mem["memory_type"] == "Unbuffered-DDR4"
    assert mem["dimms_total"] == 4
    assert mem["dimm_size_mib"] == 65536


def test_memory_channels_returns_nothing_without_labels():
    assert memory_channels("dmidecode refused") is None
