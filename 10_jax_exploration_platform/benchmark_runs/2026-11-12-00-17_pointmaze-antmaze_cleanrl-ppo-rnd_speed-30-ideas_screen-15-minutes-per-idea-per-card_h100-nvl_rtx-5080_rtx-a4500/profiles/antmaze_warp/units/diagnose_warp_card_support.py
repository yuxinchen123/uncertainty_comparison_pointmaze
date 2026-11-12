"""What the installed Warp build says about cards, read on a node without a card.

Run on a compute node:  <platform_jax_warp python> diagnose_warp_card_support.py
Prints the CUDA toolkit Warp was built with, the lowest driver it accepts, and the card architectures its
built-in kernel compiler (NVRTC) can target. RTX A4500 is architecture 86, RTX 5080 is 120, H100 is 90.
"""
import warp as wp

wp.init()
runtime = wp._src.context.runtime
print("warp", wp.config.version, "CUDA toolkit", runtime.toolkit_version, "lowest accepted driver (CUDA version)",
      runtime.min_driver_version, "minor-version compatibility", runtime.is_cuda_compatibility_enabled)
print("architectures the kernel compiler can target:", sorted(runtime.nvrtc_supported_archs))
print("architecture 86 (RTX A4500):", 86 in runtime.nvrtc_supported_archs,
      "| architecture 120 (RTX 5080):", 120 in runtime.nvrtc_supported_archs)
