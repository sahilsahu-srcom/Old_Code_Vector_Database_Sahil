"""
File Summary:
Hardware auto-detector for HAUP v3.0 Reverse Pipeline. Auto-configures workers, chunk size, and device based on system resources.

====================================================================
Startup
====================================================================

detect()
||
├── CPU Detection ------------------------------------> Physical core count
│       │
│       ├── psutil.cpu_count(logical=False) ----------> Physical cores
│       │
│       └── [Exception Block] -----------------------> Fallback to 4 cores, 8 GB RAM
│
├── Memory Detection ---------------------------------> Total system RAM
│       │
│       └── psutil.virtual_memory().total ------------> RAM in GB
│
├── GPU Detection ------------------------------------> CUDA availability and VRAM
│       │
│       ├── torch.cuda.is_available() ----------------> GPU presence check
│       │
│       ├── torch.cuda.get_device_properties() -------> VRAM size detection
│       │
│       ├── [Conditional Branch] gpu + not Windows ---> Set device="cuda", tune initial_batch
│       │
│       └── [Exception Block] -----------------------> Fallback to device="cpu"
│
├── Worker Calculation -------------------------------> Safe process count
│       │
│       ├── min(cpu_cores - 1, 8) --------------------> Base worker count
│       │
│       └── [Conditional Branch] Windows (os.name=='nt') -> Cap at 3 workers
│
├── Chunk Sizing -------------------------------------> Memory-based batch sizing
│       │
│       ├── (RAM * 0.08 * 1e9) / bytes_per_entry -----> Chunk size formula
│       │
│       ├── max(500, min(chunk_size, 10_000)) ---------> Clamp to safe range
│       │
│       └── [Conditional Branch] Windows -------------> Further cap at 2_000
│
├── HardwareConfig()  [Class → Object] --------------> Pack all settings into dataclass
│
└── _print_banner()  [Function] ---------------------> Print detection summary to console

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


"""================= Startup class HardwareConfig ================="""
@dataclass
class HardwareConfig:
    num_workers:   int
    chunk_size:    int
    device:        str
    initial_batch: int
    model_name:    str
    gpu_vram_gb:   float
    total_ram_gb:  float
    cpu_cores:     int
    ollama_host:   str = "http://localhost:11434"
    ollama_model:  str = "mistral:7b"
"""================= End class HardwareConfig ================="""


"""================= Startup function detect ================="""
def detect(model_name: str = "all-MiniLM-L6-v2") -> HardwareConfig:
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / 1e9
        cpu_cores    = psutil.cpu_count(logical=False) or os.cpu_count() or 2
    except (ImportError, Exception):
        total_ram_gb = 8.0
        cpu_cores    = 4

    gpu_available = False
    gpu_vram_gb   = 0.0
    device        = "cpu"
    initial_batch = 32

    try:
        import torch
        if torch.cuda.is_available() and os.name != 'nt':
            gpu_available = True
            props         = torch.cuda.get_device_properties(0)
            gpu_vram_gb   = props.total_memory / 1e9
            device        = "cuda"
            initial_batch = 64 if gpu_vram_gb > 8 else 32
        else:
            device        = "cpu"
            initial_batch = 32
    except (ImportError, RuntimeError, Exception):
        device        = "cpu"
        initial_batch = 32

    base_workers = min(max(cpu_cores - 1, 1), 8)

    if os.name == 'nt':
        num_workers = min(base_workers, 3)
    else:
        num_workers = base_workers

    bytes_per_entry = 6_144
    chunk_size = int((total_ram_gb * 0.08 * 1e9) / bytes_per_entry)
    chunk_size = max(500, min(chunk_size, 10_000))

    if os.name == 'nt':
        chunk_size = min(chunk_size, 2_000)

    config = HardwareConfig(
        num_workers   = num_workers,
        chunk_size    = chunk_size,
        device        = device,
        initial_batch = initial_batch,
        model_name    = model_name,
        gpu_vram_gb   = gpu_vram_gb,
        total_ram_gb  = total_ram_gb,
        cpu_cores     = cpu_cores,
    )

    _print_banner(config, gpu_available and os.name != 'nt')
    return config
"""================= End function detect ================="""


"""================= Startup function _print_banner ================="""
def _print_banner(cfg: HardwareConfig, gpu_available: bool) -> None:
    gpu_line = (
        f"GPU       : CUDA  {cfg.gpu_vram_gb:.1f} GB VRAM"
        if gpu_available
        else "GPU       : not found — CPU mode"
    )
    print(
        f"\n[HAUP v3.0  Hardware Detector]\n"
        f"  CPU cores : {cfg.cpu_cores}  →  {cfg.num_workers} workers\n"
        f"  RAM       : {cfg.total_ram_gb:.1f} GB  →  chunk_size {cfg.chunk_size:,}\n"
        f"  {gpu_line}\n"
        f"  Device    : {cfg.device}  |  init_batch = {cfg.initial_batch}\n"
    )
"""================= End function _print_banner ================="""