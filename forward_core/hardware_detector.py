"""
File Summary:
Hardware auto-detector for HAUP v2.0. Auto-configures workers, chunk size, and device based on system resources.

====================================================================
SYSTEM PIPELINE FLOW (Architecture + Object Interaction)
====================================================================

HardwareDetector()
||
├── detect()  [Function] --------------------------------> Main detection entry point
│       │
│       ├── CPU Detection --------------------------------> Physical and logical cores
│       │       │
│       │       └── psutil.cpu_count() -----------------> Core count extraction
│       │
│       ├── Memory Detection ----------------------------> RAM analysis
│       │       │
│       │       └── psutil.virtual_memory() ------------> Total system RAM
│       │
│       ├── GPU Detection --------------------------------> CUDA availability
│       │       │
│       │       ├── torch.cuda.is_available() -----------> GPU presence check
│       │       │
│       │       └── torch.cuda.get_device_properties() --> VRAM detection
│       │
│       ├── Worker Calculation --------------------------> Process spawning
│       │       │
│       │       └── min(cpu_physical - 1, MAX_WORKERS) --> Conservative scaling
│       │
│       ├── Chunk Sizing ---------------------------------> Memory-based batching
│       │       │
│       │       └── (RAM * fraction) / bytes_per_row ----> Chunk calculation
│       │
│       └── Device & Batch Config -----------------------> Performance tuning
│               │
│               └── GPU/CPU batch sizing ----------------> Hardware optimization
│
└── HardwareConfig  [Class] ----------------------------> Configuration container

====================================================================
FUNCTION / CLASS ENTRY POINT MARKERS
====================================================================
"""

import os
import psutil
import torch
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    num_workers:   int
    chunk_size:    int
    device:        str
    initial_batch: int
    model_name:    str = "all-MiniLM-L6-v2"
    cpu_physical:  int   = 0
    cpu_logical:   int   = 0
    total_ram_gb:  float = 0.0
    gpu_available: bool  = False
    gpu_vram_gb:   float = 0.0


class HardwareDetector:

    BYTES_PER_ROW   = 1536
    RAM_FRACTION    = 0.10
    MIN_CHUNK       = 10
    MAX_CHUNK       = 500
    MAX_WORKERS     = 8
    LARGE_GPU_VRAM  = 8.0
    BATCH_LARGE_GPU = 64
    BATCH_SMALL_GPU = 32
    BATCH_CPU       = 32

    def detect(self) -> HardwareConfig:
        
        try:
            cpu_physical  = psutil.cpu_count(logical=False) or 1
            cpu_logical   = os.cpu_count() or 1
            total_ram_gb  = psutil.virtual_memory().total / 1e9
        except Exception:
            cpu_physical = 2
            cpu_logical = 4
            total_ram_gb = 8.0
        
        # ===========Might be required GPU ========
        try:
            gpu_available = torch.cuda.is_available()
            gpu_vram_gb   = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_available else 0.0
            )
        except Exception:
            gpu_available = False
            gpu_vram_gb = 0.0

        base_workers = min(cpu_physical - 1, self.MAX_WORKERS)
        num_workers = max(base_workers, 1)
        
        if os.name == 'nt':
            num_workers = min(num_workers, 3)

        raw_chunk  = (total_ram_gb * self.RAM_FRACTION * 1e9) / self.BYTES_PER_ROW
        chunk_size = int(self._clamp(raw_chunk, self.MIN_CHUNK, self.MAX_CHUNK))
        
        if os.name == 'nt':
            chunk_size = min(chunk_size, 100)

        # ===========Might be required GPU ========
        if os.name == 'nt' or not gpu_available:
            device        = 'cpu'
            initial_batch = self.BATCH_CPU
        else:
            device        = 'cuda'
            initial_batch = (self.BATCH_LARGE_GPU
                             if gpu_vram_gb > self.LARGE_GPU_VRAM
                             else self.BATCH_SMALL_GPU)

        config = HardwareConfig(
            num_workers   = num_workers,
            chunk_size    = chunk_size,
            device        = device,
            initial_batch = initial_batch,
            cpu_physical  = cpu_physical,
            cpu_logical   = cpu_logical,
            total_ram_gb  = total_ram_gb,
            gpu_available = gpu_available and os.name != 'nt',
            gpu_vram_gb   = gpu_vram_gb,
        )
        
        return config


    @staticmethod
    def _clamp(value: float, lo: int, hi: int) -> float:
        
        result = max(lo, min(hi, value))
        
        return result