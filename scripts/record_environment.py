import json
import os
import platform
from pathlib import Path
import psutil
import torch
import transformers
import onnx
import onnxruntime as ort

def get_env():
    # RAM
    vm = psutil.virtual_memory()
    ram_gb = round(vm.total / (1024**3), 2)
    
    # GPU
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    # CPU
    cpu_model = "13th Gen Intel(R) Core(TM) i7-13700HX"
    cpu_cores_physical = psutil.cpu_count(logical=False)
    cpu_cores_logical = psutil.cpu_count(logical=True)
    
    env_info = {
        "hardware": {
            "device_class": "laptop-class x86 CPU",
            "cpu_model": cpu_model,
            "physical_cores": cpu_cores_physical,
            "logical_processors": cpu_cores_logical,
            "ram_total_gb": ram_gb,
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "gpu_count": gpu_count
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0]
        },
        "software_environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda if gpu_available else None,
            "transformers_version": transformers.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__
        }
    }
    
    out_path = Path("ARN/results/environment.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    print(f"Saved environment specs to {out_path}")
    print(json.dumps(env_info, indent=2))

if __name__ == "__main__":
    get_env()
