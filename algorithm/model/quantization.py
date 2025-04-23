import platform

import torch


def auto_select_backend():
    """自动选择量化后端"""
    arch = platform.machine().lower()
    print(f'Detected Architecture: {arch}')
    if arch in ['x86_64', 'amd64']:
        backend = 'fbgemm'  # x86架构选择FBGEMM[5,7](@ref)
    elif arch in ['aarch64', 'armv8l']:
        backend = 'qnnpack'  # ARM架构选择QNNPACK[3,5](@ref)
    else:
        raise RuntimeError(f"Unsupported architecture: {arch}")
    print(f'Backend: {backend}')
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f"{backend} backend not available")

    torch.backends.quantized.engine = backend
    return backend

def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec=None,
        dtype=torch.qint8,
        mapping=None,
        inplace=False
    )
    print("Quantization complete.")
    return quantized_model
