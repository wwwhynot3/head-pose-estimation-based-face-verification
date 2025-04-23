import torch

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
