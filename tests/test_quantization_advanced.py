"""Pure bit-budgeting helpers from quantization_advanced.py - previously untested."""
import torch.nn as nn

from ml.quantization_advanced import (
    assign_mixed_precision, theoretical_size_mb, winograd_eligible_layers,
)


def _tiny_model():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),  # "0": winograd-eligible (3x3, stride 1, groups 1)
        nn.Conv2d(4, 4, 1),             # "1": not eligible (1x1 kernel)
        nn.Flatten(),                   # "2": not quantizable
        nn.Linear(4, 2),                # "3"
    )


def test_winograd_eligible_layers_filters_to_dense_3x3_stride1():
    names = [name for name, _ in winograd_eligible_layers(_tiny_model())]
    assert names == ["0"]


def test_assign_mixed_precision_gives_int8_to_the_most_sensitive_layers():
    model = _tiny_model()
    sensitivities = {"0": 0.9, "1": 0.1, "3": 0.5}  # names match model.named_modules()

    bits_map = assign_mixed_precision(model, sensitivities, int8_ratio=0.34)

    assert bits_map == {"0": 8, "1": 4, "3": 4}  # round(3 * 0.34) = 1 -> only the top layer
    assert model[0].qconfig is not None and model[3].qconfig is not None


def test_theoretical_size_mb_uses_per_layer_bits_map_and_8bit_for_biases():
    model = nn.Sequential(nn.Linear(4, 2))  # weight: 8 elems (2D), bias: 2 elems (1D, always 8-bit)

    default_mb = theoretical_size_mb(model, w_bits=8)
    assert default_mb == (8 * 8 + 2 * 8) / 8 / (1024 ** 2)

    mixed_mb = theoretical_size_mb(model, bits_map={"0": 4})
    assert mixed_mb == (8 * 4 + 2 * 8) / 8 / (1024 ** 2)
