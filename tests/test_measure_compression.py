"""cluster_weights: k-means weight-sharing actually applied to a model, not just sized."""
import torch
import torch.nn as nn

from scripts.measure_compression import cluster_weights


def _tiny_model():
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.Linear(8, 4))


def test_cluster_weights_limits_unique_values_and_preserves_shape():
    model = _tiny_model()
    original_conv_weight = model[0].weight.detach().clone()

    clustered, codebook_mb, index_mb = cluster_weights(model, n_clusters=4, seed=42)

    conv_out, linear_out = clustered[0], clustered[1]
    assert conv_out.weight.shape == model[0].weight.shape
    assert linear_out.weight.shape == model[1].weight.shape

    all_clustered = torch.cat([conv_out.weight.flatten(), linear_out.weight.flatten()])
    assert torch.unique(all_clustered).numel() <= 4
    assert codebook_mb > 0
    assert index_mb > 0

    # original model must be untouched (deep copy, not in-place)
    assert torch.equal(model[0].weight.detach(), original_conv_weight)


def test_cluster_weights_caps_k_at_number_of_unique_values():
    model = nn.Sequential(nn.Conv2d(1, 1, 1))
    with torch.no_grad():
        model[0].weight.fill_(1.0)  # every weight identical -> only 1 unique value to cluster
    clustered, _, _ = cluster_weights(model, n_clusters=64, seed=42)
    assert torch.unique(clustered[0].weight).numel() == 1
