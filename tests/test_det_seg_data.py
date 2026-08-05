"""VOCDetectionDataset XML-parsing + box-rescale coverage - previously untested.

Injects a stub in place of torchvision.datasets.VOCDetection so no VOC download is needed.
"""
from PIL import Image

from ml.det_seg_data import VOCDetectionDataset


class _StubVOC:
    """Minimal stand-in for VOCDetection: one 100x50 image with one known + one unknown class."""

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.new("RGB", (100, 50))
        target = {"annotation": {"object": [
            {"name": "cat", "bndbox": {"xmin": "10", "ymin": "10", "xmax": "50", "ymax": "40"}},
            {"name": "not_a_voc_class", "bndbox": {"xmin": "0", "ymin": "0", "xmax": "5", "ymax": "5"}},
        ]}}
        return image, target


class _StubVOCNoObjects:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return Image.new("RGB", (64, 64)), {"annotation": {"object": []}}


def test_getitem_parses_known_classes_and_rescales_boxes_to_img_size():
    ds = VOCDetectionDataset(_StubVOC(), img_size=200, augment=False)

    image, target = ds[0]

    assert len(ds) == 1
    assert image.shape == (3, 200, 200)
    assert target["labels"].tolist() == [8]  # only "cat" (VOC_CLASSES index 7 -> label 8) survives
    # 100x50 -> 200x200 is a 2x horizontal, 4x vertical scale
    assert target["boxes"].tolist() == [[20.0, 40.0, 100.0, 160.0]]


def test_getitem_handles_image_with_no_annotated_objects():
    ds = VOCDetectionDataset(_StubVOCNoObjects(), img_size=64, augment=False)

    _, target = ds[0]

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
