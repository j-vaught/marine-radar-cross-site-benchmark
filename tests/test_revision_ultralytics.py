from src.revision.ultralytics_models import _model_spec, coco_yolo_lines


def test_empty_and_box_roundtrip():
    image = {"id": 4, "width": 1062, "height": 1030}
    coco = {"images": [image], "categories": [{"id": 7, "name": "target"}], "annotations": []}
    assert coco_yolo_lines(coco, image) == []
    box = [123.25, 456.75, 87.5, 52.25]
    coco["annotations"].append({"image_id": 4, "category_id": 7, "bbox": box})
    cls, cx, cy, width, height = map(float, coco_yolo_lines(coco, image)[0].split())
    assert cls == 0
    actual = [(cx - width / 2) * 1062, (cy - height / 2) * 1030, width * 1062, height * 1030]
    assert all(abs(a - b) < 1e-10 for a, b in zip(actual, box))


def test_scratch_uses_yaml_even_when_pretrained_is_configured():
    for method, variant in [("yolo", "yolo26m"), ("rtdetr", "rtdetr-l")]:
        cfg = {"model": {"variant": variant, "pretrained": True}}
        assert _model_spec(method, "scratch", cfg) == variant + ".yaml"
        assert _model_spec(method, "pretrained", cfg) == variant + ".pt"


def test_letterbox_inverse_preserves_edge_before_clipping():
    from types import SimpleNamespace

    import numpy as np
    import torch

    from src.revision.ultralytics_models import _coordinate_predictor, _letterbox_ratio_pad

    gain = 1024 / 1050
    assert _letterbox_ratio_pad((1024, 1024), (1024, 1050)) == ((gain, gain), (0, 12))
    predictor = _coordinate_predictor().__new__(_coordinate_predictor())
    predictor.model = SimpleNamespace(names={0: "vessel"})
    # Top edge is just inside content, but the incorrect 13-pixel pad clips it to zero.
    pred = torch.tensor([[100.0, 12.25, 200.0, 20.0, 0.8, 0.0]])
    result = predictor.construct_result(
        pred, torch.zeros((1, 3, 1024, 1024)), np.zeros((1024, 1050, 3), dtype=np.uint8), "edge.png"
    )
    assert abs(result.boxes.xyxy[0, 1].item() - 0.25 / gain) < 1e-5
