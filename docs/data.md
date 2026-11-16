# Data and artifact access

Request Dataset 2, Radar Images & AIS Data, from the [official DLR Marine Radar access page](https://www.dlr.de/en/kn/about-us/departments/nautical-systems/real-world-marine-radar-datasets-for-target-tracking) and comply with the dataset's separate access and use conditions. This repository does not distribute radar images or annotations. The code license does not grant rights to the dataset or model weights.

Training expects a processed directory with `annotations/all.json` and `images/`. The annotation file uses COCO bounding boxes in pixel `[x,y,width,height]` coordinates and one foreground category. Each image record includes integer `id`, `width`, `height`, and numeric `frame`, a relative `file_name`, and `dataset` equal to DAAN, DARC, or MANV. The frozen benchmark uses 1050 by 1024 grayscale rasters and a circular mask centered at pixel (531,515) with radius 466.

For exact reproduction, use the authorized processed inputs matching the frozen hashes below. Recreating the manifest from a different image or annotation release defines a different experiment. The ablation and adaptation runners reject mismatched frozen inputs.

```text
split_manifest.json
03b7d1d48e5faac7872a04f8fb23580a748ca1521539e99b0e54bf45c7d8f4aa
annotations/all.json
29d602bb9a5d764691e44fe697ba3449348a7e9a9cd05b8c19a415b1fa8d5911
processed-frozen-protocol.json
14a06aad755ef18a9fb4b90371ea036057af22e57f28e7a892278ac3e721ea16
adaptation protocol
d7c4d4f2395bdc5fa8bb00f9badf12fdd57c94f3c48ae5d1428ed2a6352dc549
```

## Artifact download interface

The downloader lists files from a public Zenodo record and retrieves only explicitly selected files. It verifies the repository-supplied checksum before completing each download and does not extract archives. Set `RECORD` to the published research-artifact record number and select filenames from its listing.

```sh
uv run --locked python -m src.revision.download --record "$RECORD"
uv run --locked python -m src.revision.download --record "$RECORD" --file "$FILE" --output "$ARTIFACTS"
```

Read each artifact's license and access notes before use. A record may contain only metadata or selected model families. Obtain any separately restricted data or pretrained assets from their owners. After downloading, verify the package's SHA-256 manifest before extracting into a new directory.

## Pretrained initialization

U-Net initializes from scratch. Faster R-CNN initializes from torchvision's `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` COCO_V1 asset. Its expected SHA-256 is `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Place the authorized asset in `$TORCH_HOME/hub/checkpoints/` using its upstream filename. The ablation and adaptation runners verify this asset before model construction.

The Ultralytics runners use `yolo26m.pt`, `rtdetr-l.pt`, and the automatic mixed-precision probe `yolo26n.pt`. Place these authorized upstream assets in `WEIGHTS`. The [asset configuration](../configs/pretrained-assets.json) records their upstream URLs and benchmark SHA-256 digests. Verify those digests before training. The installed libraries can retrieve pretrained assets during normal model construction; the verified ablation and adaptation paths require the Faster R-CNN asset to be staged in advance.

## Reconstructing processed inputs

The [raw conversion guide](preprocessing.md) provides a hash-locked reconstruction command for legally obtained original captures. It produces a candidate processed dataset and an explicit pixel/annotation comparison when authorized benchmark inputs are also available. Original raw captures were unavailable for validation of this implementation, so reconstruction has been tested on synthetic inputs. The preprocessing calibration and upstream label provenance remain unresolved. A successful conversion does not establish equality with the frozen benchmark inputs.
