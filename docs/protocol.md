# Protocol

The benchmark fixes each site's images in ascending numeric frame-index order. Floor-rounded boundaries at 15%, 20%, 80%, and 85% define validation, validation buffer, training, test buffer, and test. This is frame-index separation. The common image manifest binds partition membership to image and annotation SHA-256 digests.

Each learned model trains on one source and evaluates its selected checkpoint on all three destination holdouts. Faster R-CNN, U-Net, YOLO26, and RT-DETR use seeds 0–4 for 60 source fits. CA-CFAR, OS-CFAR, Otsu, SS-Otsu, and Wavelet use one deterministic parameter selection per source for 15 source runs. The adjacency comparison uses Faster R-CNN and YOLO26, three sources, and seeds 0–2 for 18 fits. The complete [job configuration](../configs/benchmark-jobs.json) also contains 17 source-only diagnostics.

For adjacency, an equal number of source-training images is replaced by pre-test buffer images. SHA-256 ordering chooses the removed images for each seed. Validation and test IDs remain fixed. Pair core and adjacency fits on the same GPU class, package environment, batch size, precision, and thread policy.

Torch models select checkpoints by source-validation loss. Ultralytics models use source-validation fitness. U-Net selects its pixel threshold on source validation. Learned operating score thresholds are selected there and fixed for all targets. Classical methods rank candidates on source training and rerank five finalists on source validation.

Bounding boxes are scored with Common Objects in Context (COCO) evaluation and `maxDets=[1,10,100]`. Faster R-CNN returns up to 300 detections with score floor zero. YOLO26 and RT-DETR use score floor 0.001. U-Net extracts connected components at its selected pixel threshold. Report average precision at intersection over union (IoU) 0.50, 0.75, and 0.50–0.95, and average recall at 100 detections. These metrics measure agreement with the supplied reference inventory.

## Preprocessing ablation

Faster R-CNN uses a factorial comparison of two intensity views and two training-scale policies. I0 divides stored grayscale values by 255 inside the fixed radar disc. I1 subtracts the stored display pedestal and scales the per-frame upper tail, with a fixed low-support fallback. S0 preserves image geometry. S1 applies deterministic scale jitter to the image and every box, rejecting transformations that move any labeled object outside the canvas or disc.

Three sites, seeds 0–2, and four factorial cells give 36 fits. Six additional DAAN/MANV S0I0 fits use 316 distinct training frames. All 42 fits use 3,432 optimizer updates, batch size two, full precision, no TF32, and source-validation checkpoint selection every 286 updates. Paired cells share initialization and presentation ordering. The frozen external protocol specifies exact transformations, fallback rules, and hashes and is verified before training.

## Supervised adaptation

The adaptation study uses DARC-to-MANV and MANV-to-DARC transfer with seeds 0–2. Six fits adapt only the Faster R-CNN predictor head using 16 target-training frames, selected deterministically across 16 frame-index strata. Each fit performs 64 updates at batch size two and learning rate 0.00025. No target validation or test frame is used for training or selection.

Two support-only diagnostics verify the trainable tensors, unchanged backbone state, finite gradients, and partition isolation before the six scoring fits. Each scoring fit evaluates the source checkpoint, adapted checkpoint, and same-seed target-local checkpoint on the same target holdout and execution backend.
