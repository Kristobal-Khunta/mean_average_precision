# mAP: Mean Average Precision for Object Detection

A simple library for the evaluation of object detectors.

<p align="center">
  <img src="resources/img0.jpeg"/>
</p>

[![Downloads](https://pepy.tech/badge/mean-average-precision)](https://pepy.tech/project/mean-average-precision)
[![Downloads](https://pepy.tech/badge/mean-average-precision/month)](https://pepy.tech/project/mean-average-precision)
[![Downloads](https://pepy.tech/badge/mean-average-precision/week)](https://pepy.tech/project/mean-average-precision)


In practice, a **higher mAP** value indicates a **better performance** of your detector, given your ground-truth and set of classes.

## Example
```python
import numpy as np
from mean_average_precision import MetricBuilder

# [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
gt = np.array([
    [439, 157, 556, 241, 0, 0, 0],
    [437, 246, 518, 351, 0, 0, 0],
    [515, 306, 595, 375, 0, 0, 0],
    [407, 386, 531, 476, 0, 0, 0],
    [544, 419, 621, 476, 0, 0, 0],
    [609, 297, 636, 392, 0, 0, 0]
])

# [xmin, ymin, xmax, ymax, class_id, confidence]
preds = np.array([
    [429, 219, 528, 247, 0, 0.460851],
    [433, 260, 506, 336, 0, 0.269833],
    [518, 314, 603, 369, 0, 0.462608],
    [592, 310, 634, 388, 0, 0.298196],
    [403, 384, 517, 461, 0, 0.382881],
    [405, 429, 519, 470, 0, 0.369369],
    [433, 272, 499, 341, 0, 0.272826],
    [413, 390, 515, 459, 0, 0.619459]
])

# print list of available metrics
print(MetricBuilder.get_metrics_list())

# create metric_fn
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

# add some samples to evaluation
for i in range(10):
    metric_fn.add(preds, gt)

# compute PASCAL VOC metric
print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

# compute PASCAL VOC metric at the all points
print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

# compute metric COCO metric
print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
```


## Example 3d Voxels
```python
import numpy as np
from mean_average_precision import MetricBuilder
from mean_average_precision.adapter import Adapter3dVoxels

# [xmin, ymin, zmin, xmax, ymax, zmax, class_id, difficult, crowd]
gt = np.array(
    [
        [439.0, 157.0, 157.0, 556.0, 241.0, 241.0, 0.0, 0.0, 0.0],
        [437.0, 246.0, 246.0, 518.0, 351.0, 351.0, 0.0, 0.0, 0.0],
        [515.0, 306.0, 306.0, 595.0, 375.0, 375.0, 0.0, 0.0, 0.0],
        [407.0, 386.0, 386.0, 531.0, 476.0, 476.0, 0.0, 0.0, 0.0],
        [544.0, 419.0, 419.0, 621.0, 476.0, 476.0, 0.0, 0.0, 0.0],
        [609.0, 297.0, 297.0, 636.0, 392.0, 392.0, 0.0, 0.0, 0.0],
    ]
)

# [xmin, ymin, zmin, xmax, ymax, zmax, class_id, confidence]
preds = np.array(
    [
        [429.0, 219.0, 219.0, 528.0, 247.0, 247.0, 0.0, 0.461],
        [433.0, 260.0, 260.0, 506.0, 336.0, 336.0, 0.0, 0.27],
        [518.0, 314.0, 314.0, 603.0, 369.0, 369.0, 0.0, 0.463],
        [592.0, 310.0, 310.0, 634.0, 388.0, 388.0, 0.0, 0.298],
        [403.0, 384.0, 384.0, 517.0, 461.0, 461.0, 0.0, 0.383],
        [405.0, 429.0, 429.0, 519.0, 470.0, 470.0, 0.0, 0.369],
        [433.0, 272.0, 272.0, 499.0, 341.0, 341.0, 0.0, 0.273],
        [413.0, 390.0, 390.0, 515.0, 459.0, 459.0, 0.0, 0.619],
    ]
)
# print list of available metrics
print(MetricBuilder.get_metrics_list())

# create metric_fn
metric_fn = MetricBuilder.build_evaluation_metric("map_3d", async_mode=True, num_classes=1, adapter_type = Adapter3dVoxels)

# add some samples to evaluation
for i in range(10):
    metric_fn.add(preds, gt)

# compute PASCAL VOC metric
print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.3, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")

# compute PASCAL VOC metric at the all points
print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")

# compute metric COCO metric
print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
```