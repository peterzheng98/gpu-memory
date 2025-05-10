# Range Analysis

Due to interval analysis and kernel integration, in order to directly demonstrate the effectiveness of our analysis, we have provided a simplified equivalent Python code to analyze the numerical intervals. 

You can run `python resnet.py` to see the results. We output the expected value of each interval and the overhead time for interval analysis.

One expected output is shown below:

```bash
...(Omitted)
[OK]   Layer layer4.1.bn1: [-1.0898, 0.5164]
[OK]   Layer layer4.1.relu: [0.0000, 0.5164]
[OK]   Layer layer4.1.conv2: [-0.2679, 0.4377]
[OK]   Layer layer4.1.bn2: [-3.3684, 8.0954]
[OK]   Layer layer4.1.relu: [0.0000, 8.6762]
[OK]   Layer avgpool: [0.0000, 4.8097]
[OK]   Layer fc: [-4.4105, 5.8543]
[Ok] Inference time: 6.69s for 1000 times, per image time: 0.00669s.
[Ok] Inference time: 6.12s for 1000 times, per image time: 0.00612s.
```