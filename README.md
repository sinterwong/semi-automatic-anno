# semi-automatic-anno

## introduce

Based onnxruntime to inference models or generating the results of algo to iter model. 

## requirements 

- onnxruntime >= 1.7.0
- opencv-python >= 4.2

## Run demo

All demos is named in the format main_xxx_.py, You can run demo in the following ways.

```bash
export PYTHONPATH=/home/wangxt/workspace/projects/pyonnx-example:$PYTHONPATH
python samples/main_gesture.py --video_path data/smoking.flv --out_root data/main_result/gesture
```

