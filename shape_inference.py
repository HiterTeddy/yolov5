import onnx
from onnx import shape_inference
import torch

if __name__ == '__main__':
    path = "./yolov5s.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
    ckpt = torch.load("yolov5s.pt", map_location='cpu')
