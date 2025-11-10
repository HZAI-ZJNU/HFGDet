import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from torch.hub import load
from ultralytics import YOLO, RTDETR
from ultralytics.utils.benchmarks import benchmark
# from thop import profile
# pip install fvcore

if __name__ == "__main__":
    name = 'yolov10n.yaml'
    
    model = YOLO(name)
    model.eval()
    input_tensor = torch.randn(1, 3, 640, 640)  # (batch_size, channels, height, width)
    
    # 计算 FLOPs 论文中的浮点数以此函数为准
    flops = FlopCountAnalysis(model.model, input_tensor)
    total_flops = flops.total()
    print(f"FLOPs: {total_flops / 1e9:.2f} GFLOPs")

    # 计算参数量 参数不采用，仍然以yolo输出的参数为准
    params = parameter_count(model)
    total_params = params[""]
    print(f"Parameters: {total_params / 1e6:.2f} Million")
    
    # 这个和 fvcore计算出的结果基本一致
    # flops, params = profile(model.model, inputs=(input_tensor,))
    # print(f"wtfusion FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）
    # print(f"wtfusion Params: {params / 1e6} M")
    