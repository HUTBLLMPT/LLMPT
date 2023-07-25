import numpy as np
import pytest#python的软件测试框架
import torch
from PIL import Image

import clip


@pytest.mark.parametrize('model_name', clip.available_models())#运行同一个测试函数的多个实例时，提供不同的参数组合。这个函数接受两个参数：一个是参数名称的字符串列表，另一个是参数值的列表或元组。使用@pytest.mark.parametrize可以将参数化的测试用例组织成表格形式，每一行代表一个实例，每一列代表一个参数。
def test_consistency(model_name):
    device = "cpu"
    jit_model, transform = clip.load(model_name, device=device, jit=True)#函数来自clip文件夹下面clip的python文件中load函数，JIT版本
    py_model, _ = clip.load(model_name, device=device, jit=False)#非JIT版本的预训练模型和变换函数

    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)#.unsqueeze(0)将单个图像转换为批处理的形式
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    with torch.no_grad():#执行模型推理，以下的模型参数不参与梯度跟踪和计算梯度
        logits_per_image, _ = jit_model(image, text)#传入图像和文本张量，获得推理结果
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()#使用 softmax获得概率分布

        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)#比较两个模型输出的概率数组是否在给定的误差范围内相等。atol参数指定绝对误差容许范围，rtol参数指定相对误差容许范围。如果所有元素的差距都在指定容许范围内，断言就会通过。
