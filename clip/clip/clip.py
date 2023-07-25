import hashlib   #提供加密和哈希算法功能
import os
import urllib #处理URL和进行网络请求
import warnings
from typing import Any, Union, List#用于注解类型
from pkg_resources import packaging#管理和访问python包中的资源

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize#compose函数用来将多个图像变换操作组合到一起，resize用来调整大小,centercrop是中心裁剪, totensor将图像或者numpy数组转换为torch。
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode#导入的枚举类interpolationmode，用于指定图像插值的模式。插值是在调整图像大小时用于计算新像素值的方法
    BICUBIC = InterpolationMode.BICUBIC#双三次插值法（Bicubic Interpolation）
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()#分词器，并且把utf-8转换为unicode

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)#创建多级目录，并且提供了一个方便的参数来控制是否忽略已存在的目录。
    filename = os.path.basename(url)#basename() 函数来获取 URL 中的文件名部分

    expected_sha256 = url.split("/")[-2]#使用分割字符 "/" 来从给定 URL 中提取出预期的 SHA-256 哈希值的操作。
    download_target = os.path.join(root, filename)#合并路径

    if os.path.exists(download_target) and not os.path.isfile(download_target):#判断语句的第二个函数是判断路径是否对应一个文件。
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:#验证下载文件的完整性，通过比较哈希值来确保下载的文件与预期的一致。
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:#打开给定的 URL 并将其作为源文件，然后打开下载目标路径的文件并将其作为输出目标：
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:#实现一个带有进度条的文件下载过程
            while True:
                buffer = source.read(8192)#从源文件 source 中读取一个缓冲区（这里是 8192 字节）
                if not buffer:#如果读取的内容为空，则说明已经读取完整个文件，退出循环。
                    break

                output.write(buffer)
                loop.update(len(buffer))#更新进度条的当前值

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:#同54行，验证下载文件的完整性
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target#返回的是下载文件的存储路径


def _convert_image_to_rgb(image):#图像转换为RGB
    return image.convert("RGB")


def _transform(n_px):#像素数为参数
    return Compose([
        Resize(n_px, interpolation=BICUBIC),#调整图像大小为 n_px，插值方法使用双三次插值法。
        CenterCrop(n_px),#中心裁剪图像为 n_px 大小。
        _convert_image_to_rgb,#将图像转换为RGB模式。
        ToTensor(),#转换为张量
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),#归一化
    ])


def available_models() -> List[str]:#调用这个函数来获取可用的CLIP模型的名称列表
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:# 如果在模型列表中，则下载模型并设置模型路径
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):# 如果名字不在模型列表中，则检查是否为一个文件路径，如果是文件路径，则将其设置为模型路径
        model_path = name
    else: # 如果既不在模型列表中，也不是文件路径，则抛出错误
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:# 如果文件不是JIT归档文件，则发出警告并将jit设置为False
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)# 构建模型并根据提供的state_dict或者模型的state_dict进行初始化
        if str(device) == "cpu":
            model.float()# 如果设备为CPU，则将模型转换为float类型
        return model, _transform(model.visual.input_resolution)# 返回模型和模型输入的转换函数

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])# 创建一个用于跟踪设备的`torch.jit.trace`对象
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]# 在跟踪对象的图中查找包含"Device"的prim::Constant节点，并获取最后一个节点

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        # 遍历图中的每个节点
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                # 判断节点是否包含"value"属性，并且是否以"cuda"开头
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    # 将节点的属性复制为device_node的属性
                    node.copyAttributes(device_node)

    model.apply(patch_device)  # 对整个模型应用patch_device函数，即遍历模型中的所有模块，并执行patch_device函数对模块进行修复

    patch_device(model.encode_image)  # 对模型中的encode_image模块应用patch_device函数，修复模块中的节点设备信息

    patch_device(model.encode_text)  # 对模型中的encode_text模块应用patch_device函数，修复模块中的节点设备信息

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        # 创建一个用于跟踪float类型的`torch.jit.trace`对象
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        # 在跟踪对象的图中查找"aten::to"节点，并获取其输入节点
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        # 定义一个用于修复float类型的patch_float函数
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            # 遍历图中的每个节点
            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    # 在aten::to节点的输入中寻找dtype为5的参数，即float类型
                    for i in [1, 2]:  # dtype可以作为aten::to的第二个或第三个参数
                        if inputs[i].node()["value"] == 5:
                            # 将该节点的属性复制为float_node的属性
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)  # 对整个模型应用patch_float函数修复float类型节点
        patch_float(model.encode_image)  # 对模型中的encode_image模块应用patch_float函数修复float类型节点
        patch_float(model.encode_text)  # 对模型中的encode_text模块应用patch_float函数修复float类型节点

        model.float()  # 将整个模型转换为float类型

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length上下文长度

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    返回给定输入字符串的标记化表示

    参数
    ----------
    texts : Union[str, List[str]]
        要标记化的输入字符串或输入字符串的列表

    context_length : int
        使用的上下文长度；所有CLIP模型使用77作为上下文长度

    truncate: bool
        如果编码长度超过上下文长度是否截断文字

    返回
    -------
    一个二维张量，包含结果标记，形状为[输入字符串数量，上下文长度]。
    当torch版本低于1.8.0时，返回LongTensor，因为旧版本的index_select要求索引为long类型。
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
