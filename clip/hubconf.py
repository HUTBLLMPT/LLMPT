from clip.clip import tokenize as _tokenize, load as _load, available_models as _available_models
import re
import string
#于创建CLIP模型库的入口点（entry points）。它定义了用于加载和使用CLIP模型的函数，并提供了一些辅助函数。
dependencies = ["torch", "torchvision", "ftfy", "regex", "tqdm"]

# For compatibility (cannot include special characters in function name)
model_functions = { model: re.sub(f'[{string.punctuation}]', '_', model) for model in _available_models()}#该行代码的作用是创建一个字典，将每个 CLIP 模型的名称作为键，对应的替换了标点字符的字符串作为值。re.sub() 函数，它使用正则表达式的替换功能，将 model 中的标点字符替换为下划线 _。

def _create_hub_entrypoint(model):
    def entrypoint(**kwargs):      
        return _load(model, **kwargs)
    #创建每个 CLIP 模型的加载入口点函数。接收一个模型名称 model 作为输入参数，然后创建一个内部函数 entrypoint，该函数允许用户通过关键字参数传递选项并加载相应的 CLIP 模型。
    entrypoint.__doc__ = f"""Loads the {model} CLIP model

        Parameters
        ----------
        device : Union[str, torch.device]
            The device to put the loaded model

        jit : bool
            Whether to load the optimized JIT model or more hackable non-JIT model (default).

        download_root: str
            path to download the model files; by default, it uses "~/.cache/clip"

        Returns
        -------
        model : torch.nn.Module
            The {model} CLIP model

        preprocess : Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
        """
    return entrypoint

def tokenize():
    return _tokenize

_entrypoints = {model_functions[model]: _create_hub_entrypoint(model) for model in _available_models()}#键是模型函数名（经过处理的字符串），值是对应模型的加载入口点函数

globals().update(_entrypoints)#全局命名空间，在代码的其他部分直接使用这些函数，无需使用字典访问它们。