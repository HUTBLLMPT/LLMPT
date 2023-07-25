from collections import OrderedDict#有序字典（Ordered Dictionary）的实现。它是字典（Dictionary）的子类，但与普通的字典不同，OrderedDict会记住键值对的插入顺序。在使用OrderedDict时，键值对的迭代顺序将保持与插入顺序一致。
from typing import Tuple, Union#Union也是typing模块中定义的泛型类型，表示可以是多个类型中的一个。导入了Union后，你可以在类型注解中使用Union[int, float]来表示一个可以是整数或浮点数的类型

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4#conv3时候用到

    def __init__(self, inplanes, planes, stride=1):#inplanes是输入特征图的通道数，planes是输出特征图的通道数
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        #所有卷积层的步长都是1，在步长大于1的时候，在第二次卷积后使用平均池化
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)#二维卷积层，卷积核大小是1*1，无偏置项，输入与输出除了通道数量之外都一样
        self.bn1 = nn.BatchNorm2d(planes)#归一化
        self.relu1 = nn.ReLU(inplace=True)#relu function，inplace=True：表示在原地执行ReLU激活函数，即将激活函数的计算结果直接覆盖在输入上，节省内存空间。

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)#卷积核3*3，padding=1，通道数不变，输入与输出维度也没有发生变化
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()#Identity()函数就是输入输出不变，不进行任何操作

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)#扩张输出的通道数为输入的4倍
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None#下采样，指示标记
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            """这样的下采样层通常用于在残差网络（ResNet）或类似的网络结构中，在不同块（block）之间的维度变换或分辨率下降的情况下进行处理，以便实现特征的整合和维度的一致性。
通过条件判断创建不同的下采样层，可以根据不同的条件灵活地构建模型结构，并确保模型的正确性和一致性。"""
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            #不满足这两个条件的话就创建一个下采样层，下采样层用avgpool进行预处理，其中的卷积层步长是1

            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),#池化的步长是stribe
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        """前向传输的定义"""
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity#残查网络，加上了输入
        out = self.relu3(out)#残差之后采用的relu function
        return out


class AttentionPool2d(nn.Module):
    """基于注意力机制的二维池化层"""
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)#位置编码
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)#output_dim 是一个可选参数，如果不提供，则默认与 embed_dim 相同。此处这个c指的是上下文，通过注意力权重对输入序列进行加权求和得到的。
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC ，使用 torch.cat 在第一个维度上拼接两个张量。第一个张量是 x 在第 0 维度上的均值，即对批次中所有样本的特征进行求均值操作得到的张量。第二个张量是原始的 x 张量。通过这个操作，将均值特征添加到输入 x 中，使得张量的第一个维度多出一个样本。
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC，田间位置嵌入，并且输出的x与输入的x格式保持不变
        x, _ = F.multi_head_attention_forward(#使用 F.multi_head_attention_forward 对输入张量 x 进行多头注意力计算，并将计算后的结果赋值给 x，忽略了注意力权重的返回。
            query=x[:1], key=x, value=x,#q的尺寸与x的第一个样本的大小一致，k和v大小和x一致
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),#输入投影的偏置项，由查询、键和值的偏置项组合而成。
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False#函数的返回值包括两个部分，分别是计算后的注意力张量 x 和注意力权重（如果 need_weights 设置为 True，则返回注意力权重张量）
        )
        return x.squeeze(0)#将输出张量 x 的第一个维度压缩，去除多出的样本维度，并作为模型的输出返回。


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions（抗锯齿跨步卷积）, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)#输出通道数32，维度112
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)#输出通道数与改层的输入通道数一致32，维度112
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)#输出通道数64，维度112
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)#窗口是2*2，维度56

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """创建一层上面定义的bottleneck后，连续再创建block次，后面的这几次的最后一层是不一样的，有下采样层"""
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16.
    是对 PyTorch 的 nn.LayerNorm 的子类进行的扩展。重写了forward函数。目的是为了处理在输入张量 x 的数据类型为 fp16 时，可能由于层归一化操作中的计算精度问题导致的错误。
    通过在层归一化前将输入张量转换为 float32 进行计算，然后再将结果转换回原始数据类型，可以确保在 fp16 数据类型下的正确性。"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype#首先保存了输入张量 x 的原始数据类型 orig_type。
        ret = super().forward(x.type(torch.float32))#调用父类 super().forward(x.type(torch.float32)) 来将输入张量 x 的数据类型转换为 float32（单精度浮点数）并进行层归一化操作。
        return ret.type(orig_type)#通过 ret.type(orig_type) 将结果转换回原始的数据类型，并作为输出返回。


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):#残查注意力块
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)#第一个参数是嵌入的维度embedding_dim，第二个参数多头
        self.ln_1 = LayerNorm(d_model)#自己定义的层归一化
        self.mlp = nn.Sequential(OrderedDict([#定义了多层感知机
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask#注意力机制是否带有mask操作

    def attention(self, x: torch.Tensor):#这里是是否带有mask操作的判断函数
        '''将输入 x 作为查询、键和值传入 self.attn 方法中进行注意力计算。设置 need_weights=False 表示不需要计算和返回注意力权重。同时，将之前准备好的掩码（self.attn_mask）传递给 attn_mask 参数，以确保掩码在注意力计算中被应用。
最后，函数将注意力计算结果的第一个元素返回'''
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))#输入加上层归一化后的注意力
        x = x + self.mlp(self.ln_2(x))#输入加上第二层归一化后的多层感知机结果
        return x


class Transformer(nn.Module):#Transformer：建layers层的残查注意力块
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):#Vision Transformer
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution#输入图片的宽度
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)#卷积核和步长此处设置的始终一样

        scale = width ** -0.5#在注意力机制中，为了控制注意力的强度和稳定性，通常会对注意力权重进行缩放。通过将权重除以一个缩放因子，可以确保注意力权重在计算过程中不会过大或过小。这个缩放因子一般与输入的维度有关，通常会取倒数的平方根。
        self.class_embedding = nn.Parameter(scale * torch.randn(width))#诸如梯度和偏置这样的参数定义，通过将乘以 scale 的随机张量 torch.randn(width) 包装在 nn.Parameter 中，class_embedding 成为一个可学习的参数。这意味着在模型的训练过程中，class_embedding 的值将被优化器根据损失函数进行更新。可学习的类别嵌入（class embedding）。类别嵌入是一种将类别信息转换为连续值的方法，通常用于在某些任务中对类别信息进行编码。
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)#自定义的层归一化

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid],通道数是width
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]，一个矩阵变成了一个向量
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]，改变了前后顺序
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]，先将class_embedding扩展成和x相同的形状，在第一维上与x拼接
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])#使用ln_post层对张量x进行标准化操作，并只保留第一个位置的特征。

        if self.proj is not None:#如果存在proj投影层，则将张量x与proj矩阵相乘。
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,#嵌入维度
                 # vision，图像
                 image_resolution: int,#输入图像的尺寸
                 vision_layers: Union[Tuple[int, int, int, int], int],#视觉层的结构。可以是一个四元组或一个整数。
                 vision_width: int,#视觉层的宽度（通道数）
                 vision_patch_size: int,#视觉层的补丁尺寸
                 # text，文本
                 context_length: int,#文本上下文长度
                 vocab_size: int,#词汇表的大小
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()#在初始化函数中，调用了父类nn.Module的__init__方法。

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):#用于检查一个对象是否属于指定的类型
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(#调用了上面定义的modified resnet,
                layers=vision_layers,#这里的layer是多个数值，控制resnet中相同类型的网络用了多少次的参数
                output_dim=embed_dim,
                heads=vision_heads,#其中用了基于注意力的池化层，该网络中并没有transformer层
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(#用vision transformer处理图像
                input_resolution=image_resolution,
                patch_size=vision_patch_size,#卷积核和步长的定义
                width=vision_width,
                layers=vision_layers,#控制其中的transformer层有多少层
                heads=vision_heads,#控制其中transformer层的头数
                output_dim=embed_dim
            )

        self.transformer = Transformer(#上面的网络搭建完成后全部要经过这一个transformer
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        #上面是image encoder网络的搭建，下面是文本的处理
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)#嵌入层,第一个参数指的是表示嵌入层的输入空间大小，第二个参数指的是嵌入向量的维度
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))#context_length 表示文本上下文的长度，transformer_width 表示Transformer模型的宽度（维度）。初始化是空的，即它的值是随机的。具体的初始化操作在其他地方进行，这行代码只是创建了一个可训练的参数，并为其分配了适当的形状。
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))#映射层，这里的transformer_width是隐藏层中向量的维度。
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))#用于缩放模型输出的标量值。用来调整输出分布的范围和精确度。

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)#对模型中的 token_embedding 参数进行正态分布初始化。
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):#也是用来判断image encoder用的是resnet还是vision transformer
            if self.visual.attnpool is not None:#基于注意力机制的池化层
                std = self.visual.attnpool.c_proj.in_features ** -0.5#in_features 表示  Attention Pooling 模块中 c_proj 的输入特征数，应该指的是embedding_dims。
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)#初始化基于注意力机制池化层中的q,k,v,c
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):#但凡有第三层的，全部0初始化。对每个 resnet_block 中的指定的 Batch Normalization 层的权重。通过将权重初始化为零，可以帮助模型在初始训练阶段更快地学习适当的偏移量和缩放因子。
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:#若文本映射层存在的话，按照以下的标准差初始化权重
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        '''构建注意力掩码函数，该函数将创建一个指定尺寸的空的张量（mask），然后将其填充为负无穷大。
         之后，函数将把 mask 的下三角部分（包括主对角线）置零（即遮盖），从而只保留上三角部分作为注意力的可见区域。 最
         后，函数将返回生成的注意力掩码。'''
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property#用于将类中的方法转换为属性。可以像访问属性一样来访问它，而不需要使用函数调用语法。
    def dtype(self):
        return self.visual.conv1.weight.dtype#resnet或者VIT第一层权重的数据类型，诸如float, int32,64...

    def encode_image(self, image):#返回值是经过图像编码器编码完成后的结果
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)#只过了一个transformer块
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection#应该是获取了每个文字最后那个符号的嵌入，整个前面所有字的嵌入都被包含进去了

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features，归一化图像与文本特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits，通过矩阵运算，计算余弦相似性
        logit_scale = self.logit_scale.exp()#self.logit_scale 的值作为指数运算的底数，然后取得其指数值。这个操作可以调整分布范围，用来缩放得分。调整分布范围的
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16，转换参数格式，主要目的是减少模型在计算资源上的占用"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):#对于这三个层，将其权重参数和偏置参数转换为半精度浮点数格式
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):#包括输入、查询、键、值的投影权重，投影偏置，以及键和值的偏置。
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)#将 _convert_weights_to_fp16 函数应用到模型的所有参数上，实现将适用的模型参数转换为半精度浮点数格式。


def build_model(state_dict: dict):#构建模型函数，用于根据给定的状态字典构建模型。
    '''此处应该是在运行的时候，导入openai训练好的参数'''
    vit = "visual.proj" in state_dict#对给定的状态字典进行检查，判断字典中是否包含键值为 "visual.proj" 的项。将结果存储在变量 vit 中，用于后续的条件判断或其他操作。

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32
    #获取参数的形状
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))#通过计算预训练参数中以 "transformer.resblocks" 开头的键中不同的层的数量，得到 Transformer 模型的层数。
    #使用CLIP类构建模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:#通过遍历一些指定的键，并在状态字典中删除这些键，以保证状态字典与模型的加载匹配。
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)#转换为float16
    model.load_state_dict(state_dict)#使用状态字典加载模型
    return model.eval()#用于设置模型处于推理模式（evaluation mode）
