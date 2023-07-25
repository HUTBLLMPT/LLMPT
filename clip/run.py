import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)#加载模型
text = clip.tokenize(["a diagram", "a dog", "a cat","company","school"]).to(device)#给定需要用到的类，并经历分词器
image_input=Image.open("截屏2023-07-10 16.35.20.png")
image = preprocess(image_input).unsqueeze(0).to(device)#诸如通道这些图像信息进行修改

image_input.show()#展示刚导入的这张图片

with torch.no_grad():
    image_features = model.encode_image(image)#图像编码器
    text_features = model.encode_text(text)#文本编码器

    logits_per_image, logits_per_text = model(image, text)#执行推理，返回预测标签的得分
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()#转换为softmax
print("text:         a diagram      a dog       a cat")
print("Label probs:", probs)
