import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "./pretrained/zimage",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [可选] 注意力后端
# 默认情况下，Diffusers 使用 SDPA。如果支持的话，切换到 Flash Attention 以获得更高的效率：
# pipe.transformer.set_attention_backend(";flash";)    # 启用 Flash-Attention-2
# pipe.transformer.set_attention_backend(";_flash_3";) # 启用 Flash-Attention-3

# [可选] 模型编译
# 编译 DiT 模型可以加速推理，但第一次运行时将需要更长的时间进行编译。
# pipe.transformer.compile()

# [可选] CPU 卸载
# 为内存受限的设备启用 CPU 卸载。
# pipe.enable_model_cpu_offload()

prompt = "身着红色汉服的中国年轻女子，衣饰绣有精美繁复的刺绣。妆容精致无瑕，额间绘有红色花卉图案。高耸繁复的发髻上佩戴金色凤凰头饰，点缀红花与珠串。她手持一柄圆形折扇，扇面绘有仕女、树木与飞鸟。一道霓虹闪电造型的灯（⚡️）悬浮于她伸出的左掌上方，散发出明亮的黄色光芒。背景为柔和夜光下的户外场景，远处可见西安大雁塔的层叠剪影，以及模糊而绚丽的彩色灯火。"

# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("test.png")
