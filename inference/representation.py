import numpy as np
import torch
from PIL import Image
torch.set_grad_enabled(False)

from mashiro.hyperstroke.modules.vqgan.model import VQModel as HyperstrokeVQModel


prev = Image.open("inference/images/00090_prev.png").convert("RGB").resize((256, 256))
next = Image.open("inference/images/00090_next.png").convert("RGB").resize((256, 256))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = HyperstrokeVQModel.from_pretrained("gyrojeff/Hyperstroke-VQ-Illustration")
model = model.to(device)

prev = np.array(prev).astype(np.uint8)
next = np.array(next).astype(np.uint8)

# concat prev and next on channel axis
image = np.concatenate([prev, next], axis=2)

x = torch.Tensor((image / 127.5 - 1.0).astype(np.float32)).unsqueeze(0)
x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
x = x.to(device)

_, _, [_, _, indicies] = model.encode(x)

d = model.decode_code(indicies[None, :])
d = torch.clamp(d, -1., 1.).cpu()
d = (((d + 1.0) / 2.0).permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

Image.fromarray(d[0]).convert("RGBA").save("diff.png")
