from functools import partial
from src.models_mae_torch import MaskedAutoencoderViT
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import einops
from PIL import Image
import numpy as np
import requests

if __name__ == "__main__":
    model = MaskedAutoencoderViT(img_size=224,
                                 patch_size=16, embed_dim=192, depth=12, num_heads=3,
                                 decoder_embed_dim=512, decoder_depth=1, decoder_num_heads=16,
                                 mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                 )
    print(model.state_dict().keys())

    print(model.load_state_dict(torch.load('checkpoints/torch/mae-deit-t16-224-in1k-800ep-d1-ema.pth')))

    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'  # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba
    # .jpg' # cucumber, from ILSVRC2012_val_00047851
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    img=einops.rearrange(img,'h w c->c h w')

    x = torch.from_numpy(img).unsqueeze(0).to(torch.float)

    loss, pred, mask = model(x, mask_ratio=0.2)
    print(loss)

    y = model.unpatchify(pred)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    # show_image(x[0], "original")

    plt.imshow(x[0])
    plt.subplot(1, 4, 2)
    plt.imshow(im_masked[0])

    plt.subplot(1, 4, 3)
    plt.imshow(y[0])

    plt.subplot(1, 4, 4)
    plt.imshow(im_paste[0])

    plt.show()



    # data = einops.rearrange(x, 'b c h w ->b h w c')
    """
    x_flat = einops.rearrange(x, 'b c (h d1) (w d2) -> b (h w) (d1 d2  c)', d1=16, d2=16)

    print(pred.shape, mask.shape,x_flat.shape)

    print(loss)

    im_masked = x_flat * (1 - mask[:, :, None])
    out = mask[:, :, None] * pred + (1 - mask[:, :, None]) * x_flat

    # while True:
    #     pass

    pred = einops.rearrange(pred, 'b (h w) (d1 d2 c )-> b (h d1) (w d2) c ', d1=2, d2=2, h=16)
    out = einops.rearrange(out, 'b (h w) (d1 d2 c )-> b (h d1) (w d2) c ', d1=2, d2=2, h=16)
    im_masked = einops.rearrange(im_masked, 'b (h w) (d1 d2 c )-> b (h d1) (w d2) c ', d1=2, d2=2, h=16)

    data = einops.rearrange(x, 'b c h w ->b h w c')

    # print(pred)
    plt.subplot(131)
    plt.imshow(im_masked[0].detach().numpy())

    plt.subplot(132)
    plt.imshow(data[0].detach().numpy())

    plt.subplot(133)
    plt.imshow(out[0].detach().numpy())

    plt.show()
    """