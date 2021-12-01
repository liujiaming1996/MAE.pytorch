# An unofficial PyTorch implementation of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

This repository is based on [MAE-pytorch](https://github.com/pengzhiliang/MAE-pytorch.git), thanks very much!!!

I'm conducting extensive experiments mentioned in the paper. The performance still seems to be a little bit different from the original.

<!-- ## Difference

### `shuffle` and `unshuffle`

`shuffle` and `unshuffle` operations don't seem to be directly accessible in pytorch, so we use another method to realize this process:
+ For `shuffle`, we use the method of randomly generating mask-map (14x14) in BEiT, where `mask=0` illustrates keeping the token, `mask=1` denotes dropping the token (not participating caculation in encoder). Then all visible tokens (`mask=0`) are fed into encoder network.
+ For `unshuffle`, we get the postion embeddings (with adding the shared mask token) of all masked tokens according to the mask-map and then concate them with the visible tokens (from encoder), and feed them into the decoder network to recontrust. -->

<!-- ### sine-cosine positional embeddings

The positional embeddings mentioned in the paper are `sine-cosine` version. And we adopt the implemention of [here](https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31), but it seems like a 1-D embeddings not 2-D's. So we don't know what effect it will bring. -->


## TODO
- [x] implement the finetune process
- [ ] reuse the model in `modeling_pretrain.py`
- [x] caculate the normalized pixels target
- [x] add the `cls` token in the encoder
- [x] visualization of reconstruction image
- [ ] knn and linear prob
- [x]2D `sine-cosine` position embeddings
- [x]Fine-tuning semantic segmentation on Cityscapes & ADE20K
- [x]Fine-tuning instance segmentation on COCO

## Setup

```
pip install -r requirements.txt
```

## Run
1. Pretrain & Finetune
```
bash pretrain.sh
```

2. Visualization of reconstruction
```bash
# Set the path to save images
OUTPUT_DIR='output/'
# path to image for visualization
IMAGE_PATH='files/ILSVRC2012_val_00031649.JPEG'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'

# Now, it only supports pretrained models with normalized pixel targets
python run_mae_vis.py ${IMAGE_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
```

## Result

|   model  | pretrain | finetune | accuracy | log | weight |
|:--------:|:--------:|:--------:|:--------:| :--------:|:--------:|
| vit-base |800e (unnormed pixel)|   100e   |   83.2%  | - | - |
<!-- | vit-large| 400e    | 50e      | 84.5%    | - | - | -->

I'm really appreaciate for your star!