from .mae import MAE, VisionTransformer


def create_model(name, mask_ratio):
    if name == 'pretrain_mae_small_patch16_224':
        model = MAE(
            mask_ratio=mask_ratio,
            img_size=224,
            patch_size=16,
            encoder_embed_dims=384,
            encoder_num_layers=12,
            encoder_num_heads=6,
            decoder_num_classes=768,
            decoder_embed_dims=192,
            decoder_num_layers=4,
            decoder_num_heads=3,
        )
    elif name == 'pretrain_mae_base_patch16_224':
        model = MAE(
            mask_ratio=mask_ratio,
            img_size=224,
            patch_size=16,
            encoder_embed_dims=768,
            encoder_num_layers=12,
            encoder_num_heads=12,
            decoder_num_classes=768,
            decoder_embed_dims=384,
            decoder_num_layers=4,
            decoder_num_heads=6,
        )
    elif name == 'pretrain_mae_large_patch16_224':
        model = MAE(
            mask_ratio=mask_ratio,
            img_size=224,
            patch_size=16,
            encoder_embed_dims=1024,
            encoder_num_layers=24,
            encoder_num_heads=16,
            decoder_num_classes=768,
            decoder_embed_dims=512,
            decoder_num_layers=8,
            decoder_num_heads=8,
        )
    elif name == 'vit_small_patch16_224':
        model = VisionTransformer(
            mask_ratio=0,
            img_size=224,
            patch_size=16,
            embed_dims=384,
            num_layers=12,
            num_heads=6,
        )
    elif name == 'vit_base_patch16_224':
        model = VisionTransformer(
            mask_ratio=0,
            img_size=224,
            patch_size=16,
            embed_dims=768,
            num_layers=12,
            num_heads=12,
        )
    elif name == 'vit_base_patch16_384':
        model = VisionTransformer(
            mask_ratio=0,
            img_size=224,
            patch_size=16,
            embed_dims=1024,
            num_layers=24,
            num_heads=16,
        )
    else:
        raise NotImplementedError('Wrong Model Type.')
    
    return model
