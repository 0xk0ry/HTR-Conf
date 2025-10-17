import argparse
import os
import sys
from collections import defaultdict
import torch

# Ensure we can import the local model package
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'model')
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from model.HTR_VT import MaskedAutoencoderViT, create_model  # noqa: E402


def fmt(n: int) -> str:
    return f"{n:,}"  # thousands separator


def count_params(params):
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    return total, trainable


def line(name, total, trainable, indent=0, width=40):
    pad = ' ' * indent
    print(f"{pad}- {name:<{width - indent}} total={fmt(total):>10} trainable={fmt(trainable):>10}")


def theoretical_block_params(embed_dim: int, mlp_ratio: float) -> int:
    """Return the approximate parameter count per Transformer block (ignoring LayerNorm/bias)."""
    # (qkv + proj + 2-layer MLP) ≈ (3D^2 + D^2 + 2 * r * D^2) = (4 + 2r) D^2
    return int((4 + 2 * mlp_ratio) * (embed_dim ** 2))


def build_model(args):
    # If the user overrides architecture settings we construct directly; else use factory.
    if any(v is not None for v in [args.embed_dim_override, args.depth_override, args.num_heads_override, args.mlp_ratio_override]):
        embed_dim = args.embed_dim_override or 768
        depth = args.depth_override or 4
        num_heads = args.num_heads_override or 6
        mlp_ratio = args.mlp_ratio_override or 4.0
        # Mirror create_model defaults (patch size & norm_layer) for consistency
        model = MaskedAutoencoderViT(
            nb_cls=args.nb_cls,
            img_size=[args.img_w, args.img_h],
            patch_size=[4, 64],
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=torch.nn.LayerNorm,
        )
    else:
        model = create_model(nb_cls=args.nb_cls, img_size=[args.img_w, args.img_h])
    return model


def main():
    parser = argparse.ArgumentParser(description="Parameter breakdown for model_v1 MaskedAutoencoderViT")
    parser.add_argument('--nb-cls', type=int, default=80)
    parser.add_argument('--img-w', type=int, default=512)
    parser.add_argument('--img-h', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dummy-forward', action='store_true', help='Run a dummy forward pass to verify shape (not required for param count).')
    # Optional overrides (will construct model directly instead of using create_model)
    parser.add_argument('--embed-dim-override', type=int, default=None)
    parser.add_argument('--depth-override', type=int, default=None)
    parser.add_argument('--num-heads-override', type=int, default=None)
    parser.add_argument('--mlp-ratio-override', type=float, default=None)
    args = parser.parse_args()

    model = build_model(args).to(args.device)

    if args.dummy_forward:
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.img_h, args.img_w, device=args.device)
            out = model(dummy, use_masking=False)
            print(f"[DUMMY] Output shape: {tuple(out.shape)}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] Total={fmt(total)} Trainable={fmt(trainable)}")

    # High-level groups
    groups = []
    groups.append(("patch_embed (ResNet18)", model.patch_embed.parameters()))
    groups.append(("transformer.blocks", model.blocks.parameters()))
    groups.append(("transformer.norm", model.norm.parameters()))
    groups.append(("head", model.head.parameters()))
    groups.append(("layer_norm (post-head)", model.layer_norm.parameters()))
    groups.append(("mask_token", [model.mask_token]))
    groups.append(("pos_embed (frozen)", [model.pos_embed]))

    print("\n[PARAM] Component summary:")
    for name, params in groups:
        t, tr = count_params(list(params))
        line(name, t, tr)

    # Per block breakdown
    print("\n[PARAM] Transformer blocks detail:")
    for i, blk in enumerate(model.blocks):
        bt, btr = count_params(list(blk.parameters()))
        line(f"blocks[{i}]", bt, btr)
        # Attention internals
        qkv = [blk.attn.qkv.weight]
        if blk.attn.qkv.bias is not None:
            qkv.append(blk.attn.qkv.bias)
        proj = [blk.attn.proj.weight]
        if blk.attn.proj.bias is not None:
            proj.append(blk.attn.proj.bias)
        fc1 = [blk.mlp.fc1.weight, blk.mlp.fc1.bias]
        fc2 = [blk.mlp.fc2.weight, blk.mlp.fc2.bias]
        qkv_t, qkv_tr = count_params(qkv)
        proj_t, proj_tr = count_params(proj)
        fc1_t, fc1_tr = count_params(fc1)
        fc2_t, fc2_tr = count_params(fc2)
        line("  attn.qkv", qkv_t, qkv_tr, indent=2)
        line("  attn.proj", proj_t, proj_tr, indent=2)
        line("  mlp.fc1", fc1_t, fc1_tr, indent=2)
        line("  mlp.fc2", fc2_t, fc2_tr, indent=2)

    # Aggregate by top prefix
    agg = defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        agg[top][0] += p.numel()
        if p.requires_grad:
            agg[top][1] += p.numel()

    print("\n[PARAM] Top-level aggregation:")
    for k, (t, tr) in sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True):
        line(k, t, tr)

    # Theoretical block estimate (using current model attributes)
    embed_dim = model.embed_dim
    mlp_ratio = model.blocks[0].mlp.fc1.weight.shape[0] / embed_dim if model.blocks else 4.0
    theo_per_block = theoretical_block_params(embed_dim, mlp_ratio)
    print(f"\n[INFO] Approx theoretical params per block (ignoring norms/bias): {fmt(theo_per_block)}")
    print("Done.")


if __name__ == '__main__':
    main()
