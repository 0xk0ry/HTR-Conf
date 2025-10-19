import argparse
import os
import sys
from typing import List

import torch


def ensure_path_on_sys_path(path: str):
    if path not in sys.path:
        sys.path.insert(0, path)


def get_model(nb_cls: int, img_size_wh: List[int]):
    """
    Build the HTR_Conformer_MFCF model.
    img_size_wh: [W, H] as expected by HTR_VT.create_model
    """
    # Point sys.path to the specific model folder to avoid ambiguity with siblings
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    mfcf_dir = os.path.join(repo_root, 'model', 'HTR_Conformer')
    mfcf_model_dir = os.path.join(mfcf_dir, 'model')
    ensure_path_on_sys_path(mfcf_dir)           # so 'model' (namespace) is discoverable
    ensure_path_on_sys_path(mfcf_model_dir)     # and direct module imports work
    try:
        # Prefer direct module import to avoid namespace issues
        from HTR_VT import create_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import HTR_Conformer_MFCF model: {e}")

    model = create_model(nb_cls=nb_cls, img_size=img_size_wh)
    model.eval()
    return model


def try_fvcore_flops(model: torch.nn.Module, inp: torch.Tensor):
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        with torch.no_grad():
            flops = FlopCountAnalysis(model, (inp,))
            total_flops = flops.total()
            table = flop_count_table(flops, max_depth=3)
        return total_flops, table
    except Exception as e:
        return None, f"fvcore failed: {e}"


def try_thop_flops(model: torch.nn.Module, inp: torch.Tensor):
    try:
        from thop import profile
        # Register trivial flops for unsupported modules (e.g., DropPath)
        try:
            from timm.models.layers import DropPath  # type: ignore
            from thop.vision.basic_hooks import count_zero_ops  # type: ignore
            from thop.profile import register_hooks  # type: ignore
            register_hooks.update({DropPath: count_zero_ops})
        except Exception:
            pass

        with torch.no_grad():
            macs, params = profile(model, inputs=(inp,), verbose=False)
        # thop returns MACs; FLOPs approx 2*MACs for conv/linear
        flops = 2 * macs
        return flops, params
    except Exception as e:
        return None, f"thop failed: {e}"


def human_readable(num: float, unit: str = 'FLOPs') -> str:
    for suffix, div in [('T', 1e12), ('G', 1e9), ('M', 1e6), ('K', 1e3)]:
        if num >= div:
            return f"{num/div:.3f} {suffix}{unit}"
    return f"{num:.3f} {unit}"


def main():
    parser = argparse.ArgumentParser(description='Profile FLOPs/params for HTR_Conformer_MFCF')
    parser.add_argument('--width', type=int, default=512, help='Input image width')
    parser.add_argument('--height', type=int, default=64, help='Input image height (>=64 recommended)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for profiling')
    parser.add_argument('--nb-cls', type=int, default=80, help='Number of CTC classes (alphabet size)')
    parser.add_argument('--sizes', type=str, default='', help='Optional comma-separated widths to sweep, e.g. 256,512,1024')
    args = parser.parse_args()

    sizes = [args.width]
    if args.sizes:
        try:
            sizes = [int(s.strip()) for s in args.sizes.split(',') if s.strip()]
        except Exception:
            pass

    # Build model once; pos_embed uses img_size, which we adapt per run as needed by rebuilding
    device = torch.device('cpu')

    print('HTR_Conformer_MFCF FLOPs/Params profiling')
    print('========================================')

    for w in sizes:
        h = args.height
        print(f"\nInput size: (B={args.batch_size}, C=1, H={h}, W={w})")
        model = get_model(nb_cls=args.nb_cls, img_size_wh=[w, h]).to(device)
        inp = torch.randn(args.batch_size, 1, h, w, device=device)

        # Parameter count
        params = sum(p.numel() for p in model.parameters())
        print(f"Params: {human_readable(params, unit=' params')}")

        # Try fvcore first for more accurate attention flops
        flops, detail = try_fvcore_flops(model, inp)
        if flops is not None:
            print(f"FLOPs (fvcore): {human_readable(flops)} per batch; {human_readable(flops/args.batch_size)} per image")
            print("\nTop-level breakdown (fvcore):")
            print(detail)
        else:
            print(detail)
            # Fallback to thop (will estimate using MACs*2)
            flops2, params2 = try_thop_flops(model, inp)
            if flops2 is not None:
                print(f"FLOPs (thop est.): {human_readable(flops2)} per batch; {human_readable(flops2/args.batch_size)} per image")
            else:
                print(params2)

        # small separator between sizes
        print('-' * 60)


if __name__ == '__main__':
    main()
