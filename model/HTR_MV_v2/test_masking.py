import torch
from model import HTR_VT


def approx_equal(a, b, tol=1):
    return abs(int(a) - int(b)) <= tol


def run_masking_tests():
    torch.manual_seed(0)
    model = HTR_VT.create_model(nb_cls=80, img_size=[512, 64])

    # create a small batch image [B, C, H, W] (grayscale C=1)
    B = 2
    x = torch.randn(B, 1, 64, 512)

    # get sequence features from the backbone+transformer: [B, L, D]
    feats = model.forward_features(x)
    print("feats.shape:", feats.shape)

    Bf, L, D = feats.shape
    # Print a small slice of the feature tensor for inspection (keep output compact)
    print("feats[0, :5, :5]:\n", feats[0, :min(5, L), :min(5, D)])

    # Test mask_random_1d
    ratio = 0.2
    mask = model.mask_random_1d(feats, ratio)
    print("mask_random_1d shape:", mask.shape, "dtype:", mask.dtype)
    assert mask.shape == (
        Bf, L), f"Expected mask shape {(Bf, L)}, got {mask.shape}"
    num_masked = (~mask).sum(dim=1)
    expected = int(round(ratio * L))
    print("mask_random_1d masked counts per-sample:",
          num_masked.tolist(), "expected approx:", expected)
    # Print a compact view of mask (first sample, first 50 positions)
    print("mask_random_1d sample[0, :50]:\n", mask[0, :].int())
    # Print masked indices for first sample
    masked_idx = (~mask[0]).nonzero(as_tuple=True)[0].tolist()
    print(
        f"mask_random_1d masked indices (sample 0) count={len(masked_idx)}:\n", masked_idx[:100])

    # Test mask_span_1d
    ratio2 = 0.40
    max_span = 8
    mask2 = model.mask_span_1d(feats, ratio2, max_span)
    print("mask_span_1d shape:", mask2.shape, "dtype:", mask2.dtype)
    # mask_span_1d returns [B, L, 1]
    num_masked2 = (mask2 == 0).sum(dim=1).squeeze(-1)
    print("mask_span_1d sample[0, :50]:\n", mask2[0, :, 0].int())
    masked_idx2 = (mask2[0, :, 0] == 0).nonzero(as_tuple=True)[0].tolist()
    print(
        f"mask_span_1d masked indices (sample 0) count={len(masked_idx2)}:\n", masked_idx2[:200])

    print("All masking tests passed.")

    # Test mask_span_1d
    ratio2 = 0.60
    max_span = 4
    mask2 = model.mask_block_1d(feats, ratio2, max_span)
    print("mask_block_1d shape:", mask2.shape, "dtype:", mask2.dtype)
    # mask_block_1d returns [B, L, 1]
    num_masked2 = (mask2 == 0).sum(dim=1).squeeze(-1)
    print("mask_block_1d sample[0, :50]:\n", mask2[0, :, 0].int())
    masked_idx2 = (mask2[0, :, 0] == 0).nonzero(as_tuple=True)[0].tolist()
    print(
        f"mask_block_1d masked indices (sample 0) count={len(masked_idx2)}:\n", masked_idx2[:200])

    print("All masking tests passed.")


if __name__ == "__main__":
    run_masking_tests()
