import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Origami for HTR',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--out-dir', type=str,
                        default='./output', help='output directory')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint_{cer}_{wer}_{iter}.pth to resume from')
    parser.add_argument('--train-bs', default=8,
                        type=int, help='train batch size')
    parser.add_argument('--val-bs', default=1, type=int,
                        help='validation batch size')
    parser.add_argument('--num-workers', default=0,
                        type=int, help='nb of workers')
    parser.add_argument('--eval-iter', default=1000, type=int,
                        help='nb of iterations to run evaluation')
    parser.add_argument('--total-iter', default=100000,
                        type=int, help='nb of total iterations for training')
    parser.add_argument('--warm-up-iter', default=1000,
                        type=int, help='nb of iterations for warm-up')
    parser.add_argument('--print-iter', default=100, type=int,
                        help='nb of total iterations to print information')
    parser.add_argument('--max-lr', default=1e-3,
                        type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=5e-1,
                        type=float, help='weight decay')
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='wheteher use wandb, otherwise use tensorboard')
    parser.add_argument('--wandb-project', type=str,
                        default='None', help='WandB project name')
    parser.add_argument('--exp-name', type=str, default='IAM_HTR_ORIGAMI_NET',
                        help='experimental name (save dir will be out_dir + exp_name)')
    parser.add_argument('--seed', default=123, type=int,
                        help='seed for initializing training. ')

    parser.add_argument(
        '--img-size', default=[512, 64], type=int, nargs='+', help='image size')
    parser.add_argument('--attn-mask-ratio', default=0.,
                        type=float, help='attention drop_key mask ratio')
    parser.add_argument(
        '--patch-size', default=[4, 32], type=int, nargs='+', help='patch size')
    parser.add_argument('--mask-ratio', default=0.3,
                        type=float, help='mask ratio')
    parser.add_argument('--cos-temp', default=8, type=int,
                        help='cosine similarity classifier temperature')
    parser.add_argument('--max-span-length', default=4,
                        type=int, help='max mask length')
    parser.add_argument('--spacing', default=0, type=int,
                        help='the spacing between two span masks')
    parser.add_argument('--proj', default=8, type=float,
                        help='projection value')

    parser.add_argument('--dpi-min-factor', default=0.5, type=float)
    parser.add_argument('--dpi-max-factor', default=1.5, type=float)
    parser.add_argument('--perspective-low', default=0., type=float)
    parser.add_argument('--perspective-high', default=0.4, type=float)
    parser.add_argument(
        '--elastic-distortion-min-kernel-size', default=3, type=int)
    parser.add_argument(
        '--elastic-distortion-max-kernel-size', default=3, type=int)
    parser.add_argument(
        '--elastic_distortion-max-magnitude', default=20, type=int)
    parser.add_argument('--elastic-distortion-min-alpha',
                        default=0.5, type=float)
    parser.add_argument('--elastic-distortion-max-alpha',
                        default=1, type=float)
    parser.add_argument('--elastic-distortion-min-sigma', default=1, type=int)
    parser.add_argument('--elastic-distortion-max-sigma', default=10, type=int)
    parser.add_argument('--dila-ero-max-kernel', default=3, type=int)
    parser.add_argument('--jitter-contrast', default=0.4, type=float)
    parser.add_argument('--jitter-brightness', default=0.4, type=float)
    parser.add_argument('--jitter-saturation', default=0.4, type=float)
    parser.add_argument('--jitter-hue', default=0.2, type=float)

    parser.add_argument('--dila-ero-iter', default=1, type=int,
                        help='nb of iterations for dilation and erosion kernel')
    parser.add_argument('--blur-min-kernel', default=3, type=int)
    parser.add_argument('--blur-max-kernel', default=5, type=int)
    parser.add_argument('--blur-min-sigma', default=3, type=int)
    parser.add_argument('--blur-max-sigma', default=5, type=int)
    parser.add_argument('--sharpen-min-alpha', default=0, type=int)
    parser.add_argument('--sharpen-max-alpha', default=1, type=int)
    parser.add_argument('--sharpen-min-strength', default=0, type=int)
    parser.add_argument('--sharpen-max-strength', default=1, type=int)
    parser.add_argument('--zoom-min-h', default=0.8, type=float)
    parser.add_argument('--zoom-max-h', default=1, type=float)
    parser.add_argument('--zoom-min-w', default=0.99, type=float)
    parser.add_argument('--zoom-max-w', default=1, type=float)
    parser.add_argument('--proba', default=0.5, type=float)

    parser.add_argument('--ema-decay', default=0.9999, type=float,
                        help='Exponential Moving Average (EMA) decay')
    parser.add_argument('--alpha', default=0, type=float,
                        help='kld loss ratio')

    # Encoder-Decoder specific arguments
    parser.add_argument('--model-type', default='ctc', type=str, choices=['ctc', 'encoder_decoder'],
                        help='Model type: ctc (original) or encoder_decoder (new)')
    parser.add_argument('--decoder-layers', default=6, type=int,
                        help='Number of transformer decoder layers')
    parser.add_argument('--decoder-heads', default=8, type=int,
                        help='Number of attention heads in decoder')
    parser.add_argument('--max-seq-len', default=256, type=int,
                        help='Maximum sequence length for decoder')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='Label smoothing factor for cross-entropy loss')
    parser.add_argument('--beam-size', default=5, type=int,
                        help='Beam size for beam search decoding')
    parser.add_argument('--generation-method', default='nucleus', type=str,
                        choices=['greedy', 'nucleus', 'beam_search'],
                        help='Generation method for inference')
    parser.add_argument('--generation-temperature', default=0.7, type=float,
                        help='Temperature for sampling-based generation')
    parser.add_argument('--repetition-penalty', default=1.3, type=float,
                        help='Penalty for repeated tokens during generation')
    parser.add_argument('--top-p', default=0.9, type=float,
                        help='Top-p (nucleus) sampling threshold')

    # Model loading/saving arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to pre-trained model to load (for fine-tuning)')
    parser.add_argument('--load-encoder-only', action='store_true', default=False,
                        help='Only load encoder weights (useful for transfer learning)')
    parser.add_argument('--strict-loading', action='store_true', default=True,
                        help='Use strict loading for model weights')

    parser.add_argument('--train-data-list', type=str, default='./data/iam/train.ln',
                        help='train data list (gc file)(ln file)')
    parser.add_argument('--data-path', type=str, default='./data/iam/lines/',
                        help='train data list')
    parser.add_argument('--val-data-list', type=str, default='./data/iam/val.ln',
                        help='val data list')
    parser.add_argument('--test-data-list', type=str, default='./data/iam/test.ln',
                        help='test data list')
    parser.add_argument('--nb-cls', default=80, type=int,
                        help='nb of classes, IAM=79+1, READ2016=89+1')

    parser.add_argument('--sgm-enable', action='store_true',
                        default=False, help='whether to use SGM')
    parser.add_argument('--sgm-lambda', default=1.0, type=float,
                        help='SGM loss weight (λ2 in the paper)')
    parser.add_argument('--ctc-lambda', default=0.1, type=float,
                        help='CTC loss weight (λ1 in the paper)')
    parser.add_argument('--sgm-sub-len', default=5, type=int,
                        help='SGM context sub-string length')
    parser.add_argument('--sgm-warmup-iters', default=0,
                        type=int, help='SGM warmup iters, 0 = start immediately')
    return parser.parse_args()
