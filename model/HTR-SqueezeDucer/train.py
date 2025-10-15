import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
try:
    # PyTorch/Torchaudio >= 2.1
    from torchaudio.functional import rnnt_loss
    HAS_TORCHAUDIO_RNNT = True
except Exception:
    HAS_TORCHAUDIO_RNNT = False

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial
import random
import numpy as np
import re
import importlib
from model.sgm_head import SGMHead, build_sgm_vocab, make_context_batch
from torch.cuda.amp import autocast, GradScaler


# ---------------- RNNT helpers -----------------
def build_rnnt_special_tokens(args):
    """
    Define special token ids for RNNT: BOS and BLANK.
    Assumes model.blank_id == args.nb_cls - 1.
    """
    BOS_ID = 0
    BLANK_ID = args.nb_cls - 1
    return BOS_ID, BLANK_ID


def encode_batch_rnnt(texts, converter, bos_id, blank_id, device):
    """
    Pack a batch of variable-length targets to fixed [B, U] for prediction net and
    return flat targets for loss.
    Returns:
      y_in: [B, U+1] int64 with BOS in the first column, remaining padded with BLANK
      y_flat: [sumU] int64 concatenated targets (no BOS, no BLANK insertion)
      tgt_lengths: [B] int64 lengths per item (no BOS)
    """
    # converter.encode returns (flat_targets, lengths) with 1-based ids for real symbols, 0 reserved for CTC blank
    # IntTensor (labels in 1..V-1), 0 reserved for BOS here
    tgt_flat, tgt_lengths = converter.encode(texts)
    # Create padded targets y (without BOS)
    B = len(texts)
    max_u = int(tgt_lengths.max().item()) if B > 0 else 0
    # Padded targets for RNNT loss [B, U] (values on padded positions are ignored)
    y = torch.zeros((B, max_u), dtype=torch.long, device=device)
    offset = 0
    for b, L in enumerate(tgt_lengths.tolist()):
        if L > 0:
            y[b, :L] = tgt_flat[offset:offset+L]
        offset += L
    # Prepend BOS
    y_in = torch.full((B, max_u + 1), fill_value=bos_id,
                      dtype=torch.long, device=device)
    if max_u > 0:
        y_in[:, 1:] = y
    return y_in, y.to(device=device, dtype=torch.long), tgt_lengths.to(device=device, dtype=torch.long)


def compute_rnnt_loss(args, model, image, texts, converter):
    if not HAS_TORCHAUDIO_RNNT:
        raise RuntimeError(
            "torchaudio rnnt_loss not available. Install torchaudio>=2.1 or add warprnnt.")

    B = image.size(0)
    device = image.device
    BOS_ID, BLANK_ID = build_rnnt_special_tokens(args)
    # Prepare RNNT inputs
    y_in, y_pad, y_len = encode_batch_rnnt(
        texts, converter, BOS_ID, BLANK_ID, device)
    # Optionally cap target length to reduce U dimension
    U_cap = int(getattr(args, 'rnnt_max_target_len', 0) or 0)
    if U_cap > 0:
        max_u = min(U_cap, y_pad.size(1))
        y_len = torch.clamp(y_len, max=max_u)
        y_pad = y_pad[:, :max_u]
        y_in = y_in[:, :max_u+1]
    # Forward RNNT
    logits, T = model.forward_rnnt(image, y_in)  # [B, T, U, V]
    # Lengths as int32 tensors
    logit_lengths = torch.full((B,), T, dtype=torch.int32, device=device)
    target_lengths = y_len.to(dtype=torch.int32)

    loss_rnnt = rnnt_loss(
        logits.float(), y_pad.to(dtype=torch.int32), logit_lengths, target_lengths,
        blank=model.blank_id, reduction="mean", clamp=0.0
    )
    return loss_rnnt


def compute_losses(
    args,
    model,
    sgm_head,
    image,
    texts,
    batch_size,
    criterion_ctc,
    converter,
    nb_iter,
    ctc_lambda,
    sgm_lambda,
    stoi,
    mask_mode='span_old',
    mask_ratio=0.30,
    max_span_length=8,
):
    # 1) Forward
    if sgm_head is None or nb_iter < getattr(args, 'sgm_warmup_iters', 0):
        preds = model(image, use_masking=True, mask_mode=mask_mode,
                      # [B, N, V_ctc]
                      mask_ratio=mask_ratio, max_span_length=max_span_length)
        feats = None
    else:
        # Updated call: removed outdated positional arguments (args.mask_ratio, args.max_span_length)
        # to avoid passing multiple values for 'use_masking' (TypeError). Use keyword args instead.
        preds, feats = model(
            image,
            use_masking=True,
            return_features=True,
            mask_mode=mask_mode,
            mask_ratio=mask_ratio if mask_ratio is not None else getattr(
                args, 'mask_ratio', 0.0),
            max_span_length=max_span_length if max_span_length is not None else getattr(
                args, 'max_span_length', 0)
        )   # [B, N, V_ctc], [B, N, D]
        # Avoid double backprop through encoder via SGM: detach features
        feats = feats.detach()

    # 2) CTC loss
    text_ctc, length_ctc = converter.encode(
        texts)    # existing path (targets for CTC)
    preds_sz = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    loss_ctc = criterion_ctc(preds.permute(1, 0, 2).log_softmax(2).float(),
                             text_ctc.cuda(), preds_sz, length_ctc.cuda()).mean()

    # 3) SGM loss (optional)
    loss_sgm = torch.zeros((), device=preds.device)
    if sgm_head is not None and feats is not None:
        left_ctx, right_ctx, tgt_ids, tgt_mask = make_context_batch(
            texts, stoi, sub_str_len=getattr(args, 'sgm_sub_len', 5), device=preds.device)
        out = sgm_head(feats, left_ctx, right_ctx, tgt_ids,
                       tgt_mask)   # feats: [B,N,D] (detached)
        loss_sgm = out['loss_sgm']

    # 4) Optional RNNT loss (multi-task). Only computed when enabled at args and torchaudio available.
    loss_rnnt = torch.zeros((), device=image.device)
    rnnt_lambda = getattr(args, 'rnnt_lambda', 0.0)
    rnnt_enable = rnnt_lambda > 0.0 and HAS_TORCHAUDIO_RNNT
    if rnnt_enable and getattr(args, 'rnnt_after_warmup', 0) <= nb_iter:
        try:
            loss_rnnt = compute_rnnt_loss(args, model, image, texts, converter)
        except Exception as e:
            # Fallback to zero RNNT loss if something goes wrong; do not crash training
            print(f"[WARN] RNNT loss failed this step: {e}")
            loss_rnnt = torch.zeros((), device=image.device)

    # 5) Combine with weights
    total = ctc_lambda * loss_ctc + sgm_lambda * loss_sgm + rnnt_lambda * loss_rnnt
    return total, loss_ctc.detach(), loss_sgm.detach(), loss_rnnt.detach()


def tri_masked_loss(args, model, sgm_head, image, labels, batch_size,
                    criterion, converter, nb_iter, ctc_lambda, sgm_lambda, stoi,
                    r_rand=0.30, r_block=0.20, r_span=0.20, max_span=8):
    total = 0.0
    total_ctc = 0.0
    total_sgm = 0.0
    total_rnnt = 0.0
    weights = {"random": 1.0, "block": 1.0, "span_old": 1.0}
    plans = [("random", r_rand), ("block", r_block), ("span_old", r_span)]

    for mode, ratio in plans:
        out = compute_losses(
            args, model, sgm_head, image, labels, batch_size, criterion, converter,
            nb_iter, ctc_lambda, sgm_lambda, stoi,
            mask_mode=mode, mask_ratio=ratio, max_span_length=max_span
        )
        if len(out) == 4:
            loss, loss_ctc, loss_sgm, loss_rnnt = out
        else:
            loss, loss_ctc, loss_sgm = out
            loss_rnnt = torch.zeros((), device=image.device)
        w = weights[mode]
        total += w * loss
        total_ctc += w * loss_ctc
        total_sgm += w * loss_sgm
        total_rnnt += w * loss_rnnt

    denom = sum(weights.values())
    return total/denom, total_ctc/denom, total_sgm/denom, total_rnnt/denom


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    # Initialize wandb only if enabled
    if getattr(args, 'use_wandb', False):
        try:
            wandb = importlib.import_module('wandb')
            wandb.init(project=getattr(args, 'wandb_project', 'None'), name=args.exp_name,
                       config=vars(args), dir=args.save_dir)
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(
                f"Failed to initialize wandb: {e}. Continuing without wandb.")
            wandb = None
    else:
        wandb = None

    model = HTR_VT.create_model(
        nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    # Ensure EMA decay is properly accessed (handle both ema_decay and ema-decay)
    ema_decay = getattr(args, 'ema_decay', 0.9999)
    logger.info(f"Using EMA decay: {ema_decay}")
    model_ema = utils.ModelEma(model, ema_decay)
    model.zero_grad()

    # Use centralized checkpoint loader like model_v4-2
    resume_path = args.resume if getattr(
        args, 'resume', None) else getattr(args, 'resume_checkpoint', None)
    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = utils.load_checkpoint(
        model, model_ema, None, resume_path, logger)

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size, lang=getattr(args, 'lang', 'eng'))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph, lang=getattr(args, 'lang', 'eng'))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    sgm_enable = getattr(args, 'sgm_enable', True)
    sgm_lambda = getattr(args, 'sgm_lambda', 1.0)       # λ2 in the paper
    ctc_lambda = getattr(args, 'ctc_lambda', 0.1)       # λ1 in the paper
    sgm_sub_len = getattr(args, 'sgm_sub_len', 5)
    sgm_warmup = getattr(args, 'sgm_warmup_iters', 0)   # 0 = start immediately
    stoi, itos, pad_id, eos_id, bos_l_id, bos_r_id = build_sgm_vocab(converter)
    vocab_size_sgm = len(itos)
    d_vis = model.embed_dim

    sgm_head = SGMHead(d_vis=d_vis, vocab_size_sgm=vocab_size_sgm,
                       sub_str_len=sgm_sub_len).cuda()
    if sgm_head is not None:
        sgm_head.train()
    # Respect flag to disable SGM entirely
    if not sgm_enable:
        sgm_head = None

    # Build optimizer over model + SGM head (if enabled) so SGM params actually update
    param_groups = list(model.parameters())
    if sgm_enable and sgm_head is not None:
        param_groups += list(sgm_head.parameters())
        logger.info(
            f"Optimizing {sum(p.numel() for p in sgm_head.parameters())} SGM params in addition to model params")
    optimizer = sam.SAM(param_groups, torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    # Load optimizer & SGM head state after initialization
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            logger.info("Successfully loaded optimizer state")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            logger.info(
                "Continuing training without optimizer state (will restart from initial lr/momentum)")
    elif resume_path and os.path.isfile(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location='cpu',
                              weights_only=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                logger.info("Loaded optimizer state from checkpoint directly")
        except Exception as e:
            logger.warning(
                f"Could not load optimizer state from checkpoint: {e}")

    # If resuming and SGM head exists in checkpoint, restore it so SGM loss doesn't reset
    if resume_path and os.path.isfile(resume_path) and sgm_head is not None:
        try:
            ckpt = torch.load(resume_path, map_location='cpu',
                              weights_only=False)
            if 'sgm_head' in ckpt:
                sgm_head.load_state_dict(ckpt['sgm_head'], strict=False)
                logger.info("Restored SGM head state from checkpoint")
            else:
                logger.info(
                    "No SGM head state found in checkpoint; training SGM from scratch")
        except Exception as e:
            logger.warning(f"Failed to restore SGM head from checkpoint: {e}")

    best_cer, best_wer = best_cer, best_wer
    train_loss = train_loss
    train_loss_count = train_loss_count
    #### ---- train & eval ---- ####
    logger.info('Start training...')
    scaler = GradScaler(enabled=getattr(args, 'amp', False))
    for nb_iter in range(start_iter, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        optimizer.zero_grad()
        batch = next(train_iter)
        image = batch[0].cuda(non_blocking=True)
        text, length = converter.encode(batch[1])
        batch_size = image.size(0)

        with autocast(enabled=getattr(args, 'amp', False)):
            loss, loss_ctc, loss_sgm, loss_rnnt = tri_masked_loss(
                args, model, sgm_head, image, batch[1], batch_size, criterion, converter,
                nb_iter, ctc_lambda, sgm_lambda, stoi,
                r_rand=0.60, r_block=0.40, r_span=0.40, max_span=8
            )
        scaler.scale(loss).backward()
        optimizer.first_step(zero_grad=True)

       # ---- SECOND SAM PASS: recompute tri-CTC loss at the perturbed weights ----
        with autocast(enabled=getattr(args, 'amp', False)):
            loss2, _, _, _ = tri_masked_loss(
                args, model, sgm_head, image, batch[1], batch_size, criterion, converter,
                nb_iter, ctc_lambda, sgm_lambda, stoi,
                r_rand=0.60, r_block=0.40, r_span=0.40, max_span=8
            )
        scaler.scale(loss2).backward()
        optimizer.second_step(zero_grad=True)
        scaler.update()

        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)
        train_loss += loss.item()
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            log_msg = (
                f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t total : {train_loss_avg:0.5f} '
                f'\t CTC : {loss_ctc.mean():0.5f} \t SGM : {loss_sgm.mean():0.5f}'
            )
            if getattr(args, 'rnnt_lambda', 0.0) > 0.0:
                log_msg += f' \t RNNT : {loss_rnnt.mean():0.5f}'
            logger.info(log_msg)

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            # Per-loss scalars for TensorBoard
            writer.add_scalar('./Train/CTC', loss_ctc.mean(), nb_iter)
            writer.add_scalar('./Train/SGM', loss_sgm.mean(), nb_iter)
            if getattr(args, 'rnnt_lambda', 0.0) > 0.0:
                writer.add_scalar('./Train/RNNT', loss_rnnt.mean(), nb_iter)
            if wandb is not None:
                wandb.log({
                    'train/lr': current_lr,
                    'train/loss': train_loss_avg,
                    'train/CTC': loss_ctc.mean(),
                    'train/SGM': loss_sgm.mean(),
                    **({'train/RNNT': loss_rnnt.mean()} if getattr(args, 'rnnt_lambda', 0.0) > 0 else {}),
                    'iter': nb_iter,
                }, step=nb_iter)
            train_loss = 0.0
            train_loss_count = 0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)
                # Save checkpoint every print interval (like model_v4-2)
                ckpt_name = f"checkpoint_{best_cer:.4f}_{best_wer:.4f}_{nb_iter}.pth"
                checkpoint = {
                    'model': model.state_dict(),
                    'state_dict_ema': model_ema.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'nb_iter': nb_iter,
                    'best_cer': best_cer,
                    'best_wer': best_wer,
                    'args': vars(args),
                    'random_state': random.getstate(),
                    'numpy_state': np.random.get_state(),
                    'torch_state': torch.get_rng_state(),
                    'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    'train_loss': train_loss,
                    'train_loss_count': train_loss_count,
                }
                if sgm_head is not None:
                    checkpoint['sgm_head'] = sgm_head.state_dict()
                torch.save(checkpoint, os.path.join(args.save_dir, ckpt_name))
                if val_cer < best_cer:
                    logger.info(
                        f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    if sgm_head is not None:
                        checkpoint['sgm_head'] = sgm_head.state_dict()
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(
                        f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    if sgm_head is not None:
                        checkpoint['sgm_head'] = sgm_head.state_dict()
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                if wandb is not None:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/CER': val_cer,
                        'val/WER': val_wer,
                        'val/best_CER': best_cer,
                        'val/best_WER': best_wer,
                        'iter': nb_iter,
                    }, step=nb_iter)
                model.train()


if __name__ == '__main__':
    main()
