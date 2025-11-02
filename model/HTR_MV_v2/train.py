import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

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
import wandb


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
    mask_mode='span',
    mask_ratio=0.30,
    block_span=4,
    max_span_length=8,
    pre_sgm_ctx=None
):
    # 1) Forward
    if sgm_head is None or nb_iter < args.sgm_warmup_iters:
        preds = model(image, use_masking=True, mask_mode=mask_mode,
                      mask_ratio=mask_ratio, max_span_length=max_span_length)
        feats = None
    else:
        preds, feats, vis_mask = model(
            image,
            use_masking=True,
            return_features=True,
            return_mask=True,
            mask_mode=mask_mode,
            mask_ratio=mask_ratio,
            block_span=block_span,
            max_span_length=max_span_length
        )
    text_ctc, length_ctc = converter.encode(texts)
    text_ctc = text_ctc.to(preds.device)
    length_ctc = length_ctc.to(preds.device)
    preds_sz = torch.full((batch_size,), preds.size(
        1), dtype=torch.int32, device=preds.device)
    loss_ctc = criterion_ctc(preds.permute(1, 0, 2).log_softmax(2),
                             text_ctc, preds_sz, length_ctc).mean()

    loss_sgm = torch.zeros((), device=preds.device)
    if sgm_head is not None and feats is not None:
        left_ctx, right_ctx, tgt_ids, tgt_mask = pre_sgm_ctx if pre_sgm_ctx is not None else make_context_batch(
            texts, stoi, sub_str_len=args.sgm_sub_len, device=image.device)
        # Map visual mask (vis_mask: [B, N]) to target positions (tgt_mask: [B, L])
        if vis_mask is not None:
            # vis_mask may have shape [B, N]; tgt_mask is [B, L]
            B_v, N_v = vis_mask.shape
            B_t, L_t = tgt_mask.shape
            if N_v != L_t:
                # linearly sample N_v positions to produce L_t entries
                idx = torch.linspace(0, N_v - 1, steps=L_t, device=vis_mask.device).long()
                focus_mask = vis_mask[:, idx]
            else:
                focus_mask = vis_mask
        else:
            focus_mask = None

        out = sgm_head(
            feats,
            left_ctx, right_ctx,
            tgt_ids, tgt_mask,
            focus_mask=focus_mask  # [B, L] float mask: 1 where visual tokens were masked (mapped)
        )
        loss_sgm = out['loss_sgm']

    total = ctc_lambda * loss_ctc + sgm_lambda * loss_sgm
    return total, loss_ctc.detach(), loss_sgm.detach()


def tri_masked_loss(args, model, sgm_head, image, labels, batch_size,
                    criterion, converter, nb_iter, ctc_lambda, sgm_lambda, stoi,
                    r_rand=0.6, r_block=0.6, block_span=4, r_span=0.4, max_span=8):
    total = 0.0
    total_ctc = 0.0
    total_sgm = 0.0
    plans = [("random", r_rand), ("block", r_block), ("span", r_span)]

    if sgm_head is not None and nb_iter >= args.sgm_warmup_iters:
        pre_sgm_ctx = make_context_batch(
            labels, stoi, sub_str_len=args.sgm_sub_len, device=image.device)

    for mode, ratio in plans:
        loss, loss_ctc, loss_sgm = compute_losses(
            args, model, sgm_head, image, labels, batch_size, criterion, converter,
            nb_iter, ctc_lambda, sgm_lambda, stoi,
            mask_mode=mode, mask_ratio=ratio, block_span=block_span, max_span_length=max_span,
            pre_sgm_ctx=pre_sgm_ctx
        )
        total += loss
        total_ctc += loss_ctc
        total_sgm += loss_sgm

    denom = 3.0
    return total/denom, total_ctc/denom, total_sgm/denom


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

    torch.backends.cudnn.benchmark = True

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
    accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
    micro_step = 0
    # stats across macro iters
    avg_loss_ctc = 0.0
    avg_loss_sgm = 0.0

    # Pure FP32 training (AMP and grad clipping removed by request)

    for nb_iter in range(start_iter, args.total_iter):

        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)

        # Accumulate gradients over accum_steps micro-batches, then perform one SAM step
        optimizer.zero_grad()
        total_loss_this_macro = 0.0
        avg_loss_ctc = 0.0
        avg_loss_sgm = 0.0
        cached_batches = []
        for micro_step in range(accum_steps):
            batch = next(train_iter)
            # cache CPU tensors and labels for SAM second pass
            cached_batches.append(batch)
            image = batch[0].cuda(non_blocking=True)
            text, length = converter.encode(batch[1])
            batch_size = image.size(0)

            loss, loss_ctc, loss_sgm = tri_masked_loss(
                args, model, sgm_head, image, batch[1], batch_size, criterion, converter,
                nb_iter, ctc_lambda, sgm_lambda, stoi,
                r_rand=0.60, r_block=0.60, r_span=0.40, max_span=8
            )
            # average over accum steps
            (loss / accum_steps).backward()
            total_loss_this_macro += loss.item()
            avg_loss_ctc += loss_ctc.mean().item()
            avg_loss_sgm += loss_sgm.mean().item()

        # SAM first step after accumulating all gradients
        optimizer.first_step(zero_grad=True)

        # Recompute with perturbed weights and accumulate again for the second step
        for micro_step in range(accum_steps):
            batch = cached_batches[micro_step]
            image = batch[0].cuda(non_blocking=True)
            text, length = converter.encode(batch[1])
            batch_size = image.size(0)

            loss2, _, _ = compute_losses(
                args, model, sgm_head, image, batch[1], batch_size, criterion, converter,
                nb_iter, ctc_lambda, sgm_lambda, stoi,
                mask_mode='span', mask_ratio=0.6, block_span=4, max_span_length=8
            )
            (loss2 / accum_steps).backward()

        # SAM second step
        optimizer.second_step(zero_grad=True)

        model.zero_grad()
        # Update EMA once per macro-iteration (after one optimizer step)
        model_ema.update(model, num_updates=nb_iter / 2)

        # Aggregate stats for logging (use averages across accum_steps)
        train_loss += total_loss_this_macro / accum_steps
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            logger.info(
                f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t total : {train_loss_avg:0.5f} \t CTC : {(avg_loss_ctc/accum_steps):0.5f} \t SGM : {(avg_loss_sgm/accum_steps):0.5f} \t ')

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            if wandb is not None:
                wandb.log({
                    'train/lr': current_lr,
                    'train/loss': train_loss_avg,
                    'train/CTC': (avg_loss_ctc/accum_steps),
                    'train/SGM': (avg_loss_sgm/accum_steps),
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
