"""
grid_search.py
==============
Staged grid search over learning rate, epochs and batch size.

Stage 1: search learning rate     (fix epochs, batch size)
Stage 2: search epochs+batch size (fix best lr from stage 1)
"""

import json
import logging
import os
from datetime import datetime
from itertools import product

import pandas as pd
import torch

from src.metrics  import log_metrics
from src.model    import build_model, build_optimizer, build_scheduler, build_loss
from src.trainer  import train_epoch, evaluate
from src.utils    import (
    setup_logging, get_device, load_data,
    load_tokenizer, build_dataloaders, save_checkpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_single(
    config:        dict,
    learning_rate: float,
    epochs:        int,
    batch_size:    int,
    df_train,
    df_test,
    df_annotated,
    tokenizer,
    class_weights,
    device,
    run_dir:       str,
) -> dict:
    """
    Train one model with given hyperparameters.
    Saves checkpoints per epoch.
    Returns best epoch metrics dict.
    """
    os.makedirs(run_dir, exist_ok=True)

    train_loader, test_loader, annotated_loader = build_dataloaders(
        df_train, df_test, df_annotated,
        tokenizer, config, batch_size,
    )

    model    = build_model(config, device)
    loss_fn  = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = build_optimizer(model, config, learning_rate)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(config['BERT']['warmup_ratio'] * total_steps)
    scheduler    = build_scheduler(optimizer, total_steps, warmup_steps)

    logger.info("  Total steps: %d | Warmup: %d", total_steps, warmup_steps)

    history   = []
    best_f1   = 0.0
    best_row  = {}

    for epoch in range(1, epochs + 1):
        logger.info("  --- Epoch %d/%d ---", epoch, epochs)

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch,
        )
        test_loss,  test_metrics = evaluate(model, test_loader,      loss_fn, device, "test")
        ann_loss,   ann_metrics  = evaluate(model, annotated_loader,  loss_fn, device, "annotated")

        logger.info(
            "  Train: %.4f | Test: %.4f | Gap: %.4f",
            train_loss, test_loss, test_loss - train_loss,
        )
        log_metrics(test_metrics)
        log_metrics(ann_metrics)

        row = {
            'epoch':          epoch,
            'learning_rate':  learning_rate,
            'batch_size':     batch_size,
            'train_loss':     train_loss,
            'test_loss':      test_loss,
            'test_f1_hate':   test_metrics['f1_hate'],
            'test_f1_macro':  test_metrics['f1_macro'],
            'test_accuracy':  test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall':    test_metrics['recall'],
            'ann_loss':       ann_loss,
            'ann_f1_hate':    ann_metrics['f1_hate'],
            'ann_f1_macro':   ann_metrics['f1_macro'],
            'ann_accuracy':   ann_metrics['accuracy'],
        }
        history.append(row)

        # Save checkpoint every epoch
        ckpt_path = os.path.join(run_dir, f'epoch_{epoch}')
        save_checkpoint(model, tokenizer, ckpt_path)

        # Track best epoch
        if test_metrics['f1_hate'] > best_f1:
            best_f1       = test_metrics['f1_hate']
            best_row      = row.copy()
            best_row['model_path'] = ckpt_path

    # Save run history
    pd.DataFrame(history).to_csv(
        os.path.join(run_dir, 'history.csv'), index=False
    )

    # Free GPU memory before next run
    del model
    torch.cuda.empty_cache()

    return best_row


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid_search(config: dict) -> None:
    """
    Staged grid search:
      Stage 1 — learning rate search (fixed epochs + batch size)
      Stage 2 — epochs + batch size search (fixed best lr)
    """
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config['PATHS']['output_dir'], f'search_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    device    = get_device()
    df_train, df_test, df_annotated = load_data(config)
    tokenizer = load_tokenizer(config)
    loss_fn   = build_loss(df_train, config, device)

    # Extract class weights tensor for passing to run_single
    if config['BERT']['use_class_weights']:
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=df_train[config['PATHS']['label_col']].astype(int).values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float)
    else:
        class_weights = None

    search       = config['BERT']['search']
    all_results  = []

    # -------------------------------------------------------------------
    # Stage 1: Learning rate search
    # -------------------------------------------------------------------
    stage1_epochs = config['BERT']['stage1_epochs']
    stage1_bs     = config['BERT']['stage1_batch_size']

    logger.info("=" * 60)
    logger.info("STAGE 1: Learning Rate Search")
    logger.info("Fixed: epochs=%d, batch_size=%d", stage1_epochs, stage1_bs)
    logger.info("Search lr: %s", search['learning_rate'])
    logger.info("=" * 60)

    stage1_results = []
    for lr in search['learning_rate']:
        run_name = f"lr{lr}_ep{stage1_epochs}_bs{stage1_bs}"
        run_dir  = os.path.join(output_dir, 'stage1', run_name)
        logger.info("\n[Stage 1] lr=%.0e | epochs=%d | batch=%d",
                    lr, stage1_epochs, stage1_bs)

        result = run_single(
            config=config, learning_rate=lr,
            epochs=stage1_epochs, batch_size=stage1_bs,
            df_train=df_train, df_test=df_test,
            df_annotated=df_annotated, tokenizer=tokenizer,
            class_weights=class_weights, device=device, run_dir=run_dir,
        )
        result['stage']    = 1
        result['run_name'] = run_name
        stage1_results.append(result)
        all_results.append(result)
        logger.info("[Stage 1] lr=%.0e → F1=%.4f", lr, result['test_f1_hate'])

    best_lr = max(stage1_results, key=lambda x: x['test_f1_hate'])['learning_rate']
    logger.info("\nBest lr from Stage 1: %.0e", best_lr)

    # -------------------------------------------------------------------
    # Stage 2: Epochs + batch size search
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Epochs + Batch Size Search")
    logger.info("Fixed lr=%.0e | Search: epochs=%s batch=%s",
                best_lr, search['epochs'], search['batch_size'])
    logger.info("=" * 60)

    for epochs, batch_size in product(search['epochs'], search['batch_size']):
        if epochs == stage1_epochs and batch_size == stage1_bs:
            logger.info("[Stage 2] Skipping ep=%d bs=%d (done in stage 1)",
                        epochs, batch_size)
            continue

        run_name = f"lr{best_lr}_ep{epochs}_bs{batch_size}"
        run_dir  = os.path.join(output_dir, 'stage2', run_name)
        logger.info("\n[Stage 2] lr=%.0e | epochs=%d | batch=%d",
                    best_lr, epochs, batch_size)

        result = run_single(
            config=config, learning_rate=best_lr,
            epochs=epochs, batch_size=batch_size,
            df_train=df_train, df_test=df_test,
            df_annotated=df_annotated, tokenizer=tokenizer,
            class_weights=class_weights, device=device, run_dir=run_dir,
        )
        result['stage']    = 2
        result['run_name'] = run_name
        all_results.append(result)
        logger.info("[Stage 2] ep=%d bs=%d → F1=%.4f",
                    epochs, batch_size, result['test_f1_hate'])

    # -------------------------------------------------------------------
    # Save results + best model
    # -------------------------------------------------------------------
    df_results   = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'search_results.csv')
    df_results.to_csv(results_path, index=False)
    logger.info("Search results saved: %s", results_path)

    best_run = df_results.loc[df_results['test_f1_hate'].idxmax()]
    logger.info("=" * 60)
    logger.info("BEST RUN")
    logger.info("  lr=%s | epochs=%s | batch=%s",
                best_run['learning_rate'],
                best_run['epoch'],
                best_run['batch_size'])
    logger.info("  F1 hate  : %.4f", best_run['test_f1_hate'])
    logger.info("  F1 macro : %.4f", best_run['test_f1_macro'])
    logger.info("  Accuracy : %.4f", best_run['test_accuracy'])
    logger.info("=" * 60)

    # Copy best checkpoint to final location
    from transformers import BertForSequenceClassification
    best_model = BertForSequenceClassification.from_pretrained(
        best_run['model_path']
    )
    best_path = os.path.join(output_dir, 'best_model')
    save_checkpoint(best_model, tokenizer, best_path)
    logger.info("Best model saved: %s", best_path)

    print("\n=== Search Results ===")
    print(df_results[[
        'stage', 'learning_rate', 'epoch', 'batch_size',
        'train_loss', 'test_loss', 'test_f1_hate',
        'test_f1_macro', 'test_accuracy',
    ]].sort_values('test_f1_hate', ascending=False).to_string(index=False))
