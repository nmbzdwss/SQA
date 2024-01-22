import argparse
import datetime
import json
import random
import time
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import src.data.datasets as datasets
import src.util.misc as utils
from src.engine.arg_parser import get_args_parser
from src.data.datasets import build_dataset, get_coco_api_from_dataset
from src.engine.trainer import train_one_epoch
from src.engine import hoi_evaluator, hoi_accumulator
from src.models import build_model
import wandb
from src.engine.evaluator_coco import coco_evaluate

from src.util.logger import print_params, print_args
from collections import OrderedDict

import torch.nn
# use for Visualization
import cv2
import os
from PIL import Image
from src.data.datasets.hico import make_hico_transforms

def save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename):
    # save_ckpt: function for saving checkpoints
    output_dir = Path(args.output_dir)
    if args.output_dir:
        checkpoint_path = output_dir / f'{filename}.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

def main(args):
    utils.init_distributed_mode(args)

    if not args.train_detr is not None: # pretrained DETR
        print("Freeze weights for detector")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data Setup
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val' if not args.eval else 'test', args=args)
    assert dataset_train.num_action() == dataset_val.num_action(), "Number of actions should be the same between splits"
    args.num_classes = dataset_train.num_category()
    args.num_actions = dataset_train.num_action()
    args.action_names = dataset_train.get_actions()
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers
    if args.dataset_file == 'vcoco':
        # Save V-COCO dataset statistics
        args.valid_ids = np.array(dataset_train.get_object_label_idx()).nonzero()[0]
        args.invalid_ids = np.argwhere(np.array(dataset_train.get_object_label_idx()) == 0).squeeze(1)
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()
    elif args.dataset_file == 'hico-det':
        args.valid_obj_ids = dataset_train.get_valid_obj_ids()
        args.correct_mat = torch.tensor(dataset_val.correct_mat).to(device)
    print_args(args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = print_params(model)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if ("detr" not in n  and 'clip' not in n) and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("detr" in n and 'backbone' not in n and 'clip' not in n) and p.requires_grad],
            "lr": args.lr * 0.1,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if ("detr" in n and 'backbone' in n and 'clip' not in n) and p.requires_grad],
            "lr": args.lr * 0.01,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.reduce_lr_on_plateau_factor, patience=args.reduce_lr_on_plateau_patience, verbose=True)

    # Weight Setup
    if args.detr_weights is not None:
        print(f"Loading detr weights from args.detr_weights={args.detr_weights}")
        if args.detr_weights.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.detr_weights, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.detr_weights, map_location='cpu')

        if 'hico_ft_q16.pth' in args.detr_weights: # hack: for loading hico fine-tuned detr
            mapped_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if k.startswith('detr.'):
                    mapped_state_dict[k.replace('detr.', '')] = v
            model_without_ddp.detr.load_state_dict(mapped_state_dict)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint['model'])

    if args.resume:
        print(f"Loading model weights from args.resume={args.resume}")
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])


    if args.demo:
        model.eval()
        nums = 500
        with open("action.txt", 'r', encoding='utf-8') as file:
            lines = file.readlines()

        verb_labels = []
        for line in lines:
            verb_label = ''.join([char for char in line if char.isascii()])
            verb_labels.append(verb_label.split()[1])

        for fname in os.listdir(args.img_file):
            path = args.img_file + '/' + fname
            if os.path.isdir(path):
                continue
            if nums == 0:
                break
            nums = nums - 1
            src = Image.open(path).convert('RGB')

            transforms = make_hico_transforms('test')
            img, _ = transforms[0](src, None)
            img, _ = transforms[1](img, None)
            img = img.to(device)
            mask = torch.zeros(img.shape[1], img.shape[2]).unsqueeze(0).to(device)
            img = img.unsqueeze(0)

            output, dec_attn, feature = model(utils.NestedTensor(img, mask))
            feature = feature[0]
            hdim, h, w = feature.shape

            output_verb = output["pred_actions"][0]
            output_boxo = output["pred_boxes"][0]
            output_pairs = output["pred_rel_pairs"][0]

            dec_attn = dec_attn[-1].view(32, 256)

            output_verb = torch.softmax(output_verb, -1)
            _, ind = torch.topk(output_verb, 1, dim=-1)
            _, index = torch.topk(output_verb[:, ind[0]], 1, dim=0)
            logit = output_verb[:, ind[0]][index[0]][0][0].cpu().tolist()

            action = verb_labels[ind[index][0][0].cpu()]
            st = str(round(logit, 2)) + " " + action

            def c2x(list, src):
                h = src.height
                w = src.width
                cx = list[0] * w
                cy = list[1] * h
                cw = list[2] * w
                ch = list[3] * h
                return [(int(cx - 0.5 * cw), int(cy - 0.5 * ch)), (int(cx + 0.5 * cw), int(cy + 0.5 * ch))]

            boxh = output_boxo[output_pairs[index[0]][0][0]].cpu().tolist()
            boxh = c2x(boxh, src)
            boxo = output_boxo[output_pairs[index[0]][0][1]].cpu().tolist()
            boxo = c2x(boxo, src)
            src.save(f'imgs/img_src/{fname}')

            dec_attn = torch.sum(dec_attn, dim=-1).view(32, -1)
            feature = feature[-1].view(1, -1)
            d_attn = torch.mm(dec_attn, feature)
            d_attn = torch.softmax(d_attn, dim=-1)[index[0]].view(h, w).detach().cpu().numpy()
            d_attn = (d_attn - d_attn.min()) / (d_attn.max() - d_attn.min())
            cvimg = cv2.imread(args.img_file + '/' + fname)
            d_attn = cv2.resize(d_attn, (cvimg.shape[1], cvimg.shape[0]))
            d_attn = (255 - (255 * d_attn)).astype("uint8")
            heat = cv2.applyColorMap(d_attn, cv2.COLORMAP_JET)
            img_rgb = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            img_heat = cv2.addWeighted(heat, 0.6, img_rgb, 0.4, 0)

            cv2.imwrite(f'imgs/img_heat/{fname}', img_heat)
            cv2.putText(cvimg, st, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            cv2.rectangle(cvimg, boxh[0], boxh[1], (0, 255, 255), thickness=2)
            cv2.rectangle(cvimg, boxo[0], boxo[1], (255, 0, 0), thickness=2)
            cv2.imwrite(f'imgs/img_result/{fname}', cvimg)
        return

    if args.eval:
        # test only mode
        if args.HOIDet:
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                sc1, sc2 = hoi_accumulator(args, total_res, True, False)
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f}')
                print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f}')
                print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f}')
            else: raise ValueError(f'dataset {args.dataset_file} is not supported.')
            return
        else:
            # check original detr code
            base_ds = get_coco_api_from_dataset(data_loader_val)
            test_stats, coco_evaluator = coco_evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, args.output_dir / "eval.pth")
            return

    # stats
    scenario1, scenario2 = 0, 0
    best_mAP, best_rare, best_non_rare = 0, 0, 0

    # add argparse
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            name=args.run_name,
            config=args
        )
        wandb.watch(model)

    # Training starts here!
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.epochs,
            args.clip_max_norm, dataset_file=args.dataset_file, log=args.wandb)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.StepLR): lr_scheduler.step()

        # Validation
        if args.validate:
            print('-'*100)
            if args.dataset_file == 'vcoco':
                total_res = hoi_evaluator(args, model, criterion, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    sc1, sc2 = hoi_accumulator(args, total_res, False, args.wandb)
                    if sc1 > scenario1:
                        scenario1 = sc1
                        scenario2 = sc2
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| Scenario #1 mAP : {sc1:.2f} ({scenario1:.2f})')
                    print(f'| Scenario #2 mAP : {sc2:.2f} ({scenario2:.2f})')
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): lr_scheduler.step(sc1)
            elif args.dataset_file == 'hico-det':
                test_stats = hoi_evaluator(args, model, None, postprocessors, data_loader_val, device)
                if utils.get_rank() == 0:
                    if test_stats['mAP'] > best_mAP:
                        best_mAP = test_stats['mAP']
                        best_rare = test_stats['mAP rare']
                        best_non_rare = test_stats['mAP non-rare']
                        save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename='best')
                    print(f'| mAP (full)\t\t: {test_stats["mAP"]:.2f} ({best_mAP:.2f})')
                    print(f'| mAP (rare)\t\t: {test_stats["mAP rare"]:.2f} ({best_rare:.2f})')
                    print(f'| mAP (non-rare)\t: {test_stats["mAP non-rare"]:.2f} ({best_non_rare:.2f})')
                    if args.wandb and utils.get_rank() == 0:
                        wandb.log({
                            'mAP': test_stats['mAP']
                        })
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): lr_scheduler.step(test_stats['mAP'])
            print('-'*100)

        # if epoch%2==0:
        #     save_ckpt(args, model_without_ddp, optimizer, lr_scheduler, epoch, filename=f'checkpoint_{epoch}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.dataset_file == 'vcoco':
        print(f'| Scenario #1 mAP : {scenario1:.2f}')
        print(f'| Scenario #2 mAP : {scenario2:.2f}')
    elif args.dataset_file == 'hico-det':
        print(f'| mAP (full)\t\t: {best_mAP:.2f}')
        print(f'| mAP (rare)\t\t: {best_rare:.2f}')
        print(f'| mAP (non-rare)\t: {best_non_rare:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    # training
    parser.add_argument('--detr_weights', default=None, type=str)
    parser.add_argument('--train_detr', action='store_true', default=False)
    parser.add_argument('--finetune_detr_weight', default=0.1, type=float)
    parser.add_argument('--lr_detr', default=1e-5, type=float)
    parser.add_argument('--reduce_lr_on_plateau_patience', default=2, type=int)
    parser.add_argument('--reduce_lr_on_plateau_factor', default=0.1, type=float)

    # loss
    parser.add_argument('--proposal_focal_loss_alpha', default=0.75, type=float) # large alpha for high recall
    parser.add_argument('--action_focal_loss_alpha', default=0.5, type=float)
    parser.add_argument('--proposal_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--action_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--proposal_loss_coef', default=1, type=float)
    parser.add_argument('--action_loss_coef', default=1, type=float)

    # ablations
    parser.add_argument('--no_hard_mining_for_relation_discovery', dest='use_hard_mining_for_relation_discovery', action='store_false', default=True)
    parser.add_argument('--no_relation_dependency_encoding', dest='use_relation_dependency_encoding', action='store_false', default=True)
    parser.add_argument('--no_memory_layout_encoding', dest='use_memory_layout_encoding', action='store_false', default=True, help='layout encodings')
    parser.add_argument('--no_nms_on_detr', dest='apply_nms_on_detr', action='store_false', default=True)
    parser.add_argument('--no_tail_semantic_feature', dest='use_tail_semantic_feature', action='store_false', default=True)
    parser.add_argument('--no_spatial_feature', dest='use_spatial_feature', action='store_false', default=True)
    parser.add_argument('--no_interaction_decoder', action='store_true', default=False)

    # not sensitive or effective
    # parser.add_argument('--use_memory_union_mask', action='store_true', default=False)
    # parser.add_argument('--use_union_feature', action='store_true', default=False)
    parser.add_argument('--adaptive_relation_query_num', action='store_true', default=False)
    # parser.add_argument('--use_relation_tgt_mask', action='store_true', default=False)
    # parser.add_argument('--use_relation_tgt_mask_attend_topk', default=10, type=int)
    # parser.add_argument('--use_prior_verb_label_mask', action='store_true', default=False)
    parser.add_argument('--relation_feature_map_from', default='backbone', help='backbone | detr_encoder')
    # parser.add_argument('--use_query_fourier_encoding', action='store_true', default=False)

    # SQA ablations
    parser.add_argument('--use_ho_rel_location', action='store_true', default=True)
    parser.add_argument('--use_clip_fusion_q', action='store_true', default=False)
    parser.add_argument('--use_attn_mask', action='store_true', default=True)
    parser.add_argument('--mode', default=0, type=int) # Different mask settings number modes. Detail in Transformer.MaskTransformerDecoder

    # Visualization
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--img_file', default=None, type=str)

    args = parser.parse_args()
    args.STIP_relation_head = True

    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
