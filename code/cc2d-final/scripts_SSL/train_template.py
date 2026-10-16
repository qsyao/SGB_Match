import argparse
import datetime
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

from models.network import UNet_Pretrained
from datasets import select_dataset_SSL_Train
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train, merge_from_args
from utils.mylogger import get_mylogger, set_logger_dir
import numpy as np
import random

from torch.backends import cudnn
cudnn.benchmark = False      # if benchmark=True, deterministic will be False
cudnn.deterministic = True
# Keep each experiment light on CPU-side thread/process quotas so multiple
# GPU ablations can run concurrently on shared servers.
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
torch.set_num_interop_threads(int(os.environ.get('TORCH_NUM_INTEROP_THREADS', '1')))

def ce_loss(cos_map, gt_y, gt_x, nearby=None, add_bias=True, args=None, edge=None):
    """InfoNCE loss over a similarity map.

    Coordinate convention in this repository is always [y, x] after a
    landmark enters the dataloader. `gt_y/gt_x` are the raw-image coordinates
    projected to the current feature level, while `nearby` restricts negatives
    to a local window for the shallower, higher-resolution features.
    """
    b, h, w = cos_map.shape
    total_loss = list()
    for id in range(b):
        # cos_map[id] = cos_map[id].exp()
        # gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
        if nearby is not None:
            # Higher-resolution feature levels use a local negative window.
            min_y, max_y = max(gt_y[id] - nearby, 0), min(gt_y[id] + nearby, h)
            min_x, max_x = max(gt_x[id] - nearby, 0), min(gt_x[id] + nearby, w)
            chosen_patch = cos_map[id, min_y:max_y, min_x:max_x]
            gap_y, gap_x = gt_y[id] - min_y, gt_x[id] - min_x
            y, x = chosen_patch.shape
            mesh = torch.meshgrid(torch.linspace(0, y-1, y), torch.linspace(0, x-1, x), indexing='ij')
            distance_y, distance_x = mesh[0] - gap_y, mesh[1] - gap_x
            distance = torch.sqrt(distance_y**2 + distance_x**2).cuda()
            distance = torch.clamp(distance*args.strength * args.temprature, 0, args.clip * args.temprature)
            bias_terms = []
            if add_bias:
                bias_terms.append(distance)
            if edge is not None and args.edge_factor > 0:
                chosen_edge = edge[id, min_y:max_y, min_x:max_x].float() / 255
                bias_terms.append((1 - chosen_edge) * args.temprature * args.edge_factor)
            if bias_terms:
                bias = torch.stack(bias_terms).sum(dim=0)
                bias = torch.clamp(bias, 0, args.temprature * args.clip)
                chosen_patch = chosen_patch + bias
            chosen_patch = chosen_patch.exp()
            gt_value = chosen_patch[gap_y, gap_x].clone()
        else:
            # The deepest feature level compares the positive against all pixels.
            y, x = cos_map[id].shape
            # Distance bias downweights far-away negatives.
            mesh = torch.meshgrid(torch.linspace(0, y-1, y), torch.linspace(0, x-1, x), indexing='ij')
            distance_y, distance_x = mesh[0] - gt_y[id], mesh[1] - gt_x[id]
            distance = torch.sqrt(distance_y**2 + distance_x**2).cuda()
            distance = torch.clamp(distance*args.strength * args.temprature, 0, args.clip * args.temprature)
            if add_bias:
                cos_map[id] += distance
            cos_map[id] = cos_map[id].exp()
            gt_value = cos_map[id, gt_y[id], gt_x[id]].clone()
            chosen_patch = cos_map[id]
        # InfoNCE: chosen_patch contains all candidates, gt_value is the positive pair.
        id_loss = - torch.log(gt_value / chosen_patch.sum())
        total_loss.append(id_loss)
    return torch.stack(total_loss).mean()


def match_inner_product(feature, template):
    # feature: [B, C, H, W], template: [B, C].
    # Return one cosine-similarity map per batch item: [B, H, W].
    feature = feature.permute(0, 2, 3, 1)
    template = template.unsqueeze(1).unsqueeze(1)
    fea_L2 = torch.norm(feature, dim=-1)
    template_L2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_L2 * template_L2 + 1e-3)
    # cos_similarity = torch.clamp(cos_similarity, 0., 1 - 1e-6)
    return cos_similarity

def random_all(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True, warn_only=True)


def prune_numbered_checkpoints(runs_dir, logger=None):
    """Delete legacy per-epoch checkpoints from a run directory.

    Training now keeps only two checkpoints: `model_epoch_current.pth` for
    the latest evaluated epoch and `model_epoch_best.pth` for the lowest Test
    MRE seen so far. Old `model_epoch_<number>.pth` files are safe to remove
    once best/current have been written.
    """
    removed = 0
    for ckpt in Path(runs_dir).glob("model_epoch_*.pth"):
        epoch_token = ckpt.stem.replace("model_epoch_", "", 1)
        if not epoch_token.isdigit():
            continue
        ckpt.unlink()
        removed += 1
    if removed and logger is not None:
        logger.info("Removed %d numbered checkpoints from %s", removed, runs_dir)

if __name__ == "__main__":
    # Parse command line options
    parser = argparse.ArgumentParser(description="Train a CC2D coarse or fine SSL landmark model")
    parser.add_argument("--tag", default='layer2_0.001', help="final_runs tag for this experiment")
    parser.add_argument("--dataset", default='head', choices=['head', 'hand', 'leg', 'chest'], help="dataset config")
    parser.add_argument("--config_file", default="config.yaml", help="legacy argument; dataset SSL configs are loaded automatically")
    parser.add_argument("--epoch", type=int, default=0, help="legacy unused starting-epoch argument")
    parser.add_argument("--add_bias", type=int, default=1, help="add distance bias inside InfoNCE maps")
    parser.add_argument("--mask_fine_loss", type=int, default=0, help="legacy no-op flag kept for old scripts")
    parser.add_argument("--clip", type=float, default=0.8, help="distance-bias clipping value")
    parser.add_argument("--strength", type=float, default=0.07, help="distance-bias strength")
    parser.add_argument("--temprature", type=float, default=10, help="contrastive temperature; spelling kept for compatibility")
    parser.add_argument("--min_prob", type=float, default=0.0, help="minimum probability for probmap sampling")
    parser.add_argument("--radius_ratio", type=float, default=0.02, help="probmap radius as a fraction of input size")
    parser.add_argument("--nearby", type=int, default=11, help="local negative window radius for higher-resolution feature maps")
    parser.add_argument("--length_embedding", type=int, default=64, help="network embedding length")
    parser.add_argument("--use_probmap", type=float, default=0, help="start training with probability-map guided sampling")
    parser.add_argument("--vit", type=str, default='SAM_Med2D')
    parser.add_argument("--coarse_stage", type=int, default=1, help="1 trains coarse full-image stage; 0 trains fine crop stage")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--edge_factor", type=float, default=0.3, help="edge-map penalty weight in the finest loss")
    parser.add_argument("--fine_probmap_start_epoch", type=int, default=None, help="fine epoch where probmap sampling starts; -1 keeps random sampling")
    parser.add_argument("--fine_sample_margin_ratio", type=float, default=None, help="fine random point sampling margin ratio inside each crop")
    parser.add_argument("--fine_crop_padding", default=None, choices=[None, 'clamp', 'constant', 'replicate', 'reflect'], help="fine crop padding mode")
    parser.add_argument("--fine_crop_context_size", nargs=2, type=int, default=None, help="fine source crop size before resize")
    parser.add_argument("--fine_search_radius", type=int, default=None, help="fine evaluation local search radius; -1 means global search")
    parser.add_argument("--fine_total_epochs", type=int, default=None, help="override fine-stage total epochs")
    parser.add_argument("--coarse_total_epochs", type=int, default=None, help="override coarse-stage total epochs")
    parser.add_argument("--coarse_random_epochs", type=int, default=None, help="override coarse-stage random-sampling epochs")
    parser.add_argument("--save_seq", type=int, default=None, help="evaluation/checkpoint cadence in epochs")
    parser.add_argument("--num_workers", type=int, default=None, help="override dataloader worker count")
    args = parser.parse_args()
    GLOBAL_SEED = args.seed
    random_all(GLOBAL_SEED)

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SSL')

    merge_from_args(cfg, args, key_list=['nearby', 'add_bias', 'edge_factor', 'clip', 'strength', 'tag', 'seed',\
                                'temprature', 'min_prob', 'radius_ratio', 'length_embedding'])
    if args.fine_probmap_start_epoch is not None:
        cfg.train.fine_probmap_start_epoch = args.fine_probmap_start_epoch
    if args.fine_sample_margin_ratio is not None:
        cfg.train.fine_sample_margin_ratio = args.fine_sample_margin_ratio
    if args.fine_crop_padding is not None:
        cfg.train.fine_crop_padding = args.fine_crop_padding
    if args.fine_crop_context_size is not None:
        cfg.train.fine_crop_context_size = list(args.fine_crop_context_size)
    if args.fine_search_radius is not None:
        cfg.test.fine_search_radius = args.fine_search_radius
    if args.fine_total_epochs is not None:
        cfg.train.fine_total_epochs = args.fine_total_epochs
    if args.coarse_total_epochs is not None:
        cfg.train.coarse_total_epochs = args.coarse_total_epochs
    if args.coarse_random_epochs is not None:
        cfg.train.coarse_random_epochs = args.coarse_random_epochs
    if args.save_seq is not None:
        cfg.train.save_seq = args.save_seq
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    

    # Create Logger
    logger = get_mylogger()
    
    # Create runs dir
    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    tag_ssl = 'coarse_stage' if args.coarse_stage else 'fine_stage'
    runs_dir = os.path.join("../../final_runs/", tag)
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    runs_dir = os.path.join("../../final_runs/", tag, tag_ssl)
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    cfg.runs_dir = runs_dir
    cfg.coarse_stage = args.coarse_stage
    cfg.tag = tag
    set_logger_dir(logger, runs_dir)
    logger.info(cfg)

    # Tester
    # `coarse_stage=1` trains/tests on the resized full image and writes
    # pseudo labels under final_runs/<tag>/coarse_stage/pseudo_labels.
    # `coarse_stage=0` trains/tests on per-landmark crops, using those coarse
    # pseudo labels to decide where each fine crop should be placed.
    if args.coarse_stage:
        from .test_coarse import Tester
        tester = Tester(logger, cfg, tag=args.tag)
        cfg.train.total_epochs = cfg.train.coarse_total_epochs
        cfg.train.random_epochs = cfg.train.coarse_random_epochs
        cfg.train.decay_gamma = cfg.train.coarse_decay_gamma
        cfg.train.decay_step = cfg.train.coarse_decay_step
    else:
        from .test_fine import Tester
        tester = Tester(logger, cfg, tag=args.tag)
        cfg.train.total_epochs = cfg.train.fine_total_epochs
        cfg.train.random_epochs = cfg.train.fine_random_epochs
        cfg.train.decay_gamma = cfg.train.fine_decay_gamma
        cfg.train.decay_step = cfg.train.fine_decay_step
        args.radius_ratio = 0.05

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    GLOBAL_WORKER_ID = None
    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    id_shot = cfg.test.id_oneshot[0]
    dataset_class = select_dataset_SSL_Train(cfg.dataset.dataset)
    # Train samples supply a raw image plus an augmented patch. Oneshot samples
    # are repeated many times by BaseTrainDataset so this iterator does not
    # exhaust during long runs.
    initial_use_probmap = bool(args.use_probmap)
    dataset = dataset_class(cfg, 'Train', use_probmap=initial_use_probmap, \
        radius_ratio=args.radius_ratio, min_prob=args.min_prob, size=cfg.train.input_size, tag_ssl=args.tag)
    oneshot_dataset = dataset_class(cfg, 'Oneshot', use_probmap=initial_use_probmap, \
        radius_ratio=args.radius_ratio, min_prob=args.min_prob, size=cfg.train.input_size, tag_ssl=args.tag, id_oneshot=id_shot)
    # Eagerly validate one sample so missing pseudo labels fail before training starts.
    oneshot_dataset.__getitem__(0)
    oneshot_loader = DataLoader(oneshot_dataset, batch_size=cfg.train.batch_size // 2, drop_last=True, shuffle=True, num_workers=cfg.train.num_workers, worker_init_fn=worker_init_fn)
    oneshot_iterater = iter(oneshot_loader)
    
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size // 2,
                drop_last=True, shuffle=True, num_workers=cfg.train.num_workers, worker_init_fn=worker_init_fn)
    dataset.__getitem__(0)
    net = UNet_Pretrained(3, non_local=True, length_embedding=args.length_embedding, \
      vit=args.vit, image_size=cfg.train.input_size)

    net = net.cuda() 


    optimizer = optim.Adam(params=net.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    net.eval()
    tester.test(net, mode='Infer_Train', dump_label=False)

    best_mre = float("inf")
    best_epoch = None
    best_info_path = Path(runs_dir) / "model_epoch_best.txt"

    probmap_start_epoch = cfg.train.random_epochs + 1
    if (not cfg.coarse_stage) and cfg.train.fine_probmap_start_epoch >= 0:
        probmap_start_epoch = cfg.train.fine_probmap_start_epoch
    probmap_started = initial_use_probmap

    for epoch in range(cfg.train.total_epochs):
        if epoch >= probmap_start_epoch:
            if not probmap_started:
                logger.info("Start probability-map guided sampling steps at epoch %d", epoch)
                if cfg.coarse_stage:
                    net.eval()
                    logger.info("Inferring pseudo labels for training set")
                    tester.test(net, dump_label=True, mode="Infer_Train")
                    net.train()
                # The probability map starts guiding sampled positives only
                # once. Reset the optimizer at this transition, not every epoch,
                # otherwise the LR schedule never progresses.
                dataloader.dataset.use_probmap = True
                oneshot_loader.dataset.use_probmap = True
                # Keep the configured evaluation/checkpoint cadence. Starting
                # probmap at epoch 0 for fine-stage chest should not force a
                # full Test pass after every epoch.
                optimizer = optim.Adam(params=net.parameters(), \
                               lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
                scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)
                probmap_started = True
            else:
                dataloader.dataset.use_probmap = True
                oneshot_loader.dataset.use_probmap = True

        net.train()
        logic_loss_list = list()
        for index, (raw_img, crop_imgs, chosen_y, chosen_x, raw_y, raw_x, edge) in enumerate(dataloader):
            template_list = next(oneshot_iterater)
            # Every dataset uses the same contrastive batch layout: half
            # pseudo-labeled train crops and half true one-shot template crops.
            raw_img = torch.cat([raw_img, template_list[0]], dim=0)
            crop_imgs = torch.cat([crop_imgs, template_list[1]], dim=0)
            chosen_y = torch.cat([chosen_y, template_list[2]], dim=0)
            chosen_x = torch.cat([chosen_x, template_list[3]], dim=0)
            raw_y = torch.cat([raw_y, template_list[4]], dim=0)
            raw_x = torch.cat([raw_x, template_list[5]], dim=0)
            edge = torch.cat([edge, template_list[6]], dim=0)
            
            raw_img = raw_img.cuda()
            crop_imgs = crop_imgs.cuda()
            edge = edge.cuda()
            
            # Extract feature pyramids for the reference image and augmented crop.
            raw_fea_list = net(raw_img)
            crop_fea_list = net(crop_imgs)


            # Six feature maps are returned from coarse to fine. Training uses
            # levels 5..1 here; level 0 is kept for compatibility with older
            # experiments. `raw_y/raw_x` are coordinates in the raw image fed
            # to the network; `chosen_y/chosen_x` are the matching coordinates
            # after the random patch augmentation.
            gt_y, gt_x = raw_y // (2 ** 5), raw_x // (2 ** 5)
            # tmpl_y/tmpl_x are the same point inside the augmented patch.
            tmpl_y, tmpl_x = chosen_y // (2 ** 5), chosen_x // (2 ** 5)
            # Template features come from the augmented crop at the same point.
            tmpl_feature = torch.stack([crop_fea_list[0][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_5 = match_inner_product(raw_fea_list[0], tmpl_feature) * args.temprature
            loss_5 = ce_loss(ret_inner_5, gt_y, gt_x, add_bias=args.add_bias, args=args)

            gt_y, gt_x = raw_y // (2 ** 4), raw_x // (2 ** 4)
            tmpl_y, tmpl_x = chosen_y // (2 ** 4), chosen_x // (2 ** 4)
            tmpl_feature = torch.stack([crop_fea_list[1][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_4 = match_inner_product(raw_fea_list[1], tmpl_feature) * args.temprature
            loss_4 = ce_loss(ret_inner_4, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            gt_y, gt_x = raw_y // (2 ** 3), raw_x // (2 ** 3)
            tmpl_y, tmpl_x = chosen_y // (2 ** 3), chosen_x // (2 ** 3)
            tmpl_feature = torch.stack([crop_fea_list[2][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_3 = match_inner_product(raw_fea_list[2], tmpl_feature) * args.temprature
            loss_3 = ce_loss(ret_inner_3, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            gt_y, gt_x = raw_y // (2 ** 2), raw_x // (2 ** 2)
            tmpl_y, tmpl_x = chosen_y // (2 ** 2), chosen_x // (2 ** 2)
            tmpl_feature = torch.stack([crop_fea_list[3][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_2 = match_inner_product(raw_fea_list[3], tmpl_feature) * args.temprature
            loss_2 = ce_loss(ret_inner_2, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args)

            gt_y, gt_x = raw_y // (2 ** 1), raw_x // (2 ** 1)
            tmpl_y, tmpl_x = chosen_y // (2 ** 1), chosen_x // (2 ** 1)
            tmpl_feature = torch.stack([crop_fea_list[4][[id],:,tmpl_y[id], tmpl_x[id]]\
                for id in range(gt_y.shape[0])]).squeeze()
            ret_inner_1 = match_inner_product(raw_fea_list[4], tmpl_feature) * args.temprature

            loss_1 = ce_loss(ret_inner_1, gt_y, gt_x, nearby=cfg.args.nearby, add_bias=args.add_bias, args=args, edge=edge)

            loss_compare = loss_5 + loss_4 + loss_3 + loss_2 + loss_1
            
            loss = loss_compare

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(np.array([loss_5.cpu().item(), loss_4.cpu().item(), \
                loss_3.cpu().item(), loss_2.cpu().item(), loss_1.cpu().item(), 0]))
        
        losses = np.stack(logic_loss_list).transpose()
        logger.info("Epoch {} Training logic loss 5 {:.3f} 4 {:.3f} 3 {:.3f} 2 {:.3f} 1 {:.3f} rela {:.3f}". \
                    format(epoch, losses[0].mean(), losses[1].mean(), losses[2].mean(),\
                        losses[3].mean(), losses[4].mean(), losses[5].mean()))

        scheduler.step()


        if (epoch + 1) % cfg.train.save_seq == 0:
            net.eval()
            mre = tester.test(net, dump_label=False)

            current_ckpt = Path(runs_dir) / "model_epoch_current.pth"
            best_ckpt = Path(runs_dir) / "model_epoch_best.pth"
            torch.save(net.state_dict(), current_ckpt)
            logger.info("%s latest checkpoint epoch %d MRE %.6f", current_ckpt, epoch, mre)

            if mre < best_mre:
                best_mre = mre
                best_epoch = epoch
                torch.save(net.state_dict(), best_ckpt)
                best_info_path.write_text(
                    f"best_epoch: {best_epoch}\n"
                    f"best_mre: {best_mre:.12f}\n"
                    f"latest_epoch: {epoch}\n"
                )
                logger.info("%s best checkpoint epoch %d MRE %.6f", best_ckpt, epoch, mre)

            prune_numbered_checkpoints(runs_dir, logger)
            cfg.train.last_epoch = epoch
        # Persist the resolved config for reproducibility.
        with open(runs_dir + "/config.txt", "w") as f:
            f.write(cfg.dump())
    
    net.eval()
    tester.test(net, dump_label=True, mode='Infer_Train')

