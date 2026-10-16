import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from datasets import select_dataset_SSL_Infer
from evaluation.eval import Evaluater
from models.network import UNet_Pretrained
from utils.mylogger import get_mylogger, set_logger_dir


def match_cos(feature, template):
    """Return a cosine-similarity map for one feature level and landmark."""
    feature = feature.permute(1, 2, 0)
    fea_l2 = torch.norm(feature, dim=-1)
    template_l2 = torch.norm(template, dim=-1)
    inner_product = (feature * template).sum(-1)
    cos_similarity = inner_product / (fea_l2 * template_l2 + 1e-3)
    return torch.clamp(cos_similarity, 0, 1)


class Tester:
    def __init__(self, logger, cfg, tag=None):
        self.cfg = cfg
        self.tag = tag
        self.evaluater = Evaluater(logger, cfg.dataset.num_landmarks, cfg.dataset.eval_radius)
        self.logger = logger
        self.id_landmarks = [i for i in range(cfg.dataset.num_landmarks)]

    def test(self, net, dump_label=False, mode='Test', id_shot=None):
        """Evaluate the fine-stage model on one dataset split.

        Fine inference first builds one template feature vector per landmark
        from the one-shot image, then searches each test crop for the best
        matching feature response. Coordinates are mapped back from local crop
        space to original image space before metric computation.
        """
        coarse_as_fine = bool(self.cfg.test.coarse_as_fine)
        if net is not None:
            net.eval()
        runs_dir = self.cfg.runs_dir

        if id_shot is None:
            id_shots = self.cfg.test.id_oneshot
        else:
            id_shots = [id_shot]

        dataset_class = select_dataset_SSL_Infer(self.cfg.dataset.dataset)
        dataset = dataset_class(self.cfg, mode, size=self.cfg.train.input_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # shot_features[id_landmark][depth] stores the support feature vector
        # used to localize that landmark in every fine crop.
        shot_features = []

        if not coarse_as_fine:
            for id_shot in id_shots:
                pre_shot_features_n_d = []
                one_shot_loader = dataset_class(
                    self.cfg, mode='Oneshot', size=self.cfg.train.input_size, id_oneshot=id_shot,
                )
                self.logger.info(f'ID Oneshot : {id_shot}')
                self.evaluater.reset()
                shot_images, patch_landmarks, template_patches, landmarks = one_shot_loader.__getitem__(0)
                shot_images = torch.stack(shot_images).cuda()

                for image in shot_images:
                    with torch.no_grad():
                        features = net(image.unsqueeze(0))
                    pre_shot_features_n_d.append(features)

                for id_landmark, net_feature in enumerate(pre_shot_features_n_d):
                    feature_list = [None] * 6
                    for id_depth in range(6):
                        landmark = patch_landmarks[id_landmark]
                        tmpl_y = landmark[0] // (2 ** (5 - id_depth))
                        tmpl_x = landmark[1] // (2 ** (5 - id_depth))
                        h, w = net_feature[id_depth][0, 0].shape
                        tmpl_y, tmpl_x = min(tmpl_y, h - 1), min(tmpl_x, w - 1)
                        assert tmpl_y < h, f'template shape {h} chosen y {tmpl_y}'
                        assert tmpl_x < w, f'template shape {w} chosen x {tmpl_x}'
                        mark_feature = net_feature[id_depth][0, :, tmpl_y, tmpl_x]
                        feature_list[id_depth] = mark_feature.detach()

                    shot_features.append(feature_list)

        for images, gt_landmark, id_str, scale_rate, crop_pos, raw_image, coarse_landmark, gt_adjust, crop_scale in tqdm(dataloader):
            images = torch.stack(images).cuda()
            num_landmarks, b, c, h, w = images.shape

            id_str = id_str[0]
            scale_rate = [scale_rate[0][0].item(), scale_rate[1][0].item()]

            pred_landmarks_y, pred_landmarks_x = [], []
            if coarse_as_fine:
                for id_mark in range(num_landmarks):
                    pred_landmarks_y.append(coarse_landmark[id_mark][0].item())
                    pred_landmarks_x.append(coarse_landmark[id_mark][1].item())
            else:
                for id_mark in range(len(shot_features)):
                    with torch.no_grad():
                        features = net(images[id_mark])

                    final_cos = torch.ones_like(images[id_mark][0, 0]).cuda()
                    for id_depth in range(5):
                        cos_similarity = match_cos(
                            features[id_depth].squeeze(), shot_features[id_mark][id_depth],
                        )
                        cos_similarity = F.interpolate(
                            cos_similarity.unsqueeze(0).unsqueeze(0),
                            scale_factor=2 ** (5 - id_depth), mode='bilinear', align_corners=False,
                        ).squeeze()
                        final_cos = final_cos * cos_similarity

                    cos_lists = final_cos

                    search_radius = int(self.cfg.test.fine_search_radius)
                    if search_radius >= 0:
                        # Restrict the fine argmax around the coarse point for
                        # low-texture or repeated-texture datasets. A negative
                        # radius keeps the historical global argmax behavior.
                        center_y = (
                            (coarse_landmark[id_mark][0] - crop_pos[id_mark][0])
                            / crop_scale[id_mark][0]
                        ).to(cos_lists.device).float()
                        center_x = (
                            (coarse_landmark[id_mark][1] - crop_pos[id_mark][1])
                            / crop_scale[id_mark][1]
                        ).to(cos_lists.device).float()
                        yy = torch.arange(h, device=cos_lists.device).float().view(-1, 1)
                        xx = torch.arange(w, device=cos_lists.device).float().view(1, -1)
                        mask = (yy - center_y).pow(2) + (xx - center_x).pow(2) <= search_radius ** 2
                        cos_lists = cos_lists.masked_fill(~mask, -1)

                    chosen_landmark = cos_lists.argmax().item()

                    # Map model-input coordinates back through the optional
                    # context-crop resize and then to original image space.
                    local_y = chosen_landmark // w
                    local_x = chosen_landmark % w
                    pred_landmarks_y.append(
                        (local_y * crop_scale[id_mark][0] + crop_pos[id_mark][0]).item()
                    )
                    pred_landmarks_x.append(
                        (local_x * crop_scale[id_mark][1] + crop_pos[id_mark][1]).item()
                    )

            preds = [np.array(pred_landmarks_y), np.array(pred_landmarks_x)]
            self.evaluater.record(preds, gt_landmark, scale_rate)
            self.evaluater.save_img(raw_image, preds, gt_landmark, runs_dir, id_str)

            if dump_label:
                self.evaluater.save_preds(preds, runs_dir, id_str)

        self.evaluater.cal_metrics()
        return self.evaluater.mre


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a CC2D fine-stage landmark checkpoint")
    parser.add_argument("--tag", default='gold', help="final_runs tag to evaluate")
    parser.add_argument("--dataset", default='head', choices=['head', 'hand', 'leg', 'chest'], help="dataset config")
    parser.add_argument("--test", type=int, default=0, help="legacy no-op flag kept for old scripts")
    parser.add_argument("--epoch", default="current", help="checkpoint selector: current/latest, best, or a numeric epoch")
    parser.add_argument("--coarse_stage", type=int, default=0, help="0 evaluates fine_stage; 1 switches runs_dir to coarse_stage")
    parser.add_argument("--length_embedding", type=int, default=64, help="network embedding length")
    parser.add_argument("--vit", type=str, default='SAM_Med2D', help="backbone name passed to UNet_Pretrained")
    parser.add_argument("--use_mlp", type=int, default=0, help="enable MLP head in UNet_Pretrained")
    parser.add_argument("--coarse_as_fine", type=int, default=0, help="evaluate coarse pseudo labels through the fine evaluator")
    parser.add_argument("--fine_crop_padding", default=None, choices=[None, 'clamp', 'constant', 'replicate', 'reflect'], help="override fine crop padding mode")
    parser.add_argument("--fine_crop_context_size", nargs=2, type=int, default=None, help="override fine source crop size before resize")
    parser.add_argument("--fine_search_radius", type=int, default=None, help="override local search radius in fine crop pixels; -1 means global search")
    parser.add_argument("--skip_infer_train", type=int, default=0, help="skip Infer_Train prediction dump after Test evaluation")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'SSL')
    cfg.coarse_stage = args.coarse_stage
    cfg.test.coarse_as_fine = args.coarse_as_fine
    if args.fine_crop_padding is not None:
        cfg.train.fine_crop_padding = args.fine_crop_padding
    if args.fine_crop_context_size is not None:
        cfg.train.fine_crop_context_size = list(args.fine_crop_context_size)
    if args.fine_search_radius is not None:
        cfg.test.fine_search_radius = args.fine_search_radius

    logger = get_mylogger()

    tag = args.tag
    tag_ssl = 'coarse_stage' if args.coarse_stage else 'fine_stage'
    cfg.tag = tag
    runs_dir = "../../final_runs/" + tag + '/' + tag_ssl
    runs_path = Path(runs_dir)
    cfg.runs_dir = runs_dir
    runs_path.mkdir(parents=True, exist_ok=True)
    set_logger_dir(logger, runs_dir)
    logger.info(cfg)

    tester = Tester(logger, cfg, tag=args.tag)

    net = None
    if not args.coarse_as_fine:
        net = UNet_Pretrained(
            3, non_local=True, length_embedding=args.length_embedding,
            vit=args.vit, use_mlp=args.use_mlp, image_size=cfg.train.input_size,
        )
        net = net.cuda()
        net.eval()

        epoch = str(args.epoch).lower()
        if epoch in ["0", "current", "latest"]:
            ckpt_path = runs_path / "model_epoch_current.pth"
        elif epoch == "best":
            ckpt_path = runs_path / "model_epoch_best.pth"
        else:
            ckpt_path = runs_path / f"model_epoch_{int(epoch)}.pth"
        logger.info("Load CKPT %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
        net.load_state_dict(ckpt)
    else:
        logger.info("coarse_as_fine=1: skip network load and evaluate coarse pseudo labels in fine geometry")

    tester.test(net, dump_label=True)
    if not args.skip_infer_train:
        tester.test(net, mode='Infer_Train', dump_label=True)
