import json
import math
import os
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from imgaug.augmentables import KeypointsOnImage
from torch.utils.data import Dataset

from .augment import to_PIL

# DataLoader workers already provide process-level parallelism; leaving
# OpenCV to spawn its own large thread pool can exceed the host thread limit.
cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass

def aug_img_and_kps(image, landmark_list, augmentation):
    tmp = np.concatenate(landmark_list).reshape(-1, 2)[:, ::-1]
    kps = KeypointsOnImage.from_xy_array(tmp, shape=image.shape)
    image, kps_augmented = augmentation(image=image, keypoints=kps)
    kps = kps_augmented
    
    landmark_list = np.round(kps.to_xy_array()[:,::-1]).astype(np.int32).tolist()
    image = torch.tensor(image)
    image = torch.stack([image, image, image], 0)
    return image, landmark_list

class BasePointDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.list = []
        self.size = None
        self.original_size = None
        self.coarse = True

        self.images = []
        self.gts = []
        self.paths = []
        self.scale_rates = []

    def __len__(self) -> int:
        return len(self.list)
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def load_data(self):
        # Subclasses fill image ids, landmarks, images, and spacing metadata.
        raise NotImplementedError
    
    def _get_landmark_gt(self, item):
        raise NotImplementedError
    
    def _get_image(self, item):
        raise NotImplementedError
    
    def resize_coarse_landmark_dataset(self):
        for key, landmark in self.gts.items():
            original_spacing = self.scale_rates[key]
            image = self.images[key]
            original_size = image.size[::-1]
            new_landmark = [self._resize_coarse_landmark(point, original_size) for point in landmark]
            new_scale_rate = self._compute_spacing(original_spacing, original_size)
            self.gts[key] = new_landmark
            self.scale_rates[key] = new_scale_rate
    
    def _resize_coarse_landmark(self, landmark, original_size):
        res = [0] * len(landmark)
        res[1] = int(landmark[0] * self.size[1] / original_size[1]) # x
        res[0] = int(landmark[1] * self.size[0] / original_size[0]) # y
        return res

    def _compute_spacing(self, original_spacing, original_size):
        pixel_spacing_x, pixel_spacing_y = original_spacing

        scale_rate_x = pixel_spacing_x * original_size[1] / self.size[1]
        scale_rate_y = pixel_spacing_y * original_size[0] / self.size[0]

        return [scale_rate_y, scale_rate_x]
    
    def _resize_fine_landmark(self, landmark):
        res = [0] * len(landmark)
        res[1] = int(landmark[0])
        res[0] = int(landmark[1])
        return res

    def resize_fine_landmark_dataset(self):
        for key, landmark in self.gts.items():
            new_landmark = [self._resize_fine_landmark(point) for point in landmark]
            self.gts[key] = new_landmark
    
    def remap_landmark_to_original(self, landmark, original_shape):
        res = [0] * len(landmark)
        res[1] = int(landmark[1] * original_shape[0] / self.size[1])
        res[0] = int(landmark[0] * original_shape[1] / self.size[0])
        return res

    def _pad_tensor_crop(self, crop, pad, mode):
        if mode == 'constant':
            return F.pad(crop, pad, mode='constant', value=0)
        if mode in ['replicate', 'reflect']:
            return F.pad(crop.unsqueeze(0), pad, mode=mode).squeeze(0)
        raise ValueError(f'Unsupported fine_crop_padding: {mode}')

    def _crop_tensor_with_padding(self, image, start_y, start_x, crop_h, crop_w, pad_mode):
        height, width = image.shape[1:3]
        end_y = start_y + crop_h
        end_x = start_x + crop_w

        src_y0, src_y1 = max(start_y, 0), min(end_y, height)
        src_x0, src_x1 = max(start_x, 0), min(end_x, width)
        crop = image[:, src_y0:src_y1, src_x0:src_x1]

        pad_top = max(0, -start_y)
        pad_left = max(0, -start_x)
        pad_bottom = max(0, end_y - height)
        pad_right = max(0, end_x - width)
        if pad_top or pad_bottom or pad_left or pad_right:
            crop = self._pad_tensor_crop(crop, (pad_left, pad_right, pad_top, pad_bottom), pad_mode)
        return crop

    def _resize_fine_crop_and_point(self, crop, point_y, point_x, source_h, source_w):
        target_h, target_w = self.patch_h, self.patch_w
        if (source_h, source_w) != (target_h, target_w):
            crop = F.interpolate(
                crop.unsqueeze(0), size=(target_h, target_w),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            point_y = point_y * target_h / source_h
            point_x = point_x * target_w / source_w
        point_y = int(np.clip(round(point_y), 0, target_h - 1))
        point_x = int(np.clip(round(point_x), 0, target_w - 1))
        return crop, [point_y, point_x]

    def load_course_landmarks(self, id):
        # input id
        assert(self.tag_ssl is not None)
        image = self.images[id]
        original_shape = image.size
        pseudo_path = os.path.join('..', '..', 'final_runs', self.tag_ssl, 'coarse_stage', 'pseudo_labels', f'{id}.json')
        if not os.path.exists(pseudo_path):
            raise FileNotFoundError(
                f'Missing coarse pseudo label for {id}: {pseudo_path}. '
                f'Run scripts_SSL.test_coarse with --tag {self.tag_ssl} before fine-stage inference/training.'
            )
        with open(pseudo_path, 'rb') as f:
            pred_landmark = json.load(f)
        pred_landmark = [pred_landmark[str(i)] for i in range(self.num_landmark)]
        landmark_list = [self.remap_landmark_to_original(landmark, original_shape) for landmark in pred_landmark]
        return landmark_list

class BaseTrainDataset(BasePointDataset):
    def __init__(self, cfg=None, mode='Train', size=[384, 384],  tag_ssl=None,\
                 radius_ratio=0.05, min_prob=0.1, **kwargs) -> None:
        BasePointDataset.__init__(self)

        self.mode = mode
        self.original_size = cfg.dataset.original_size
        self.size = tuple(int(v) for v in size)
        self.tag_ssl = cfg.tag
        self.coarse = cfg.coarse_stage
        # `patch_size` is the tensor size fed to the network. Fine-stage can
        # optionally crop a larger context from the source image and resize it
        # back to this input size. Defaults keep the historical behavior.
        self.patch_size = tuple(int(v) for v in cfg.train.patch_size)
        if not self.coarse:
            self.patch_size = self.size
        self.patch_h, self.patch_w = self.patch_size
        self.crop_context_size = self.patch_size
        fine_context_size = tuple(int(v) for v in cfg.train.fine_crop_context_size)
        if not self.coarse and fine_context_size:
            self.crop_context_size = fine_context_size
        self.crop_context_h, self.crop_context_w = self.crop_context_size
        self.fine_crop_padding = cfg.train.fine_crop_padding
        self.fine_sample_margin_ratio = float(cfg.train.fine_sample_margin_ratio)
        self.use_probmap = bool(kwargs.get('use_probmap', False))
        self.radius_ratio = radius_ratio
        self.min_prob = min_prob

        normalize = transforms.Normalize([0], [1])
        transformList = []
        if self.coarse:
            transformList.append(transforms.Resize(self.size))      
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.basic_transform = transforms.Compose(transformList)

        self.Radius = int(max(size) * radius_ratio)
        guassian_mask = torch.zeros(2*self.Radius, 2*self.Radius, dtype=torch.float)
        for i in range(2*self.Radius):
            for j in range(2*self.Radius):
                distance = np.linalg.norm([i+1 - self.Radius, j+1 - self.Radius])
                if distance < self.Radius:
                    guassian_mask[i][j] = math.exp(- 0.5 * math.pow(distance, 2) /\
                        math.pow(self.Radius, 2))
        self.guassian_mask = guassian_mask

        transform_list = [
            transforms.ColorJitter(brightness=0.15, contrast=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0], [1])
        ]
        self.full_image_intensity_aug = transforms.Compose(transform_list)

        data_aug_params = cfg.dataset.AUGMENTATION
        self.patch_augmentation = iaa.Sequential([
            iaa.Affine(translate_px={"x": (-data_aug_params.TRANSLATION_X, data_aug_params.TRANSLATION_X),
                                     "y": (-data_aug_params.TRANSLATION_Y, data_aug_params.TRANSLATION_Y)},
                       scale=[1 - data_aug_params.SF, 1],
                       rotate=[-data_aug_params.ROTATION_FACTOR, data_aug_params.ROTATION_FACTOR]),
            iaa.Multiply(mul=(1 - data_aug_params.INTENSITY_FACTOR, 1 + data_aug_params.INTENSITY_FACTOR)),
            # iaa.GammaContrast(data_aug_params.GAMMA_RANGE),
            iaa.ElasticTransformation(alpha=(data_aug_params.ELASTIC_STRENGTH, data_aug_params.ELASTIC_STRENGTH),
                                      sigma=data_aug_params.ELASTIC_SMOOTHNESS, order=3)
        ])

        self.patch_augmentation_nomultiply = iaa.Sequential([
            iaa.Affine(translate_px={"x": (-data_aug_params.TRANSLATION_X, data_aug_params.TRANSLATION_X),
                                     "y": (-data_aug_params.TRANSLATION_Y, data_aug_params.TRANSLATION_Y)},
                       scale=[1 - data_aug_params.SF, 1],
                       rotate=[-data_aug_params.ROTATION_FACTOR, data_aug_params.ROTATION_FACTOR]),
            # iaa.Multiply(mul=(1 - data_aug_params.INTENSITY_FACTOR, 1 + data_aug_params.INTENSITY_FACTOR)),
            # iaa.GammaContrast(data_aug_params.GAMMA_RANGE),
            iaa.ElasticTransformation(alpha=(data_aug_params.ELASTIC_STRENGTH, data_aug_params.ELASTIC_STRENGTH),
                                      sigma=data_aug_params.ELASTIC_SMOOTHNESS, order=3)
        ])

        self.full_template_augmentation = iaa.Sequential([
            iaa.Affine(translate_px={"x": (-data_aug_params.TRANSLATION_X, data_aug_params.TRANSLATION_X),
                                     "y": (-data_aug_params.TRANSLATION_Y, data_aug_params.TRANSLATION_Y)},
                       scale=[1 - data_aug_params.SF, 1],
                       rotate=[-3, 3]),
            # iaa.Multiply(mul=(1 - data_aug_params.INTENSITY_FACTOR, 1 + data_aug_params.INTENSITY_FACTOR)),
            # iaa.GammaContrast(data_aug_params.GAMMA_RANGE),
            # iaa.ElasticTransformation(alpha=(data_aug_params.ELASTIC_STRENGTH, data_aug_params.ELASTIC_STRENGTH),
            #                           sigma=data_aug_params.ELASTIC_SMOOTHNESS, order=3)
        ])

        self.load_data()
        
        if self.coarse:
            self.resize_coarse_landmark_dataset()
        else:
            self.resize_fine_landmark_dataset()

        num_repeat = 10
        if mode == 'Train':
            temp = self.list.copy()
            for _ in range(num_repeat):
                self.list.extend(temp)
        
        # Repeat the one-shot sample so long training runs never exhaust it.
        if mode == 'Oneshot':
            temp = self.list.copy()
            for _ in range(1000000):
                self.list.extend(temp)
        
    
    def gen_probmap(self, landmark):
        prob_map = torch.zeros(self.size).float()
        for i in range(len(landmark)):
            margin_x_left = max(0, landmark[i][1] - self.Radius)
            margin_x_right = min(self.size[1], landmark[i][1] + self.Radius)
            margin_y_bottom = max(0, landmark[i][0] - self.Radius)
            margin_y_top = min(self.size[0], landmark[i][0] + self.Radius)
            prob_map[margin_y_bottom:margin_y_top, margin_x_left:margin_x_right] += \
                self.guassian_mask[0:margin_y_top-margin_y_bottom, 0:margin_x_right-margin_x_left]
        return torch.clamp(prob_map, self.min_prob, 1)

    def load_landmarks(self, id, pseudo=False):
        tag_stage = 'coarse_stage' if self.coarse else 'fine_stage'
        if pseudo:
            assert self.tag_ssl is not None
            pseudo_path = os.path.join('..', '..', 'final_runs', self.tag_ssl, tag_stage, 'pseudo_labels', f'{id}.json')
            with open(pseudo_path, 'rb') as f:
                pred_landmark = json.load(f)
            return [pred_landmark[str(i)] for i in range(self.num_landmark)]
        return self._get_landmark_gt(id)
    
    def crop_patch(self, image, landmark):
        """Crop a fine-stage training sample around a coarse/GT landmark.

        The returned crop is always `self.patch_h x self.patch_w` for the
        network. `self.crop_context_size` controls how much source-image
        context is first cropped before optional resize.
        """
        crop_h, crop_w = self.crop_context_h, self.crop_context_w
        height, width = image.shape[1:3]
        y, x = int(landmark[0]), int(landmark[1])

        half_h, half_w = crop_h // 2, crop_w // 2
        sub_half_h, sub_half_w = half_h // 2, half_w // 2

        if self.fine_crop_padding == 'clamp':
            legal_min_x = half_w
            legal_max_x = max(width - (crop_w - half_w), legal_min_x)
            legal_min_y = half_h
            legal_max_y = max(height - (crop_h - half_h), legal_min_y)

            min_center_x = max(x - sub_half_w, legal_min_x)
            max_center_x = min(x + sub_half_w, legal_max_x)
            min_center_y = max(y - sub_half_h, legal_min_y)
            max_center_y = min(y + sub_half_h, legal_max_y)

            if min_center_x > max_center_x:
                random_x = min(max(x, legal_min_x), legal_max_x)
            else:
                random_x = random.randint(int(min_center_x), int(max_center_x))
            if min_center_y > max_center_y:
                random_y = min(max(y, legal_min_y), legal_max_y)
            else:
                random_y = random.randint(int(min_center_y), int(max_center_y))

            start_x = random_x - half_w
            start_y = random_y - half_h
            cropped_patch = image[:, start_y:start_y + crop_h, start_x:start_x + crop_w]
        else:
            random_x = random.randint(int(x - sub_half_w), int(x + sub_half_w))
            random_y = random.randint(int(y - sub_half_h), int(y + sub_half_h))
            start_x = random_x - half_w
            start_y = random_y - half_h
            cropped_patch = self._crop_tensor_with_padding(
                image, start_y, start_x, crop_h, crop_w, self.fine_crop_padding,
            )

        adjusted_x = x - start_x
        adjusted_y = y - start_y
        return self._resize_fine_crop_and_point(cropped_patch, adjusted_y, adjusted_x, crop_h, crop_w)

    def __getitem_fine(self, index):
        item = self.list[index]
        gt_landmark = self.gts[item['ID']]
        coarse_landmark = self.load_course_landmarks(item['ID'])
        if self.mode == 'Oneshot':
            coarse_landmark = gt_landmark
        else:
            gt_landmark = coarse_landmark

        item['image'] = self.images[item['ID']]
        item['image'] = self.basic_transform(item['image'])

        target_landmark = random.choice(gt_landmark)
        item['image'], target_landmark = self.crop_patch(item['image'], target_landmark)
        gt_landmark = target_landmark

        if self.mode == 'Oneshot':
            item['image'], gt_landmark = aug_img_and_kps(item['image'][0].numpy(), [gt_landmark], self.patch_augmentation_nomultiply)
            item['image'] = torch.clamp(item['image'], 0, 1)


        # Build a half-resolution Canny edge prior for the finest loss level.
        image = item['image'][0].numpy() * 255
        image = image.astype(np.uint8)
        resize_wh = (self.size[1] // 2, self.size[0] // 2)
        image = cv2.resize(image, resize_wh)

        edge = np.zeros((self.size[0] // 2, self.size[1] // 2), dtype=np.float32)
        gamma_range = np.arange(0.5, 1.0, 0.1)
        for gamma in gamma_range:
            augmentation = iaa.Sequential([iaa.GammaContrast(gamma, gamma + 0.1)])
            image_aug = augmentation(image=image)

            edges = cv2.Canny(image_aug, 40, 80)
            edge += edges

        # loss_1 is computed on the half-resolution feature map, so keep edge
        # at H/2 x W/2 instead of resizing it back to the raw image size.
        edge = (edge > 0).astype(np.uint8) * 255

        # Fine stage compares the full fine crop against an augmented copy of
        # that same crop.  Random supervision points are sampled in [y, x].
        patch_h, patch_w = self.patch_h, self.patch_w
        margin_x = 0
        margin_y = 0

        sample_margin = float(np.clip(self.fine_sample_margin_ratio, 0.0, 0.45))
        min_x, max_x = int(sample_margin * patch_w), int((1 - sample_margin) * patch_w)
        min_y, max_y = int(sample_margin * patch_h), int((1 - sample_margin) * patch_h)
        chosen_x_raw = np.random.randint(min_x, max(max_x, min_x + 1))
        chosen_y_raw = np.random.randint(min_y, max(max_y, min_y + 1))
        raw_y, raw_x = chosen_y_raw + margin_y, chosen_x_raw + margin_x

        if self.use_probmap:
            if type(target_landmark[0]) == int:
                target_landmark = [target_landmark]
            probmap = self.gen_probmap(target_landmark)
            while (probmap[raw_y, raw_x] < random.random()):
                chosen_x_raw = np.random.randint(min_x, max(max_x, min_x + 1))
                chosen_y_raw = np.random.randint(min_y, max(max_y, min_y + 1))
                raw_y, raw_x = chosen_y_raw + margin_y, chosen_x_raw + margin_x

        def augment(crop_raw_image, kps):
            crop_aug_image, kps_augmented = self.patch_augmentation(image=crop_raw_image, keypoints=kps)
            point = np.round(kps_augmented.to_xy_array()[0]).astype(np.int32)
            while point[0] < 0 or point[0] >= patch_w or point[1] < 0 or point[1] >= patch_h:
                crop_aug_image, kps_augmented = self.patch_augmentation(image=crop_raw_image, keypoints=kps)
                point = np.round(kps_augmented.to_xy_array()[0]).astype(np.int32)
            assert 0 <= point[0] < patch_w
            assert 0 <= point[1] < patch_h
            return crop_aug_image, kps_augmented

        crop_raw_image = item['image'][:, margin_y:margin_y + patch_h, margin_x:margin_x + patch_w][0].numpy()
        tmp = np.array([chosen_x_raw, chosen_y_raw]).reshape(1, 2)
        kps = KeypointsOnImage.from_xy_array(tmp, shape=crop_raw_image.shape)
        crop_aug_image, kps_augmented = augment(crop_raw_image, kps)
        crop_aug_image = torch.tensor(crop_aug_image)
        crop_aug_image = torch.clamp(crop_aug_image, 0, 1)
        crop_aug_image = torch.stack([crop_aug_image, crop_aug_image, crop_aug_image], 0)

        kps_augmented = np.round(kps_augmented.to_xy_array()[0]).astype(np.int32)
        chosen_y, chosen_x = kps_augmented[1], kps_augmented[0]
        chosen_y, chosen_x = torch.tensor([chosen_y]), torch.tensor([chosen_x])
        item['image'] = self.full_image_intensity_aug(to_PIL(item['image']))

        return item['image'], crop_aug_image, chosen_y, chosen_x, raw_y, raw_x, edge


    def __getitem_coarse(self, index):
        item = self.list[index]
        gt_landmark = self.gts[item['ID']]

        item['image'] = self.images[item['ID']]
        # Basic transform normalizes the resized full image.
        item['image'] = self.basic_transform(item['image'])
        if self.mode == 'Oneshot':
            # Half of each contrastive batch is an augmented one-shot template.
            item['image'], gt_landmark = aug_img_and_kps(item['image'][0].numpy(), gt_landmark, self.full_template_augmentation)

        # Build a half-resolution Canny edge prior for the finest loss level.
        image = item['image'][0].numpy() * 255
        image = image.astype(np.uint8)
        cv2_size = [self.size[1] // 2, self.size[0] // 2]
        image = cv2.resize(image, cv2_size)

        edge = np.zeros(cv2_size[::-1])
        gamma_range = np.arange(0.5, 1.0, 0.1)
        for gamma in gamma_range:
            augmentation = iaa.Sequential([iaa.GammaContrast(gamma, gamma+0.1)])
            image_aug = augmentation(image=image)
            edges = cv2.Canny(image_aug, 40, 80)
            edge += edges
        # loss_1 is computed on the half-resolution feature map, so keep edge
        # at H/2 x W/2 instead of resizing it back to the raw image size.
        edge = (edge > 0).astype(np.uint8) * 255

        size = self.size
        
        # Coarse stage samples a smaller patch inside the resized full image.
        # patch_h/patch_w remain separate so rectangular inputs stay valid.
        patch_h, patch_w = self.patch_h, self.patch_w
        margin_x = np.random.randint(0, max(size[1] - patch_w, 1))
        margin_y = np.random.randint(0, max(size[0] - patch_h, 1))

        chosen_x_raw = np.random.randint(int(0.1 * patch_w), int(0.9 * patch_w))
        chosen_y_raw = np.random.randint(int(0.1 * patch_h), int(0.9 * patch_h))
        raw_y, raw_x = chosen_y_raw + margin_y, chosen_x_raw + margin_x

        if self.use_probmap:
            # Coarse stage uses pseudo labels to bias sampling after probmap starts.
            if self.mode == 'Train':
                pred_landmark = self.load_landmarks(item['ID'], pseudo=True)
            else:
                pred_landmark = gt_landmark
            probmap = self.gen_probmap(pred_landmark)
            while probmap[raw_y, raw_x] < random.random():
                margin_x = np.random.randint(0, max(size[1] - patch_w, 1))
                margin_y = np.random.randint(0, max(size[0] - patch_h, 1))
                chosen_x_raw = np.random.randint(int(0.1 * patch_w), int(0.9 * patch_w))
                chosen_y_raw = np.random.randint(int(0.1 * patch_h), int(0.9 * patch_h))
                raw_y, raw_x = chosen_y_raw + margin_y, chosen_x_raw + margin_x
        
        crop_raw_image = item['image'][:, margin_y:margin_y + patch_h, margin_x:margin_x + patch_w]
        # Apply intensity augmentation to the sampled raw crop before patch augmentation.
        crop_raw_image = self.full_image_intensity_aug(to_PIL(crop_raw_image)).numpy()[0]
        tmp = np.array([chosen_x_raw, chosen_y_raw]).reshape(1, 2)
        kps = KeypointsOnImage.from_xy_array(tmp, shape=crop_raw_image.shape)
        # Rotate/intensity-augment the sampled patch and keep the supervision
        # point inside the augmented crop.
        crop_aug_image, kps_augmented = self.patch_augmentation(image=crop_raw_image, keypoints=kps)
        point = np.round(kps_augmented.to_xy_array()[0]).astype(np.int32)
        while point[0] < 0 or point[0] >= patch_w or point[1] < 0 or point[1] >= patch_h:
            crop_aug_image, kps_augmented = self.patch_augmentation(image=crop_raw_image, keypoints=kps)
            point = np.round(kps_augmented.to_xy_array()[0]).astype(np.int32)
        crop_aug_image = torch.tensor(crop_aug_image)
        crop_aug_image = torch.stack([crop_aug_image, crop_aug_image, crop_aug_image], 0)
        
        
        kps_augmented = np.round(kps_augmented.to_xy_array()[0]).astype(np.int32)
        chosen_y, chosen_x = kps_augmented[1], kps_augmented[0]
        chosen_y, chosen_x = torch.tensor([chosen_y]), torch.tensor([chosen_x])
        item['image'] = self.full_image_intensity_aug(to_PIL(item['image']))

        return item['image'], crop_aug_image, chosen_y, chosen_x, raw_y, raw_x, edge
    
    def __getitem__(self, index):
        if self.coarse:
            return self.__getitem_coarse(index)
        else:
            return self.__getitem_fine(index)


class BaseInferDataset(BasePointDataset):
    def __init__(self, cfg=None, mode='Train', size=[384, 384], **kwargs) -> None:
        BasePointDataset.__init__(self)

        self.size = tuple(int(v) for v in size)
        self.patch_size = tuple(int(v) for v in cfg.train.patch_size)
        if not cfg.coarse_stage:
            self.patch_size = self.size
        self.patch_h, self.patch_w = self.patch_size
        self.crop_context_size = self.patch_size
        fine_context_size = tuple(int(v) for v in cfg.train.fine_crop_context_size)
        if not cfg.coarse_stage and fine_context_size:
            self.crop_context_size = fine_context_size
        self.crop_context_h, self.crop_context_w = self.crop_context_size
        self.fine_crop_padding = cfg.train.fine_crop_padding
        self.original_size = cfg.dataset.original_size
        self.coarse = cfg.coarse_stage
        self.tag_ssl = cfg.tag

        normalize = transforms.Normalize([0], [1])
        transformList = []
        if self.coarse:
            transformList.append(transforms.Resize(self.size))      
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.basic_transform = transforms.Compose(transformList)

        self.load_data()

        if self.coarse:
            self.resize_coarse_landmark_dataset()
        else:
            self.resize_fine_landmark_dataset()

    def crop_center_patch(self, image, landmark):
        """Return a deterministic fine-stage crop centered on a coarse point.

        When `fine_crop_padding` is not `clamp`, the crop is allowed to extend
        outside the image and is padded. This keeps edge landmarks centered
        instead of shifting them toward the crop border.
        """
        crop_h, crop_w = self.crop_context_h, self.crop_context_w
        height, width = image.shape[1:3]
        y, x = int(landmark[0]), int(landmark[1])

        if self.fine_crop_padding == 'clamp':
            start_y = min(max(y - crop_h // 2, 0), max(height - crop_h, 0))
            start_x = min(max(x - crop_w // 2, 0), max(width - crop_w, 0))
            cropped_patch = image[:, start_y:start_y + crop_h, start_x:start_x + crop_w]
        else:
            start_y = y - crop_h // 2
            start_x = x - crop_w // 2
            cropped_patch = self._crop_tensor_with_padding(
                image, start_y, start_x, crop_h, crop_w, self.fine_crop_padding,
            )

        adjusted_y = y - start_y
        adjusted_x = x - start_x
        cropped_patch, adjusted = self._resize_fine_crop_and_point(
            cropped_patch, adjusted_y, adjusted_x, crop_h, crop_w,
        )
        crop_scale = [crop_h / self.patch_h, crop_w / self.patch_w]
        return cropped_patch, adjusted, [start_y, start_x], crop_scale


    def __getitem_coarse(self, index):
        item = self.list[index]
        item['image'] = self.images[item['ID']]
        item['image'] = self.basic_transform(item['image'])
        
        gt_landmark = self.gts[item['ID']]

        scale_rate = self.scale_rates[item['ID']]

        if self.mode != 'Oneshot':
            # 图 gt spacing
            return item['image'], gt_landmark, item['ID'], scale_rate

            #     to_PIL(template_patches[id]).save('template.jpg')
        return item['image'], gt_landmark
    
    def __getitem_fine(self, index):
        item = self.list[index]
        item['image'] = self.images[item['ID']]
        item['image'] = self.basic_transform(item['image'])

        gt_landmark = self.gts[item['ID']]
        coarse_landmark = self.load_course_landmarks(item['ID'])
        if self.mode == 'Oneshot':
            coarse_landmark = gt_landmark
        
        scale_rate = self.scale_rates[item['ID']]

        image_patches = list()
        crop_pos = list()
        crop_scale = list()
        adjusted_pos = list()
        for landmark in coarse_landmark:
            crop_img, adjusted_pos_, crop_pos_, crop_scale_ = self.crop_center_patch(item['image'], landmark)
            image_patches.append(crop_img)
            crop_pos.append(crop_pos_)
            crop_scale.append(crop_scale_)
            adjusted_pos.append(adjusted_pos_)

        gt_adjust = []
        for idx, gt_lm in enumerate(gt_landmark):
            gt_adust_y = (gt_lm[0] - crop_pos[idx][0]) / crop_scale[idx][0]
            gt_adust_x = (gt_lm[1] - crop_pos[idx][1]) / crop_scale[idx][1]
            gt_adjust.append([gt_adust_y, gt_adust_x])


        if self.mode != 'Oneshot':
            return image_patches, gt_landmark, item['ID'], scale_rate, crop_pos, item[
                'image'], coarse_landmark, gt_adjust, crop_scale

        patch_landmark_list = list()
        template_patches = torch.zeros([self.num_landmark, 3, self.patch_h, self.patch_w])

        return image_patches, adjusted_pos, template_patches, gt_landmark


    def __getitem__(self, index):
        if self.coarse:
            return self.__getitem_coarse(index)
        else:
            return self.__getitem_fine(index)
