import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .segment_anything import sam_model_registry as local_sam_registry
from .segment_anything import SamPredictor
from .segment_anything.build_sammed2d import sammed2d_model_registry


def _check_ckpt(checkpoint):
    if not checkpoint:
        raise ValueError("checkpoint is required.")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


class MedSamPredictor:
    """
    MedSAM 专用 predictor：
    - 预处理: resize 到 1024 + min-max 归一化到 [0,1]
    - 推理: image_encoder + prompt_encoder + mask_decoder
    """

    def __init__(self, medsam_model, device):
        self.model = medsam_model
        self.device = device
        self.image_embedding = None
        self.original_size = None

    @staticmethod
    def _to_uint8_rgb(image):
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @torch.no_grad()
    def set_image(self, image):
        img = self._to_uint8_rgb(image)
        h, w = img.shape[:2]
        self.original_size = (h, w)

        img_1024 = np.array(Image.fromarray(img).resize((1024, 1024), Image.BILINEAR), dtype=np.float32)
        img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), 1e-8, None)
        img_1024_tensor = torch.from_numpy(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.image_embedding = self.model.image_encoder(img_1024_tensor)

    @torch.no_grad()
    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=False,
        return_logits=False,
    ):
        if self.image_embedding is None or self.original_size is None:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        if box is None:
            raise ValueError("MedSAM box prompt is required.")

        h, w = self.original_size
        box_np = np.asarray(box, dtype=np.float32).reshape(-1, 4)
        scale = np.array([w, h, w, h], dtype=np.float32)
        box_1024 = box_np / scale * 1024.0
        box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=self.device)
        if box_torch.ndim == 2:
            box_torch = box_torch[:, None, :]

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        if return_logits:
            full_pred = F.interpolate(low_res_logits, size=(h, w), mode="bilinear", align_corners=False)
        else:
            low_res_prob = torch.sigmoid(low_res_logits)
            full_pred = F.interpolate(low_res_prob, size=(h, w), mode="bilinear", align_corners=False)
            full_pred = full_pred > 0.5

        masks_np = full_pred[0].detach().cpu().numpy()
        iou_np = iou_predictions[0].detach().cpu().numpy()
        low_res_np = low_res_logits[0].detach().cpu().numpy()
        return masks_np, iou_np, low_res_np


def build_sam_predictor(checkpoint, device, model_type="vit_h"):
    _check_ckpt(checkpoint)
    if model_type not in local_sam_registry:
        raise ValueError(f"Unsupported SAM model_type: {model_type}")
    model = local_sam_registry[model_type](checkpoint=checkpoint).to(device)
    model.eval()
    return SamPredictor(model)


def build_medsam_predictor(checkpoint, device, model_type="vit_b"):
    _check_ckpt(checkpoint)
    if model_type not in local_sam_registry:
        raise ValueError(f"Unsupported MedSAM model_type: {model_type}")
    medsam_model = local_sam_registry[model_type](checkpoint=checkpoint).to(device)
    medsam_model.eval()
    return MedSamPredictor(medsam_model, device=device)


def build_sammed2d_predictor(
    checkpoint,
    device,
    model_type="vit_b",
    image_size=256,
    encoder_adapter=True,
):
    _check_ckpt(checkpoint)
    if model_type not in sammed2d_model_registry:
        raise ValueError(f"Unsupported SAM-Med2D model_type: {model_type}")
    from .segment_anything.predictor_sammed import SammedPredictor

    sammed_model = sammed2d_model_registry[model_type](
        checkpoint=checkpoint,
        image_size=int(image_size),
        encoder_adapter=bool(encoder_adapter),
    )
    sammed_model = sammed_model.to(device)
    sammed_model.eval()
    return SammedPredictor(sammed_model)
