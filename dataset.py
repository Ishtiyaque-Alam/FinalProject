"""
NSCLC-Radiomics Dataset loader.
Reads CT volumes + segmentation masks from DICOM, merges with clinical EHR data,
and prepares samples for the USCNet model.

Key improvements over v1:
  - GTV tumor crop: bounding-box crop around the segmentation mask before resize
    (Lite-ProSENet style) so the model focuses on the tumor region.
  - Random-slice dropout augmentation: randomly zero out 10% of slices to
    simulate missing/noisy slice data.
  - Stronger spatial augmentations (elastic deformation via RandAffine).
  - EHR data serialised into ClinicalBERT natural-language text (no target leakage).
"""

import os
import glob
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, RandFlip, RandRotate90, RandAffine,
    ScaleIntensityRange, Resize, EnsureChannelFirst,
    RandGaussianNoise, RandGaussianSmooth,
    RandZoom, RandAdjustContrast,
)

OLD_PREFIX = "/kaggle/input/nsclc-radiomics/NSCLC-Radiomics/"
NEW_PREFIX = "/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/NSCLC-Radiomics/"

LABEL_MAP = {
    "large cell": 0,
    "squamous cell carcinoma": 1,
    "adenocarcinoma": 2,
    "nos": 3,
}

NUM_CLASSES = len(LABEL_MAP)


def fix_path(path: str) -> str:
    """Replace old Kaggle prefix with new corrected prefix."""
    return path.replace(OLD_PREFIX, NEW_PREFIX)


def load_dicom_series(directory: str) -> np.ndarray:
    """Load a DICOM series from a directory and return a 3D numpy array."""
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(directory)
    if len(dicom_files) == 0:
        dcm_files = sorted(glob.glob(os.path.join(directory, "*.dcm")))
        if len(dcm_files) == 0:
            dcm_files = sorted(glob.glob(os.path.join(directory, "*")))
        dicom_files = dcm_files

    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)  # (D, H, W)
    return array.astype(np.float32)


def load_segmentation(directory: str) -> np.ndarray:
    """Load segmentation mask from a DICOM-SEG or DICOM RT-STRUCT directory."""
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(directory)
        if len(dicom_files) == 0:
            dcm_files = sorted(glob.glob(os.path.join(directory, "*")))
            dicom_files = dcm_files
        reader.SetFileNames(dicom_files)
        image = reader.Execute()
        array = sitk.GetArrayFromImage(image)
        mask = (array > 0).astype(np.float32)
        return mask
    except Exception:
        files = sorted(glob.glob(os.path.join(directory, "*")))
        if len(files) == 0:
            return None
        try:
            ds = pydicom.dcmread(files[0])
            if hasattr(ds, "pixel_array"):
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                return (arr > 0).astype(np.float32)
        except Exception:
            pass
        return None


def gtv_crop(ct_volume: np.ndarray, seg_mask: np.ndarray,
             margin: int = 10) -> tuple:
    """
    Crop a tight bounding box around the GTV (segmentation mask) with
    a configurable margin on each side.  Falls back to the full volume if
    the mask is empty.

    Args:
        ct_volume: (D, H, W) float32 CT array.
        seg_mask:  (D, H, W) binary float32 mask (values 0 / 1).
        margin:    Voxel padding around the mask bounding box.

    Returns:
        Cropped (ct_volume, seg_mask) with the same dtype as input.
    """
    coords = np.argwhere(seg_mask > 0)
    if coords.shape[0] == 0:
        return ct_volume, seg_mask

    d_min, h_min, w_min = coords.min(axis=0)
    d_max, h_max, w_max = coords.max(axis=0)

    D, H, W = ct_volume.shape
    d_min = max(d_min - margin, 0)
    h_min = max(h_min - margin, 0)
    w_min = max(w_min - margin, 0)
    d_max = min(d_max + margin + 1, D)
    h_max = min(h_max + margin + 1, H)
    w_max = min(w_max + margin + 1, W)

    return (
        ct_volume[d_min:d_max, h_min:h_max, w_min:w_max],
        seg_mask[d_min:d_max, h_min:h_max, w_min:w_max],
    )


def row_to_ehr_text(row: pd.Series) -> str:
    """
    Convert a structured EHR row into natural language clinical text
    suitable for ClinicalBERT tokenisation, following the USCNet paper.

    NOTE: Histology is excluded — it is the prediction target.
    """
    age = row.get("age", "unknown")
    gender = row.get("gender", "unknown")
    t_stage = row.get("clinical.T.Stage", "unknown")
    n_stage = row.get("Clinical.N.Stage", "unknown")
    m_stage = row.get("Clinical.M.Stage", "unknown")
    overall_stage = row.get("Overall.Stage", "unknown")
    survival = row.get("Survival.time", "unknown")
    dead = row.get("deadstatus.event", "unknown")

    status = "deceased" if str(dead) == "1" else "alive"

    try:
        age_str = f"{float(age):.1f}"
    except (ValueError, TypeError):
        age_str = str(age)

    text = (
        f"Patient is a {age_str} year old {gender}. "
        f"Diagnosed with non-small cell lung cancer. "
        f"Clinical staging: T{t_stage} N{n_stage} M{m_stage}, overall stage {overall_stage}. "
        f"Survival time: {survival} days, patient status: {status}."
    )
    return text


class NSCLCDataset(Dataset):
    """
    Dataset for NSCLC-Radiomics with CT, Segmentation masks, and EHR text.

    EHR fields are serialised into natural language text for ClinicalBERT.

    Improvements over v1:
      - GTV tumor crop before resize (focuses network on tumor region).
      - Stronger augmentations: elastic deformation, zoom, contrast jitter.
      - Random-slice dropout during training.

    Args:
        metadata_csv: Path to phase1_metadata CSV (patient_id, ct_path, seg_path).
        clinical_csv: Path to clinical CSV with EHR columns.
        volume_size: Target (D, H, W) for resampled volumes.
        is_train: Whether to apply data augmentation.
        gtv_margin: Voxel margin around GTV bounding box (0 = no crop).
    """

    def __init__(
        self,
        metadata_csv: str,
        clinical_csv: str,
        volume_size: tuple = (64, 128, 128),
        is_train: bool = True,
        gtv_margin: int = 10,
    ):
        self.volume_size = volume_size
        self.is_train = is_train
        self.gtv_margin = gtv_margin

        meta_df = pd.read_csv(metadata_csv)
        clinical_df = pd.read_csv(clinical_csv)

        meta_df["ct_path"] = meta_df["ct_path"].apply(fix_path)
        meta_df["seg_path"] = meta_df["seg_path"].apply(fix_path)

        merged = meta_df.merge(
            clinical_df, left_on="patient_id", right_on="PatientID", how="inner"
        )

        merged = merged[merged["Histology"].notna()]
        merged = merged[merged["Histology"].str.lower().isin(LABEL_MAP.keys())]
        merged = merged.reset_index(drop=True)

        self.data = merged
        self._build_transforms()

    def get_labels(self) -> list:
        """Return integer labels for all samples (used for stratified split)."""
        return [
            LABEL_MAP[str(row["Histology"]).lower().strip()]
            for _, row in self.data.iterrows()
        ]

    def _build_transforms(self):
        """Build MONAI transforms for CT volumes."""
        spatial_size = list(self.volume_size)

        self.resize = Resize(spatial_size=spatial_size, mode="trilinear")
        self.resize_seg = Resize(spatial_size=spatial_size, mode="nearest")

        if self.is_train:
            self.augment = Compose([
                RandFlip(spatial_axis=0, prob=0.5),
                RandFlip(spatial_axis=1, prob=0.5),
                RandFlip(spatial_axis=2, prob=0.3),
                RandRotate90(prob=0.3, spatial_axes=(1, 2)),
                RandAffine(
                    prob=0.4,
                    rotate_range=(0.1, 0.1, 0.1),
                    shear_range=(0.05, 0.05),
                    translate_range=(5, 5, 5),
                    scale_range=(0.05, 0.05, 0.05),
                    mode="bilinear",
                    padding_mode="border",
                ),
                RandZoom(prob=0.3, min_zoom=0.85, max_zoom=1.15, mode="trilinear"),
                RandGaussianNoise(prob=0.3, mean=0.0, std=0.05),
                RandGaussianSmooth(prob=0.15),
                RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2)),
            ])
        else:
            self.augment = None

    def _random_slice_dropout(self, ct_tensor: torch.Tensor,
                               drop_prob: float = 0.1) -> torch.Tensor:
        """Zero out a random fraction of depth slices to augment robustness."""
        C, D, H, W = ct_tensor.shape
        n_drop = max(1, int(D * drop_prob))
        drop_indices = torch.randperm(D)[:n_drop]
        ct_tensor = ct_tensor.clone()
        ct_tensor[:, drop_indices, :, :] = 0.0
        return ct_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        ct_path = row["ct_path"]
        seg_path = row["seg_path"]
        label_str = str(row["Histology"]).lower().strip()
        label = LABEL_MAP[label_str]

        ct_volume = load_dicom_series(ct_path)
        seg_mask = load_segmentation(seg_path)

        if seg_mask is None:
            seg_mask = np.zeros_like(ct_volume)

        # Ensure both are 3D (D, H, W)
        if ct_volume.ndim == 2:
            ct_volume = ct_volume[np.newaxis]
        if seg_mask.ndim == 2:
            seg_mask = seg_mask[np.newaxis]
        if ct_volume.ndim == 4:
            ct_volume = ct_volume.squeeze(0) if ct_volume.shape[0] == 1 else ct_volume[..., 0]
        if seg_mask.ndim == 4:
            seg_mask = seg_mask.squeeze(0) if seg_mask.shape[0] == 1 else seg_mask[..., 0]

        # Match seg mask to CT volume shape via zoom (nearest-neighbor)
        if seg_mask.shape != ct_volume.shape:
            from scipy.ndimage import zoom
            zoom_factors = tuple(t / s for s, t in zip(seg_mask.shape, ct_volume.shape))
            seg_mask = zoom(seg_mask, zoom_factors, order=0)
            seg_mask = (seg_mask > 0.5).astype(np.float32)

        # GTV tumor crop: tight bounding box around the lung tumor
        if self.gtv_margin >= 0:
            ct_volume, seg_mask = gtv_crop(ct_volume, seg_mask, margin=self.gtv_margin)

        # Window/level for lung CT (HU: -1000 to 400)
        ct_volume = np.clip(ct_volume, -1000, 400)
        ct_volume = (ct_volume - (-1000)) / (400 - (-1000))

        # Add channel dimension: (1, D, H, W)
        ct_volume = ct_volume[np.newaxis]
        seg_mask = seg_mask[np.newaxis]

        ct_tensor = torch.from_numpy(ct_volume)
        seg_tensor = torch.from_numpy(seg_mask)

        ct_tensor = self.resize(ct_tensor)
        seg_tensor = self.resize_seg(seg_tensor)

        if self.augment is not None:
            ct_tensor = self.augment(ct_tensor)
            # Random slice dropout only during training
            ct_tensor = self._random_slice_dropout(ct_tensor, drop_prob=0.10)

        ehr_text = row_to_ehr_text(row)

        return {
            "ct": ct_tensor,            # (1, D, H, W)
            "seg_gt": seg_tensor,        # (1, D, H, W)
            "ehr_text": ehr_text,        # str - natural language for ClinicalBERT
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": row["patient_id"],
        }
