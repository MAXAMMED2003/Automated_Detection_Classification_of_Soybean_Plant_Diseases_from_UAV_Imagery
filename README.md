# Soybean Plant Disease Detection from UAV Imagery

This repo holds the code behind our final-year project: spotting soybean leaf diseases
(Healthy, Mosaic, Rust, Caterpillar & Semilooper) from drone-captured field images,
using a two-stage pipeline — YOLOv11 to find and crop out leaves, then a fine-tuned
ConvNeXt-Tiny to classify each crop.

This is a team project, so before anything else — a quick word on who did what.

## Context & my role

This work is being done under the supervision of **Dr. Srinka Basu, University of
Kalyani**, as part of a multi-member research team. The base models (the YOLOv11
detector and the fine-tuned ConvNeXt classifier) were trained by the team as a whole.

**My specific contribution:**
- I designed and ran the **end-to-end evaluation** of the full pipeline — chaining
  detection and classification together and testing it on whole, uncropped UAV field
  images (as opposed to the neatly cropped leaf dataset the classifier was trained on).
  This is the part of the project I can walk through and defend in detail.
- I'm currently building out a **prototype-generation module** (K-means clustering on
  ConvNeXt features) as an extension, aimed at eventually handling disease classes the
  model hasn't seen during training. This one's still a work in progress — see the
  "Known issues" section below, I'm not hiding it.

I'm calling this out explicitly because I'd rather be upfront about which parts are
mine than have it look like I'm claiming the whole pipeline solo.

## Pipeline overview

Nothing fancy, just two models chained together:

```
train_convnext_classifier.py
        │  (trains a base 6-class classifier, saves best_convnext_model.pth)
        ▼
convnext_on_UAV_cropped_labeled_dataset.ipynb
        │  (fine-tunes that model on 4 UAV disease classes)
        ├──────────────────────────────┐
        ▼                              ▼
end_to_end_pipeline_evaluation   prototype_generation.ipynb
   (chains it with YOLO, evaluates      (feature extraction + K-means,
    on whole field images)               exploratory / ongoing)
        ▲
        │  (best.pt)
yolo_v11_training.ipynb   ← trained independently, no dependency on ConvNeXt
```

## What's in each file

| File | What it does | Status |
|---|---|---|
| `train_convnext_classifier.py` | Fine-tunes ConvNeXt-Tiny on a base 6-class leaf dataset | Done |
| `convnext-on-uav-cropped-labeled-dataset.ipynb` | Adapts that model to our 4 UAV disease classes (Healthy / Mosaic / Rust / Caterpillar & Semilooper) | Done — **94.2% accuracy, 0.94 weighted F1** on the held-out crop test set |
| `yolo-v11-training.ipynb` | Trains YOLOv11m-seg to detect and segment individual leaves in a field image | Done — **mAP50 = 0.866**, mAP50-95 = 0.745 |
| `latest-pipeline-for-exam-version-3.ipynb` | Chains YOLO + ConvNeXt and evaluates the combined system on 1,480 whole UAV field images (not pre-cropped) | Done — **84.5% accuracy, 0.98 ROC-AUC** |
| `prototype-generation.ipynb` | Extracts ConvNeXt features and clusters them per class (K-means + elbow method), building toward prototype-based classification | In progress |

The third-to-last number (84.5%/0.98 ROC-AUC) is the one I'd point to first if you
want to know "does this actually work" — it's tested on raw field images, not the
curated crop dataset, so it's closer to how this would perform in practice.

## Known issues / what's left

- `prototype-generation.ipynb` currently has a naming mismatch bug in the summary cell
  (`KeyError: 'centers'`) that I haven't fixed yet. The clustering itself runs fine —
  it's just the display/summary step that breaks. Fixing this and then actually using
  the prototypes for something (a nearest-centroid classifier, most likely) is next on
  my list.
- These notebooks were run on Kaggle, so the file paths (`/kaggle/input/...`,
  `/kaggle/working/...`) won't work out of the box on a different machine. I haven't
  gotten around to making this properly portable yet — treat this as a record of what
  was done rather than a plug-and-play repo for now.

## Dataset

Soybean UAV imagery dataset, hosted on Kaggle. (https://data.mendeley.com/datasets/hkbgh5s3b7/1)

---

If anything here is unclear or you want to see the raw training logs, the notebooks
have all outputs saved inline — nothing's been stripped out.
