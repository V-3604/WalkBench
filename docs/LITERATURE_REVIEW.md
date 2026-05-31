# Literature Review

The work this project builds on, grouped by topic, with a line on why each one matters
here.

## Cross-city transfer of image-based walkability models

**Doiron et al. (2022).** *Predicting walking-to-work using street-level imagery and deep
learning in seven Canadian cities.* Scientific Reports 12, 18380.
<https://www.nature.com/articles/s41598-022-22630-1>

Trains segmentation and object-detection models on 1.15M Google Street View images across
seven Canadian cities and reports a pooled R² of about 0.49. The models are trained on all
seven cities pooled together — there is no leave-one-city-out test — and the authors note
that transferability to new contexts "might be limited." Measuring that gap directly is the
starting point for this project.

**Xiang et al. (2025).** *WalkCLIP: Multimodal Learning for Urban Walkability Prediction.*
arXiv:2511.21947. <https://arxiv.org/abs/2511.21947>

From the same lab. Predicts commercial Walk Score in a single city (Minneapolis–St. Paul,
4,660 locations) by combining street view, satellite tiles, and ZIP-code population
embeddings, with CLIP fine-tuned on GPT-4o captions, and reports R² = 0.887. It is
single-city and does not test transfer to other cities. This project is separate work: it
uses Mapillary imagery I collected, Overture infrastructure targets instead of Walk Score,
four cities, and a leave-one-city-out protocol, so the two numbers are not directly
comparable.

**Hosseini et al. (2024).** *Urban Visual Intelligence: Studying Cities with Artificial
Intelligence and Street-Level Imagery.* Annals of the American Association of Geographers.
<https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2313515>

A survey of street-view-based urban analytics that names cross-domain generalization as one
of the field's open problems — which is the gap this project targets.

**Koo et al. (2023).** *Assessment of Perceived and Physical Walkability Using Street View
Images and Deep Learning Technology.* ISPRS International Journal of Geo-Information 12(5),
186. <https://www.mdpi.com/2220-9964/12/5/186>

Uses a pairwise-comparison network to predict *perceived* walkability. A different target
from mine (perception rather than physical infrastructure); included for breadth.

## Frozen vs fine-tuned features under distribution shift

**Kumar et al. (2022).** *Fine-Tuning can Distort Pretrained Features and Underperform
Out-of-Distribution.* ICLR 2022 (Oral). <https://openreview.net/forum?id=UYneFzXSJWh>

Across ten distribution-shift benchmarks, full fine-tuning improves in-distribution accuracy
by about 2% but *hurts* out-of-distribution accuracy by about 7% versus a frozen linear
probe, because fine-tuning distorts pretrained features that were already good for the
shifted setting. This is the closest theoretical match to what I see: the LoRA adapter does
not beat the frozen backbone once the test city is held out.

**Wortsman et al. (2022).** *Robust Fine-Tuning of Zero-Shot Models (WiSE-FT).* CVPR 2022.
<https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf>

Ensembling the zero-shot and fine-tuned CLIP weights recovers out-of-distribution robustness
while keeping in-distribution accuracy. Confirms the fine-tuning/robustness tradeoff is
general for CLIP-family models, not specific to this dataset.

## Sidewalk and crosswalk label quality

**Omar et al. (2022).** *Crowdsourcing and Sidewalk Data: A Preliminary Study on the
Trustworthiness of OpenStreetMap Data in the US.* arXiv:2210.02350.
<https://arxiv.org/abs/2210.02350>

Compares OSM sidewalk coverage across more than 50 US cities and finds it varies widely
(Seattle is much better mapped than Chicago or New York). Since Overture inherits roughly 40%
of its features from OSM (Overture Maps FAQ, <https://overturemaps.org/about/faq/>), that
variability carries straight into the sidewalk and crosswalk labels I train on.

**Saha et al. (2019).** *Project Sidewalk: A Web-based Crowdsourcing Tool for Collecting
Sidewalk Accessibility Data At Scale.* CHI 2019 (Best Paper).
<https://dl.acm.org/doi/10.1145/3290605.3300292>

The Project Sidewalk system, where trained volunteers audit sidewalks and crosswalks from
street-view panoramas. I use its labels as an independent second source to measure how noisy
the Overture sidewalk and crosswalk labels are. Project Sidewalk has live data for Seattle and
a static export for Pittsburgh; the DC endpoint has been down; MSP has no deployment.

**Duan et al. (2022).** *Scaling Crowd+AI Sidewalk Accessibility Assessments: Initial
Experiments Examining Label Quality and Cross-city Training on Performance.* ASSETS 2022.
<https://dl.acm.org/doi/10.1145/3517428.3550381>

Studies label quality and cross-city train/test transfer on Project Sidewalk data across six
cities, including Pittsburgh and Seattle. The most directly related cross-city prior work for
the sidewalk/crosswalk side of this project.

## Aerial imagery and built-environment features

**Robinson et al. (2022).** *Fast Building Segmentation From Satellite Imagery and Few Local
Labels.* CVPR EarthVision Workshop 2022.
<https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/papers/Robinson_Fast_Building_Segmentation_From_Satellite_Imagery_and_Few_Local_Labels_CVPRW_2022_paper.pdf>

Building-footprint segmentation generalizes across regions with a gap that a small amount of
local data can close. This lines up with what I find: building-footprint fraction is the most
transferable target across cities, because aerial building geometry looks similar everywhere.

## Vision encoders used here

**SigLIP 2 — Tschannen et al. (2025).** arXiv:2502.14786.
<https://arxiv.org/abs/2502.14786> An improved SigLIP (captioning pretraining,
self-distillation, masked prediction) that the authors report beats the original at every
model size. I use the SO400m variant as a drop-in swap for SigLIP SO400m.

**DINOv2 — Oquab et al. (2023).** arXiv:2304.07193. <https://arxiv.org/abs/2304.07193>
A self-supervised ViT trained with no text and no labels. Included as a backbone with a
different pretraining recipe, for diversity against the vision-language encoders.

**RemoteCLIP — Liu et al. (2024).** IEEE Transactions on Geoscience and Remote Sensing.
<https://arxiv.org/abs/2306.11029> CLIP fine-tuned on a large remote-sensing image-text
corpus. I try it on the NAIP aerial tile to test whether an aerial-specialized encoder helps
the building-footprint target.

## Spatial cross-validation

**Roberts et al. (2017).** *Cross-validation strategies for data with temporal, spatial,
hierarchical, or phylogenetic structure.* Ecography 40(8), 913–929.
<https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/ecog.02881>

Shows that random cross-validation badly underestimates error on spatially structured data
and recommends spatial block cross-validation — the basis for the 500 m-buffer in-city CV.

**Karasiak et al. (2022).** *Spatially autocorrelated training and validation samples inflate
performance assessment of convolutional neural networks.* ISPRS Open Journal of
Photogrammetry and Remote Sensing.
<https://www.sciencedirect.com/science/article/pii/S2667393222000072>

Quantifies that inflation — random CV can overstate performance by up to about 28% — which is
why I check how much the spatial buffer actually moves the numbers.
