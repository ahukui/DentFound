
## 1. Installation
> **⚠️ Important**  
> These instructions are for **Linux only** now. 
---
### 1) Clone the Repository
```bash
git clone https://github.com/ahukui/DentFound.git
cd DentFound
```
### 2) Create and Activate Conda Environment
```bash
conda create -n dentfound python=3.10 -y
conda activate dentfound
```
### 3) Install Core Package
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
### 4) Install Training Dependencies (Optional)
```bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
### 5) Upgrade to the Latest Codebase
```bash
git pull
pip install -e .
#If you encounter import errors after upgrading, try reinstalling flash-attn:
pip install flash-attn --no-build-isolation --no-cache-dir
```
## 2. Model Details 
```bash
Dentfoud mainly consists of three parts: 1) Visual Encoder; 2) Projection Layer; 3) Large Language Model (LLM).
The overall processing pipeline can be summarized as:
Visual Encoder → Visual Features → Projection Layer → Aligned Embedding → Large Language Model (LLM).
 ```
### 1) Visual Encoder
```bash
DentFound employs the pre-trained **CLIP ViT-L/14** model as the visual backbone.  
The encoder is initialized with its official pre-trained weights prior to PR image training.
Pre-trained weights are available at:
https://huggingface.co/openai/clip-vit-large-patch14
```
### 2) Projection Layer
```bash
The projection layer bridges the dimensional gap between the visual encoder and the language model.
This module maps visual features from the multimodal encoder space to the language model embedding
space, ensuring seamless cross-modal alignment.
The projection layer is constracted by fully connected layer and multi-layer MLP with GELU.
```
### 3) Large Language Model (LLM)
```bash
We choosed Vicuna [1] as LLM, as it has the best instruction following capabilities in language tasks
among publicly available checkpoints.
The LLM is initialized with its official pre-trained weights prior to dental knowledge training.
Pre-trained weights are available at:
https://huggingface.co/lmsys/vicuna-7b-v1.5
[1] Chiang, Wei-Lin, et al. "Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality."
See https://vicuna.lmsys.org (accessed 14 April 2023) 2.3 (2023):6.
```
## 3. Model Development
```bash
For DentFound training, we propose a progressive learning paradigm that enables stepwise knowledge acquisition,
progressing from discrete point-level cues to holistic surface representations. Report generation is decoupled
into two stages—diagnosis and composition—thereby addressing the limitations imposed by incomplete and sparse
dental reports. Representative samples of the training data are provided in the Dataset folder.
```
### 1) Prepare data
```bash
For each paired panoramic radiograph (PR) image and corresponding report, we construct a diversified training
corpus comprising four complementary data types, enabling multi-granularity supervision:

### 1. Tooth-level Training Data
Focuses on individual teeth, providing precise and localized diagnostic signals.

**Example:**
- **Q:** <PR Image> Please provide a diagnosis for Tooth 18.  
- **A:** Tooth 18 has a carious lesion.

### 2. Quadrant-level Training Data
Aggregates multiple teeth within a quadrant, encouraging the model to capture regional anatomical and pathological
relationships.

**Example:**
- **Q:** <PR Image> Diagnose the first quadrant.  
- **A:** Tooth 11 has a carious lesion. Tooth 12 is normal. Tooth 13 ...


### 3. Disease-oriented Training Data
Shifts the focus from anatomical structures to disease presence, promoting global disease recognition and multi-tooth
association.

**Example:**
- **Q:** <PR Image> Is there dental caries?  
- **A:** Teeth 18 and 32 have carious lesions.

### 4. Report-level Training Data
Requires comprehensive interpretation, simulating real-world clinical reporting by jointly modeling findings and
impressions.

**Example:**
- **Q:** <PR Image> Describe the findings and impressions of this image.  
- **A:**  
  - **Findings:** Tooth 51 shows radiolucency in the crown ...  
  - **Impressions:** Tooth 51 has a carious lesion ...

### Summary
These four data formats enable progressive learning from localized tooth-level cues to holistic clinical reasoning,
forming the foundation of DentFound’s multi-granularity training paradigm.
All of this training data is recorded in JSON format.
## Example Training Data Format
```json
[
  {
    "id": "c973890f-a12e-4bfb-aaca-4bf62daf2030",
    "image": "P0086918-101143697-2025-02-19.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>Please provide a diagnosis for Tooth 23."
      },
      {
        "from": "gpt",
        "value": "Tooth 23 (upper left canine) has periapical radiolucency."
      }
    ]
  },
  {
    "id": "3798e94f-f32e-4355-b202-9c7aa8aef702",
    "image": "P0086918-101143697-2025-02-19.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>Please provide a diagnosis for Tooth 37."
      },
      {
        "from": "gpt",
        "value": "Tooth 37 (lower left second molar) has a residual root."
      }
    ]
  }
......
]
```
### 2) Tranining
```bash
The model was trained using **bfloat16 mixed precision** to ensure training stability, leveraging **8 × NVIDIA
A100 GPUs (80 GB each)** in a distributed setting.
Here, We provied two training strategies:
- **Fine-tuning**, where all model parameters are updated.
- **LoRA (Low-Rank Adaptation)**, a parameter-efficient alternative for resource-constrained scenarios.

If you are interested in finetuning the model to your own task/data, careful data formatting is essential. In our implementation,
the **panoramic radiograph (PR) image and the corresponding tooth information are concatenated as a unified input**.
## Example of Training Command (DeepSpeed)
deepspeed train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./vicuna-v1-3-7b \
    --version v1 \
    --data_path ./Dataset/training.json \
    --image_folder ./Dataset/images/ \
    --vision_tower ./clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./pretrain_checkpoints/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb

For users with limited training data or computational resources, we recommend using **LoRA**, which significantly reduces training
cost while maintaining competitive performance.
## Example of Training Command (DeepSpeed)
deepspeed train_mem.py \
  --lora_enable True \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 5e-4 \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path ./checkpoints/ \
  --version v1 \
  --data_path ./Dataset/training.json \
  --image_folder "./Dataset/images/" \
  --vision_tower ./clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/lora \
  --num_train_epochs 6 \
  --per_device_train_batch_size 24 \
  --per_device_eval_batch_size 24 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 50000 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 500 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True
```
## Related Projects

Parts of our code are adapted from or inspired by the following projects. We sincerely appreciate their contributions:

- **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)**
- **[LLaVA: Large Language and Vision Assistant](https://github.com/haotian-liu/LLaVA)**
- **[Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/EvolvingLMMs-Lab/Otter)**



