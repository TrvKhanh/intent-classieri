# Fine-tune Intent Classification Model


This project addresses the **user intent classification task** in an **e-commerce RAG (Retrieval-Augmented Generation) system**, enabling the system to understand whether a user query is:  

- **chat**: Expressing opinions, emotions, reviews, praise/criticism, or general comments (no request for specific information)  
- **retrieval**: Asking about products, prices, stock availability, policies, comparisons, etc.  
- **retrieval-phone**: Specific queries targeting the phone module  

The model is fine-tuned from **Google FLAN-T5-small (77M parameters)** using **LoRA**, which reduces training cost while improving efficiency.

---

## Data Design

### 1. Data Structure

The dataset is stored in JSON format as follows:

```json
{
  "input": "Bên mình có bán poco x7 pro k ạ",
  "output": {
    "router": "retrieval-phone",
    "infor": "poco x7 pro"
  }
}

### 2. Data Processing Pipeline

#### Step 1: Data Collection and Labeling
- File: `data_raw/lableling_data.py`
- Uses Google Gemini API for automatic labeling
- Supports rate limiting handling, retry logic, and multiple API key management
- Input: CSV file with `content` column
- Output: CSV file with `label` column

#### Step 2: Text Cleaning
- File: `clean_text.py`
- **Processing steps:**
  1. Convert to lowercase
  2. Replace abbreviations with full words (using `abbreviations.json`)
     - Examples: "k" → "không", "bn" → "bạn", "đc" → "được"
  3. Remove emojis (Unicode emoji patterns)
  4. Remove special characters (keep only letters, numbers, Vietnamese diacritics, spaces)
  5. Normalize whitespace (remove extra spaces)

- **File abbreviations.json**: Contains abbreviation → full word mapping (750+ entries)

#### Step 3: Format Conversion for Training
- File: `dataset_train.py`
- **Input format**: JSON with `input` and `output`
- **Output format**: HuggingFace Dataset with:
  - `input_text`: "classify: {input}" (add prefix to specify task)
  - `target_text`: JSON string of `output` (e.g., `{"router": "retrieval", "infor": "..."}`)

**Conversion example:**
```python
# Input
{
  "input": "bên mình có bán poco x7 pro không ạ",
  "output": {"router": "retrieval-phone", "infor": "poco x7 pro"}
}

# Output for model
input_text: "classify: bên mình có bán poco x7 pro không ạ"
target_text: '{"router": "retrieval-phone", "infor": "poco x7 pro"}'
```

### 3. Data Statistics

- **Total samples**: ~19,000+ samples (from `data.json`)
- **Train/Test split**: 80/20 (configured in `train.py`)
- **Max input length**: 128 tokens
- **Max target length**: 64 tokens

---

## Model Architecture

### 1. Base Model

**Google FLAN-T5-small**
- **Architecture**: Encoder-Decoder Transformer (T5-based)
- **Parameters**: 77M (note: FLAN-T5-small is actually 77M, not 250M)
- **Task**: Sequence-to-Sequence (Seq2Seq)
- **Pre-trained on**: Instruction-following tasks, multilingual data

### 2. Fine-tuning with LoRA

**LoRA (Low-Rank Adaptation)**
- **Rank (r)**: 16
- **LoRA Alpha**: 32
- **Target modules**: `["q", "v", "k", "o", "wi_0", "wi_1", "wo"]`
  - `q`, `v`, `k`, `o`: Attention layers (Query, Value, Key, Output)
  - `wi_0`, `wi_1`, `wo`: Feed-forward layers
- **Task type**: `SEQ_2_SEQ_LM`

**Advantages of LoRA:**
- Reduces number of trainable parameters (only train ~1-2% of original parameters)
- Reduces memory footprint and training time
- Easy to switch between different tasks
- Prevents overfitting with small datasets

### 3. Training Architecture

```
Input Text (Vietnamese)
    ↓
[Tokenization] (max_length=128)
    ↓
[FLAN-T5 Encoder]
    ↓
[LoRA Adapters] (r=16, alpha=32)
    ↓
[FLAN-T5 Decoder]
    ↓
Output: JSON string
    ↓
[Post-processing] → Parse JSON → Extract router & infor
```

## Directory Structure

```
fine-tune-intent/
├── README.md                 # This file
├── requirement.txt           # Dependencies
├── abbreviations.json        # Abbreviation → full word mapping
│
├── data.json                 # Raw data (not cleaned)
├── data_train.json          # Cleaned data (auto-generated)
│
├── clean_text.py            # Text cleaning script
├── dataset_train.py         # Dataset loading and conversion
├── train.py                 # Main training script
├── inference.py             # Inference script (not implemented yet)
│
└── data_raw/                # Raw data directory
    ├── data_raw.csv         # Raw data from scraping
    └── lableling_data.py    # Automatic labeling script using Gemini API
```
