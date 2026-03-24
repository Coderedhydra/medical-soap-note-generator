# Medical SOAP Note Generator

A clean, production-style AI pipeline that converts medical dictation audio into structured SOAP notes with a validation layer to catch likely LLM misclassification.

## 1) Project Overview

This project demonstrates an end-to-end healthcare documentation workflow:

- **Input:** clinician dictation audio file
- **Step 1:** speech-to-text transcription using **faster-whisper**
- **Step 2:** SOAP note generation using an open-source HuggingFace instruct model
- **Step 3:** rule-based quality checks for section misclassification
- **Output:** JSON containing transcript, SOAP note, and validation results

## 2) System Pipeline

```text
Audio File
   ↓
transcribe_audio()        [faster-whisper]
   ↓
Transcript Text
   ↓
generate_soap()           [HF instruct LLM + strict prompt]
   ↓
SOAP JSON
   ↓
validate_soap()           [rule-based constraints]
   ↓
Final JSON Output
```

## 3) Setup Instructions

### Prerequisites

- Python 3.10+
- `ffmpeg` installed and available in PATH (required by Whisper stack)

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) How to Run

```bash
python main.py /path/to/dictation.wav
```

Optional flags:

```bash
python main.py /path/to/dictation.wav \
  --whisper-model base \
  --llm-model Qwen/Qwen2.5-3B-Instruct \
  --device cpu \
  --compute-type int8
```

## 5) Sample Input & Output

### Example dictation (conceptual)

> "Patient reports worsening headache and nausea for two days. Vitals: BP 148/92, HR 102. Exam shows mild photophobia. Impression is migraine exacerbation. Start sumatriptan, hydrate, and follow up in 1 week."

### Example output

```json
{
  "transcript": "Patient reports worsening headache and nausea for two days. Vitals: BP 148/92, HR 102. Exam shows mild photophobia. Impression is migraine exacerbation. Start sumatriptan, hydrate, and follow up in 1 week.",
  "soap_note": {
    "Subjective": "Worsening headache and nausea for two days.",
    "Objective": "BP 148/92, HR 102. Mild photophobia on exam.",
    "Assessment": "Migraine exacerbation.",
    "Plan": "Start sumatriptan, encourage hydration, follow up in 1 week."
  },
  "validation": {
    "is_valid": true,
    "issues": []
  }
}
```

## 6) Design Decisions

- **Why faster-whisper?**
  - High-quality transcription with practical CPU/GPU deployment options.
  - Good speed/accuracy tradeoff for production workflows.

- **Why open-source instruct LLM via HuggingFace?**
  - Keeps pipeline local/open-source.
  - Model swap flexibility through CLI (`--llm-model`).

- **Why strict prompt + post-processing?**
  - Prompt enforces SOAP boundaries and JSON-only behavior.
  - JSON extraction + normalization hardens output formatting.

- **Why rule-based validation?**
  - LLMs can hallucinate or misclassify sections.
  - Lightweight checks provide practical safety signals before downstream use.

## 7) Limitations

- LLM output quality depends on selected model capacity.
- Rule-based validation catches common mistakes only (not full clinical correctness).
- No external medical ontology or coding standards (e.g., ICD/SNOMED) integration.
- No diarization/speaker attribution in transcription.

## 8) Future Improvements

- Fine-tune or instruction-tune model on clinical SOAP datasets.
- Add medical NER + ontology constraints for stronger validation.
- Add retrieval augmentation (RAG) against trusted clinical guidelines.
- Add confidence scoring and human-in-the-loop review workflow.
- Package as API service with audit logging and PHI-safe deployment controls.

## File Structure

```text
.
├── main.py
├── README.md
└── requirements.txt
```
