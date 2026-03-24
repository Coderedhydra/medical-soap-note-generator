# Medical SOAP Note Generator

A minimal, production-style AI pipeline that converts medical dictation audio into structured SOAP notes with a validation layer to reduce common LLM misclassification issues.

## 1) Project Overview

This project implements an end-to-end healthcare documentation workflow:

- **Input:** clinician dictation audio
- **Transcription:** faster-whisper
- **Structuring:** open-source HuggingFace model
- **Validation:** rule-based SOAP sanity checks
- **Output:** strict JSON containing transcript, SOAP note, generation mode, and validation results

## 2) System Pipeline

```text
Audio Input
  -> transcribe_audio() [faster-whisper]
  -> generate_soap()    [LLM prompt + strict JSON parsing]
       -> fallback rule-based SOAP if model output is not valid JSON
  -> validate_soap()    [misclassification checks]
  -> Final JSON Output
```

## 3) Setup Instructions

### Prerequisites

- Python 3.10+
- ffmpeg installed and available in PATH

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) How to Run

```bash
python main.py /path/to/dictation.wav
```

Optional:

```bash
python main.py /path/to/dictation.wav \
  --whisper-model base \
  --llm-model google/flan-t5-base \
  --device cpu \
  --compute-type int8
```

> Note: default LLM is `google/flan-t5-base` to reduce model download size compared with multi-GB chat models.

## 5) Sample Input & Output

### Sample dictation

> "Patient reports chest pain for 2 days. Blood pressure is 140/90, heart rate 95. Impression: possible angina. Start aspirin and order ECG. Follow up in one week."

### Sample output

```json
{
  "transcript": "Patient reports chest pain for 2 days. Blood pressure is 140/90, heart rate 95. Impression: possible angina. Start aspirin and order ECG. Follow up in one week.",
  "soap_note": {
    "subjective": "Patient reports chest pain for 2 days.",
    "objective": "Blood pressure is 140/90, heart rate 95.",
    "assessment": "Impression: possible angina.",
    "plan": "Start aspirin and order ECG. Follow up in one week."
  },
  "generation_mode": "llm",
  "validation": {
    "is_valid": true,
    "issues": []
  }
}
```

## 6) Design Decisions

- **faster-whisper** for practical local transcription performance and quality.
- **Open-source LLM with HuggingFace pipeline** for local flexibility.
- **Strong prompt + JSON extraction** to enforce machine-readable SOAP output.
- **Rule-based validator** to explicitly address LLM misclassification risk.
- **Fallback SOAP extractor** to avoid hard failures when model output is not valid JSON.
- **Lazy dependency imports** so CLI help works even before ML dependencies are installed.

## 7) Limitations

- Fallback extractor is heuristic and less accurate than a good instruction model.
- Rule-based validation catches common mistakes, not full clinical correctness.
- No medical ontology constraints (ICD/SNOMED) yet.
- No speaker diarization.

## 8) Future Improvements

- Fine-tune on clinical SOAP corpora.
- Add medical NER + ontology-backed validation.
- Add confidence scoring and escalation rules.
- Add optional retrieval from guideline corpora.
- Expose as API service with audit logs and PHI controls.
