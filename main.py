import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from faster_whisper import WhisperModel
from transformers import pipeline


SOAP_KEYS = ["Subjective", "Objective", "Assessment", "Plan"]


class PipelineError(Exception):
    """Base exception for pipeline errors."""


class TranscriptionError(PipelineError):
    """Raised when transcription fails."""


class SOAPGenerationError(PipelineError):
    """Raised when SOAP generation fails."""


@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]


def transcribe_audio(
    audio_path: str,
    whisper_model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    """Transcribe an audio file with faster-whisper and return a clean transcript."""
    if not os.path.exists(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if os.path.getsize(audio_path) == 0:
        raise TranscriptionError("Audio file is empty.")

    try:
        model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    if not transcript.strip():
        raise TranscriptionError("No speech detected in the audio.")

    return re.sub(r"\s+", " ", transcript).strip()


def _build_soap_prompt(transcript: str) -> str:
    return f"""
You are a clinical documentation assistant.
Convert the dictation transcript into a strict SOAP note JSON object.

SOAP rules:
- Subjective: only patient-reported symptoms/history/complaints
- Objective: only measurable or observed findings (vitals, labs, physical exam)
- Assessment: diagnosis and clinical reasoning
- Plan: treatment, medications, follow-up

Hard constraints:
- Never mix categories.
- Symptoms must never be in Objective.
- Vitals/labs/measurements must never be in Subjective.
- If information for a section is missing, set it to "Not Available".
- Output must be valid JSON only. No markdown, no commentary.
- JSON schema:
{{
  "Subjective": "...",
  "Objective": "...",
  "Assessment": "...",
  "Plan": "..."
}}

Transcript:
{transcript}
""".strip()


def _extract_json_object(raw_text: str) -> Dict[str, str]:
    """Extract and parse the first JSON object from model output."""
    raw_text = raw_text.strip()

    if raw_text.startswith("{") and raw_text.endswith("}"):
        return json.loads(raw_text)

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise SOAPGenerationError("No JSON object found in model response.")

    return json.loads(match.group(0))


def _normalize_soap(soap_data: Dict[str, str]) -> Dict[str, str]:
    normalized = {}
    for key in SOAP_KEYS:
        value = soap_data.get(key)
        if value is None or not str(value).strip():
            normalized[key] = "Not Available"
        else:
            normalized[key] = str(value).strip()
    return normalized


def generate_soap(transcript: str, llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct") -> Dict[str, str]:
    """Generate SOAP JSON from transcript using an open-source instruct model."""
    if not transcript.strip():
        raise SOAPGenerationError("Transcript is empty.")

    prompt = _build_soap_prompt(transcript)

    try:
        generator = pipeline(
            task="text-generation",
            model=llm_model_name,
            device_map="auto",
        )

        output = generator(
            prompt,
            max_new_tokens=400,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )

        raw_response = output[0]["generated_text"]
        soap_data = _extract_json_object(raw_response)
        return _normalize_soap(soap_data)

    except SOAPGenerationError:
        raise
    except Exception as exc:
        raise SOAPGenerationError(f"SOAP generation failed: {exc}") from exc


def validate_soap(soap_note: Dict[str, str]) -> ValidationResult:
    """Rule-based validation to catch common section misclassification."""
    subjective = soap_note.get("Subjective", "")
    objective = soap_note.get("Objective", "")

    symptom_terms = [
        "pain",
        "nausea",
        "dizziness",
        "fatigue",
        "headache",
        "shortness of breath",
        "cough",
        "fever",
        "vomiting",
    ]

    vital_patterns = [
        r"\bBP\b",
        r"\bblood pressure\b",
        r"\bHR\b",
        r"\bheart rate\b",
        r"\btemperature\b",
        r"\btemp\b",
        r"\brespiratory rate\b",
        r"\bRR\b",
        r"\bSpO2\b",
        r"\boxygen saturation\b",
        r"\b\d{2,3}/\d{2,3}\b",  # blood pressure pattern
    ]

    issues: List[str] = []

    lowered_objective = objective.lower()
    for term in symptom_terms:
        if term in lowered_objective:
            issues.append(f"Potential misclassification: symptom '{term}' found in Objective.")

    for pattern in vital_patterns:
        if re.search(pattern, subjective, flags=re.IGNORECASE):
            issues.append(
                f"Potential misclassification: vital/sign pattern '{pattern}' found in Subjective."
            )

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)


def run_pipeline(
    audio_path: str,
    whisper_model_size: str,
    llm_model_name: str,
    device: str,
    compute_type: str,
) -> Dict[str, object]:
    transcript = transcribe_audio(
        audio_path=audio_path,
        whisper_model_size=whisper_model_size,
        device=device,
        compute_type=compute_type,
    )

    soap_note = generate_soap(transcript=transcript, llm_model_name=llm_model_name)
    validation = validate_soap(soap_note)

    return {
        "transcript": transcript,
        "soap_note": soap_note,
        "validation": {
            "is_valid": validation.is_valid,
            "issues": validation.issues,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert medical dictation audio into validated SOAP notes."
    )
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--whisper-model", default="base", help="faster-whisper model size")
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace instruct model for SOAP generation",
    )
    parser.add_argument("--device", default="cpu", help="Device for whisper model, e.g. cpu/cuda")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type, e.g. int8/float16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = run_pipeline(
            audio_path=args.audio_path,
            whisper_model_size=args.whisper_model,
            llm_model_name=args.llm_model,
            device=args.device,
            compute_type=args.compute_type,
        )
        print(json.dumps(result, indent=2))
    except PipelineError as exc:
        error_payload = {
            "error": str(exc),
            "soap_note": {key: "Not Available" for key in SOAP_KEYS},
            "validation": {"is_valid": False, "issues": [str(exc)]},
        }
        print(json.dumps(error_payload, indent=2))


if __name__ == "__main__":
    main()
