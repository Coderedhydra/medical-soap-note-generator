import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


SOAP_KEYS = ["subjective", "objective", "assessment", "plan"]
MODEL_TO_OUTPUT_KEY = {
    "Subjective": "subjective",
    "Objective": "objective",
    "Assessment": "assessment",
    "Plan": "plan",
}


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
        from faster_whisper import WhisperModel

        model = WhisperModel(whisper_model_size, device=device, compute_type=compute_type)
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc

    cleaned = re.sub(r"\s+", " ", transcript).strip()
    if not cleaned:
        raise TranscriptionError("No speech detected in the audio.")

    return cleaned


def _build_soap_prompt(transcript: str) -> str:
    return f"""
You are a clinical documentation assistant.
Convert the dictation transcript into a strict SOAP note JSON object.

SOAP definition:
- Subjective: ONLY patient-reported symptoms/history/complaints
- Objective: ONLY measurable or observed findings (vitals/labs/physical exam)
- Assessment: diagnosis and clinical reasoning
- Plan: treatment, medication, follow-up

Hard constraints:
- Do not mix information between sections.
- Symptoms must not appear in Objective.
- Vitals/labs/measurements must not appear in Subjective.
- If section data is missing, set it to "Not Available".
- Return valid JSON only. No prose. No markdown. No explanation.

Required JSON schema (exact keys):
{{
  "Subjective": "string",
  "Objective": "string",
  "Assessment": "string",
  "Plan": "string"
}}

Transcript:
{transcript}
""".strip()


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from model output."""
    normalized = raw_text.strip().replace("```json", "").replace("```", "").strip()

    if normalized.startswith("{") and normalized.endswith("}"):
        return json.loads(normalized)

    match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
    if not match:
        raise SOAPGenerationError("No JSON object found in model response.")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise SOAPGenerationError(f"Model JSON parsing failed: {exc}") from exc


def _normalize_soap(raw_soap: Dict[str, Any]) -> Dict[str, str]:
    """Normalize model output to lowercase SOAP keys with fallback values."""
    normalized: Dict[str, str] = {key: "Not Available" for key in SOAP_KEYS}

    for model_key, output_key in MODEL_TO_OUTPUT_KEY.items():
        value = raw_soap.get(model_key)
        if value is not None and str(value).strip():
            normalized[output_key] = str(value).strip()

    for key in SOAP_KEYS:
        value = raw_soap.get(key)
        if value is not None and str(value).strip():
            normalized[key] = str(value).strip()

    return normalized


def _fallback_soap_from_transcript(transcript: str) -> Dict[str, str]:
    """Heuristic fallback SOAP extraction when LLM output is not valid JSON."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", transcript) if s.strip()]

    vital_regex = re.compile(
        r"\b(BP|blood pressure|HR|heart rate|RR|respiratory rate|SpO2|oxygen saturation|temp|temperature|\d{2,3}/\d{2,3})\b",
        flags=re.IGNORECASE,
    )
    symptom_regex = re.compile(
        r"\b(pain|nausea|dizziness|fatigue|headache|shortness of breath|cough|fever|vomiting)\b",
        flags=re.IGNORECASE,
    )
    assessment_regex = re.compile(
        r"\b(assessment|impression|diagnosis|likely|possible|consistent with|suspect)\b",
        flags=re.IGNORECASE,
    )
    plan_regex = re.compile(
        r"\b(plan|start|prescribe|medication|follow[- ]?up|return|order|refer|advise|recommend)\b",
        flags=re.IGNORECASE,
    )

    buckets: Dict[str, List[str]] = {key: [] for key in SOAP_KEYS}
    for sentence in sentences:
        if vital_regex.search(sentence):
            buckets["objective"].append(sentence)
        elif assessment_regex.search(sentence):
            buckets["assessment"].append(sentence)
        elif plan_regex.search(sentence):
            buckets["plan"].append(sentence)
        elif symptom_regex.search(sentence):
            buckets["subjective"].append(sentence)
        else:
            # Default to subjective for neutral patient narrative.
            buckets["subjective"].append(sentence)

    soap = {}
    for key in SOAP_KEYS:
        soap[key] = " ".join(buckets[key]).strip() if buckets[key] else "Not Available"

    return soap


def _load_generator(llm_model_name: str):
    from transformers import AutoConfig, pipeline

    config = AutoConfig.from_pretrained(llm_model_name)
    task = "text2text-generation" if getattr(config, "is_encoder_decoder", False) else "text-generation"
    return pipeline(task=task, model=llm_model_name, device_map="auto")


def _generate_raw_response(prompt: str, llm_model_name: str) -> str:
    generator = _load_generator(llm_model_name)
    output = generator(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False,
    )
    return output[0]["generated_text"]


def generate_soap(
    transcript: str,
    llm_model_name: str = "google/flan-t5-base",
) -> Tuple[Dict[str, str], str]:
    """
    Generate SOAP JSON from transcript.
    Returns (soap_note, generation_mode) where mode is 'llm' or 'fallback'.
    """
    if not transcript.strip():
        raise SOAPGenerationError("Transcript is empty.")

    prompt = _build_soap_prompt(transcript)

    try:
        raw_response = _generate_raw_response(prompt, llm_model_name)
        soap_data = _extract_json_object(raw_response)
        return _normalize_soap(soap_data), "llm"
    except Exception:
        return _fallback_soap_from_transcript(transcript), "fallback"


def validate_soap(soap_note: Dict[str, str]) -> ValidationResult:
    """Rule-based validation to catch common section misclassification."""
    subjective = soap_note.get("subjective", "")
    objective = soap_note.get("objective", "")

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
        r"\b\d{2,3}/\d{2,3}\b",
    ]

    issues: List[str] = []

    lowered_objective = objective.lower()
    for term in symptom_terms:
        if term in lowered_objective:
            issues.append(f"Potential misclassification: symptom '{term}' found in objective.")

    for pattern in vital_patterns:
        if re.search(pattern, subjective, flags=re.IGNORECASE):
            issues.append(
                f"Potential misclassification: vital/sign pattern '{pattern}' found in subjective."
            )

    return ValidationResult(is_valid=len(issues) == 0, issues=issues)


def run_pipeline(
    audio_path: str,
    whisper_model_size: str,
    llm_model_name: str,
    device: str,
    compute_type: str,
) -> Dict[str, Any]:
    transcript = transcribe_audio(
        audio_path=audio_path,
        whisper_model_size=whisper_model_size,
        device=device,
        compute_type=compute_type,
    )
    soap_note, generation_mode = generate_soap(transcript=transcript, llm_model_name=llm_model_name)
    validation = validate_soap(soap_note)

    return {
        "transcript": transcript,
        "soap_note": soap_note,
        "generation_mode": generation_mode,
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
        default="google/flan-t5-base",
        help="HuggingFace model for SOAP generation (smaller default to reduce downloads)",
    )
    parser.add_argument("--device", default="cpu", help="Device for faster-whisper: cpu/cuda")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type: int8/float16/etc.",
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
        print(json.dumps(result, ensure_ascii=False))
    except PipelineError as exc:
        error_payload = {
            "error": str(exc),
            "transcript": "Not Available",
            "soap_note": {key: "Not Available" for key in SOAP_KEYS},
            "generation_mode": "error",
            "validation": {"is_valid": False, "issues": [str(exc)]},
        }
        print(json.dumps(error_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
