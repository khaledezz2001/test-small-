"""
RunPod serverless handler — document classification / summary / signatures.

This applies the exact prompt flow from LLMPrompts.txt
(process_with_llm + classify_text_with_LLM) to the vLLM serverless
structure of handler.py:

    Pass 1 (classification): text[:max_length_classification]
    Pass 2 (summary):        text[:max_length]
    Pass 3 (signatures):     text[-max_length:]

Each pass sends its dedicated prompt + JSON schema to the model.
The response is parsed into a dict; missing required keys are filled
with schema defaults (string -> "", number/integer -> 0, boolean -> False).

Input : {"pages": [{"page": 1, "text": "..."}, ...]}
Output: {"classification": {...}, "summary": {...}, "signatures": {...}}
"""

import os
import re
import json

import runpod
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ===============================
# MODEL CONFIG (loaded inside __main__ guard)
# ===============================
MODEL_PATH = "/app/models/Qwen3-8B"

tokenizer = None
llm = None


# ===============================
# HELPERS (same as handler.py)
# ===============================
def combine_pages(pages):
    sorted_pages = sorted(pages, key=lambda p: p["page"])
    return "\n\n".join(p["text"] for p in sorted_pages)


def strip_thinking(text):
    """Remove <think>...</think> blocks from model output (safety net)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


# ===============================
# SCHEMAS (identical to LLMPrompts.txt)
# ===============================
def build_classification_schema():
    schema = {
        "type": "object",
        "properties": {
            "classification": {"type": "string"},
            "subclassification": {"type": "string"},
            "confidence": {"type": "number"},
            "title": {"type": "string"},
        },
    }
    schema["required"] = list(schema["properties"].keys())
    return schema


def build_summary_schema():
    schema = {
        "type": "object",
        "properties": {
            "jurisdiction": {"type": "string"},
            "languagePrimary": {"type": "string"},
            "languageSecondary": {"type": "string"},
            "summaryShort": {"type": "string"},
            "summary": {"type": "string"},
            "summaryDetailed": {"type": "string"},
            "summaryDetailedTranslated": {"type": "string"},
            "hasTranslation": {"type": "boolean"},
            "errorsDetected": {"type": "string"},
            "documentDate": {"type": "string"},
            "entitiesMentioned": {"type": "string"},
            "peopleMentioned": {"type": "string"},
            "relatedDocuments": {"type": "string"},
            "documentValidFrom": {"type": "string"},
            "documentValidUntil": {"type": "string"},
            "keywords": {"type": "string"},
            "categorizationKeywords": {"type": "string"},
            "translationQuality": {"type": "number"},
            "languageConsistency": {"type": "number"},
            "headerFieldsDetected": {"type": "string"},
            "footerFieldsDetected": {"type": "string"},
            "documentThemeKeywords": {"type": "string"},
            "documentIsComplete": {"type": "boolean"},
        },
    }
    schema["required"] = list(schema["properties"].keys())
    return schema


def build_signatures_schema():
    schema = {
        "type": "object",
        "properties": {
            "hasApostille": {"type": "boolean"},
            "hasNotarization": {"type": "boolean"},
            "isFullySigned": {"type": "boolean"},
            "isPartiallySigned": {"type": "boolean"},
            "signatureType": {"type": "string"},
            "signatures": {"type": "string"},
            "witnesses": {"type": "string"},
            "handwrittenElementsPresent": {"type": "boolean"},
            "handwrittenElements": {"type": "string"},
            "sealStampDetected": {"type": "boolean"},
        },
    }
    schema["required"] = list(schema["properties"].keys())
    return schema


SUBCLASSIFICATION_LIST = [
    "Certificate of Incorporation",
    "Share Register/Share Certificates",
    "Board/Shareholder Resolutions",
    "Power Of Attorney (POA)",
    "Bank Statement",
    "Financial Statements",
    "Invoice",
    "Balance Sheet",
    "Payment Advice",
    "Agreements",
    "Tax Authority Registration",
    "VAT Registration Certificates",
    "Curriculum Vitae/Biography/Resume",
    "Passport/ID",
    "Employment Contract",
    "Dimplomas/Certifications",
    "Employment Termination Agreements/Letters",
    "Employee Evaluation Reports",
    "Travel Document/Boarding Pass",
    "Calendar",
    "Unspecified",
]


# ===============================
# PROMPTS (exact texts from LLMPrompts.txt)
# Placeholders: __TEXT__, __SUBCLASS_LIST__  (.replace is used instead of
# .format because the embedded example JSON contains literal braces)
# ===============================
PROMPT_CLASSIFICATION = """

    Process the text (__TEXT__) and return a representative JSON object with the following properties and data types (information and/or instructions about each property is provided after the -> symbol):
        - "subclassification": string (double-quoted) -> from __SUBCLASS_LIST__, select a label that best describes the given text. If the label you selected does not exist in __SUBCLASS_LIST__, return "Unspecified".
        - "classification": string (double-quoted) -> return "-".
        - "confidence": integer (unquoted) -> %, degree of confidence that both the classification and subclassication assigned are correct. Return 0 if both the assigned classification and subclassification are "Unspecified".
        - "title": string (double-quoted, 250 characters max) -> extract only the main title or heading of the document - typically located at the top of the first page. Use the original language and exact spelling from the document. Do not translate, transliterate, correct, or interpret the text. Preserve accents, punctuation, and capitalization. If multiple language versions of the title are present, return them exactly as written, separated by a slash (/). If the document has no clear title, return "Untitled". Do not extract running headers, file names, or clause titles. Only the official heading of the document.

    Make sure to return a valid JSON object using double quotes for all property names and string values, adhering to the JSON standard (RFC 8259).
    Always respect the max character limit for a property if this has been specified.
    Make sure to not return nested properties.
    Ensure numbers, booleans, and null values are unquoted.
    Format the response as a code block with only the JSON object, no additional text or explanations.
    Never use single quotes (') around property names and values.

    Always preserve the document's original language for all names and titles. Do not transliterate or translate.
    Do not infer missing data or assume that similar names in different languages refer to the same person.
    If the same entity appears multiple times in different formats or languages, list each mention separately.
    Ignore header/footer content unless it introduces new, relevant parties.
    In bilingual documents, extract each version exactly as written and treat them as distinct entries if they differ.

    Example:
    {
        "subclassification": "Agreements",
        "classification": "-",
        "confidence": 98,
        "title": "Agreement Between B Corporation and John"
    }

    """

PROMPT_SUMMARY = """

    Process the text (__TEXT__) and return a representative JSON object with the following properties and data types (information and/or instructions about each property is provided after the -> symbol):
        - "jurisdiction": string (double-quoted) -> a jurisdiction if present
        - "languagePrimary": string (double-quoted, 50 characters max) -> the primary language of the document, i.e. the language of the main body of the document. Repetitive labels, stamps, letterheads or annex headings in another language must NOT outweigh the main text.
        - "languageSecondary": string (double-quoted, 50 characters max) -> the secondary language if present
        - "summaryShort": string (double-quoted, 100 characters max) -> a summary of the document up to 100 characters. Ensure that the returned text does not exceed the 100 character limit.
        - "summary": string (double-quoted, 300 characters max) -> a summary of the document up to 300 characters. Ensure that the returned text does not exceed the 300 character limit.
        - "summaryDetailed": string (double-quoted, 700 characters max) -> a summary of the document IN THE DOCUMENT'S ORIGINAL LANGUAGE (i.e. the language identified as "languagePrimary"), up to 700 characters. Ensure that the returned text does not exceed the 700 character limit. The English version belongs in "summaryDetailedTranslated" only.
        - "summaryDetailedTranslated": string (double-quoted, 700 characters max) -> the English translation OF the "summaryDetailed" field - it must convey the same content as "summaryDetailed", up to 700 characters. Ensure that the returned text does not exceed the 700 character limit.
        - "hasTranslation": boolean (unquoted, true or false) -> whether a full translation of the document's main content is attached or embedded. Bilingual labels, letterheads, plans or annex headings alone do NOT count as a translation.
        - "errorsDetected": string (double-quoted) -> typos, inconsistencies, OCR issues or anomalies (e.g., duplicated clause numbers, missing clause numbers, truncated text, broken numbering)
        - "documentDate": date (double-quoted, in YYYY-MM-DD format) -> date mentioned as the official issuance/signing date in the document. Return '1900-01-01' if not stated/unspecified
        - "entitiesMentioned": string (double-quoted) -> companies or legal entities explicitly mentioned in the document. Extract names exactly as written in the document, preserving the original language and script. Where available, include roles such as buyer, seller, affiliate, counterparty - but only if clearly defined in the document. Avoid inferred roles or translated names. Do not deduplicate similar-looking names across languages. CRITICAL: return this field in the document's original language and script ONLY - never translate or transliterate (e.g. return «Дарбшир», NOT "Darbshir"). If the document contains a named list of companies (e.g. client companies, group members, counterparties), include EVERY entry from that list.
        - "peopleMentioned": string (double-quoted) -> individuals explicitly named in the document (e.g., directors, signatories, shareholders, legal representatives). Extract personal names exactly as written, preserving original language, spelling, and formatting. If the document clearly assigns a role or title, include it alongside the name. Do not infer missing roles or identities. Avoid merging bilingual versions of the same name unless they appear in the same clause. CRITICAL: return names in the document's original language and script ONLY - never transliterate into Latin (e.g. return Пожитков Андрей Игоревич, NOT "Pozhitkov Andrey Igorevich"). NEVER expand initials into full names - if the document says "Шатрова Ю.И.", return exactly "Шатрова Ю.И.", not an invented full name. Do NOT skip people who appear only with initials - include them exactly as written, with their role if assigned (e.g. "Шатрова Ю.И. (генеральный директор)"). Also include individuals from appendices and annexes (e.g. signatories and authorized representatives of counterparties). Include the role in parentheses where clearly assigned (e.g. "Заварина Анастасия Сергеевна (генеральный директор)").
        - "relatedDocuments": string (double-quoted) -> extract references to other documents mentioned within the current document. These may include appendices, exhibits, annexes, translations, resolutions, contracts, certificates, or attachments. Only include references that are explicitly named, numbered, or otherwise clearly linked - either in the body of the text or as labeled sections. List them in the order of appearance using the exact document titles or references, preserving the original language. If no related documents are mentioned, leave this field empty. Do not include generic mentions (e.g., "see above", "as per the annex") unless a specific document is clearly referenced. CRITICAL: keep the exact original titles in the document's language - never translate them (e.g. return Приложение 1 – Договор займа, NOT "Appendix 1 - Loan Agreement").
        - "documentValidFrom": date (double-quoted, in YYYY-MM-DD format) -> effective start date if stated. Return '1900-01-01' if not stated/unspecified
        - "documentValidUntil": date (double-quoted, in YYYY-MM-DD format) -> expiration date if applicable. Return '1900-01-01' if not stated/unspecified
        - "keywords": string (double-quoted) -> key terms or clauses identified automatically (e.g., 'termination', 'governing law')
        - "categorizationKeywords": string (double-quoted) -> extract a list of keywords that describe the core subject matter, legal or business context, and functional purpose of the document. These keywords will be used to categorize, tag, and search for documents by content - not for clause indexing.
        - "translationQuality": integer (unquoted) -> %, degree to which embedded/attached translations are accurate
        - "languageConsistency": integer (unquoted) -> %, degree to which text in multiple languages is consistent and aligned
        - "headerFieldsDetected": string (double-quoted) -> detected fields in top page headers (e.g., 'CONFIDENTIAL', company name, logo)
        - "footerFieldsDetected": string (double-quoted) -> footer content: page numbers, disclaimers, versioning
        - "documentThemeKeywords": string (double-quoted) -> context tags: e.g. 'employment', 'finance', 'real estate'
        - "documentIsComplete": boolean (unquoted, true or false)  -> whether document appears complete vs. truncated/scanned partially

    Make sure to return a valid JSON object using double quotes for all property names and string values, adhering to the JSON standard (RFC 8259).
    Always respect the max character limit for a property if this has been specified.
    Make sure to not return nested properties.
    Ensure numbers, booleans, and null values are unquoted.
    Format the response as a code block with only the JSON object, no additional text or explanations.
    Never use single quotes (') around property names and values.

    Always preserve the document's original language for all names and titles. Do not transliterate or translate.
    Do not infer missing data or assume that similar names in different languages refer to the same person.
    If the same entity appears multiple times in different formats or languages, list each mention separately.
    Ignore header/footer content unless it introduces new, relevant parties.
    In bilingual documents, extract each version exactly as written and treat them as distinct entries if they differ.

    Example:
    {
        "jurisdiction": "United Kingdom",
        "languagePrimary": "English",
        "languageSecondary": "Russian",
        "summaryShort": "An agreement of cooperation Between B Corporation and John",
        "summary": "An agreement of cooperation Between B Corporation and John lasting for 2 years",
        "summaryDetailed": "An agreement of cooperation Between B Corporation and John lasting for 2 years between 01/01/2025 and 01/01/2027",
        "summaryDetailedTranslated": "An agreement of cooperation Between B Corporation and John lasting for 2 years between 01/01/2025 and 01/01/2027",
        "hasTranslation": true,
        "errorsDetected": "",
        "documentDate": "2024-12-26",
        "entitiesMentioned": "",
        "peopleMentioned": "",
        "relatedDocuments": "",
        "documentValidFrom": "2025-01-01",
        "documentValidUntil": "2027-01-01",
        "keywords": "Agreement, Parties",
        "categorizationKeywords": "Agreement, Parties",
        "translationQuality": 100,
        "languageConsistency": 100,
        "headerFieldsDetected": "",
        "footerFieldsDetected": "",
        "documentThemeKeywords": "Agreement, Parties",
        "documentIsComplete": true
    }

    """

PROMPT_SIGNATURES = """

    Process the text (__TEXT__) and return a representative JSON object with the following properties and data types (information and/or instructions about each property is provided after the -> symbol):
        - "hasApostille": boolean (unquoted, true or false) -> whether the document is apostilled
        - "hasNotarization": boolean (unquoted, true or false) -> whether the document is notarized
        - "isFullySigned": boolean (unquoted, true or false) -> whether the document is fully signed
        - "isPartiallySigned": boolean (unquoted, true or false) -> whether only some pages are signed
        - "signatureType": string (double-quoted) -> type of signature: Handwritten/Digital/None
        - "signatures": string (double-quoted) -> list of signatories with name/role/page reference in the format "NAME; ROLE" (e.g. "NICOS MICHAELAS; LANDLORD"). Keep in the original language ALWAYS - never transliterate names into Latin script
        - "witnesses": string (double-quoted) -> list of witnesses if applicable. Keep in the original language ALWAYS - never transliterate names into Latin script
        - "handwrittenElementsPresent": boolean (unquoted, true or false) -> whether handwritten content is found
        - "handwrittenElements": string (double-quoted) -> details of handwritten parts (e.g., names, dates, annotations). Keep in original language where applicable
        - "sealStampDetected": boolean (unquoted, true or false) -> boolean flag for the existence of official stamps, seals, logos

    Make sure to return a valid JSON object using double quotes for all property names and string values, adhering to the JSON standard (RFC 8259).
    Always respect the max character limit for a property if this has been specified.
    Make sure to not return nested properties.
    Ensure numbers, booleans, and null values are unquoted.
    Format the response as a code block with only the JSON object, no additional text or explanations.
    Never use single quotes (') around property names and values.

    Always preserve the document's original language for all names and titles. Do not transliterate or translate.
    Do not infer missing data or assume that similar names in different languages refer to the same person.
    If the same entity appears multiple times in different formats or languages, list each mention separately.
    Ignore header/footer content unless it introduces new, relevant parties.
    In bilingual documents, extract each version exactly as written and treat them as distinct entries if they differ.

    Example:
    {
        "hasApostille": false,
        "hasNotarization": true,
        "isFullySigned": true,
        "isPartiallySigned": false,
        "signatureType": "Handwritten",
        "signatures": "George; Director of B Corporation, John",
        "witnesses": "",
        "handwrittenElementsPresent": false,
        "handwrittenElements": "",
        "sealStampDetected": false
    }

    """


# ===============================
# JSON PARSING + SCHEMA DEFAULTS
# ===============================
DEFAULTS_BY_TYPE = {"string": "", "number": 0, "integer": 0, "boolean": False}


def coerce_value(value, type_):
    """Coerce a model-provided value to the schema-declared JSON type."""
    if type_ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
        return bool(value)
    if type_ in ("number", "integer"):
        try:
            return int(value) if type_ == "integer" else float(value)
        except (TypeError, ValueError):
            return DEFAULTS_BY_TYPE[type_]
    # string
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def fill_schema_defaults(data, schema):
    """Ensure every required property exists with the correct JSON type.

    Guarantees a stable output shape even if the model omits fields or
    returns invalid JSON (then defaults are used for everything).
    """
    if not isinstance(data, dict):
        data = {}
    props = schema["properties"]
    required = schema.get("required", list(props.keys()))
    return {
        key: coerce_value(data.get(key), props.get(key, {}).get("type", "string"))
        for key in required
    }


# Max character limits stated inside the prompts (enforced server-side)
CHAR_LIMITS = {
    "title": 250,
    "languagePrimary": 50,
    "languageSecondary": 50,
    "summaryShort": 100,
    "summary": 300,
    "summaryDetailed": 700,
    "summaryDetailedTranslated": 700,
}


def enforce_char_limits(result):
    for key, limit in CHAR_LIMITS.items():
        value = result.get(key)
        if isinstance(value, str) and len(value) > limit:
            result[key] = value[:limit]
    return result


# ===============================
# CORE: process_with_llm
# (same contract as LLMPrompts.txt, but the JSON is actually parsed —
#  the original returned the regex match object instead of a dict)
# ===============================
def process_with_llm(prompt, schema, max_tokens=2000):
    """Send one prompt to the model and return the parsed JSON dict."""

    params = SamplingParams(
        temperature=0.1,        # same as the original OpenAI call
        max_tokens=max_tokens,  # same default: 2000
        repetition_penalty=1.1,
    )

    messages = [
        {"role": "system", "content": "You are a professional document analyst."},
        {"role": "user", "content": f"{prompt}. Use {json.dumps(schema)} for output."},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    outputs = llm.generate([prompt_str], params)
    raw = strip_thinking(outputs[0].outputs[0].text.strip())

    try:
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] process_with_llm: JSON parse failed: {e}", flush=True)
        print(f"[WARN] Raw output was: {raw[:500]}", flush=True)
        return {}


# ===============================
# MAIN PIPELINE: classify_text_with_llm
# (identical slicing logic to LLMPrompts.txt)
# ===============================
def classify_text_with_llm(text, max_length=60000, max_length_classification=8000):

    # --- slicing: exactly as in the original function ---
    text_classification = text[:max_length_classification]
    end_text = text[-max_length:] if len(text) > max_length else text
    body = text[:max_length]

    schema_classification = build_classification_schema()
    schema_summary = build_summary_schema()
    schema_signatures = build_signatures_schema()

    # --- pass 1: classification (start of document) ---
    prompt_classification = (
        PROMPT_CLASSIFICATION
        .replace("__TEXT__", text_classification)
        .replace("__SUBCLASS_LIST__", str(SUBCLASSIFICATION_LIST))
    )
    print(f"[LOG] Pass 1/3: classification ({len(text_classification)} chars)", flush=True)
    result_classification = process_with_llm(prompt_classification, schema_classification)

    # --- pass 2: summary (body of document) ---
    prompt_summary = PROMPT_SUMMARY.replace("__TEXT__", body)
    print(f"[LOG] Pass 2/3: summary ({len(body)} chars)", flush=True)
    result_summary = process_with_llm(prompt_summary, schema_summary)

    # --- pass 3: signatures (end of document) ---
    prompt_signatures = PROMPT_SIGNATURES.replace("__TEXT__", end_text)
    print(f"[LOG] Pass 3/3: signatures ({len(end_text)} chars)", flush=True)
    result_signatures = process_with_llm(prompt_signatures, schema_signatures)

    # --- enforce schema shape, types and char limits ---
    result_classification = enforce_char_limits(fill_schema_defaults(result_classification, schema_classification))
    result_summary = enforce_char_limits(fill_schema_defaults(result_summary, schema_summary))
    result_signatures = fill_schema_defaults(result_signatures, schema_signatures)

    # --- dedup comma-separated name fields (model sometimes repeats entries) ---
    for field in ("entitiesMentioned", "peopleMentioned", "relatedDocuments"):
        result_summary[field] = dedup_name_field(result_summary.get(field, ""))
    for field in ("signatures", "witnesses"):
        result_signatures[field] = dedup_name_field(result_signatures.get(field, ""))

    results = {
        "classification": result_classification,
        "summary": result_summary,
        "signatures": result_signatures,
    }
    warn_if_transliterated(results, text)
    return results


# ===============================
# TRANSLITERATION SAFETY NET
# ===============================
WORD_TOKEN = re.compile(r"[\w\-\.]+", re.UNICODE)


def warn_if_transliterated(results, full_text):
    """Log a warning when name fields contain tokens that do not appear
    anywhere in the source text. Catches two failure modes:
    - transliteration: 'Darbshir' in the output but only «Дарбшир» in the doc
    - invented data: 'Юлия' expanded from initials 'Ю.' in 'Шатрова Ю.И.'
    Short tokens (initials like 'Ю.', role words) are skipped to limit
    false positives from inflected forms.
    """
    text_lower = full_text.lower()
    fields = [
        ("summary.entitiesMentioned", results["summary"].get("entitiesMentioned", "")),
        ("summary.peopleMentioned", results["summary"].get("peopleMentioned", "")),
        ("summary.relatedDocuments", results["summary"].get("relatedDocuments", "")),
        ("signatures.signatures", results["signatures"].get("signatures", "")),
        ("signatures.witnesses", results["signatures"].get("witnesses", "")),
    ]
    for name, value in fields:
        if not isinstance(value, str):
            continue
        for match in WORD_TOKEN.finditer(value):
            token = match.group().strip(".-")
            if len(token) >= 5 and token.lower() not in text_lower:
                print(f"[WARN] {name}: token '{token}' not found in source text "
                      f"(possible transliteration or invented data)", flush=True)


def dedup_name_field(value):
    """Remove exact duplicate entries from a comma-separated name field,
    preserving order (case-insensitive). E.g. 'Сурова Е.Б., Пожитков А.И.,
    Сурова Е.Б.' -> 'Сурова Е.Б., Пожитков А.И.'.
    """
    if not isinstance(value, str) or "," not in value:
        return value
    seen, out = set(), []
    for item in value.split(","):
        item = item.strip()
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return ", ".join(out)


# ===============================
# RUNPOD HANDLER
# ===============================
def handler(event):
    try:
        pages = event["input"]["pages"]
        if not pages or not isinstance(pages, list):
            return {"error": "'pages' must be a non-empty list"}
        for p in pages:
            if "page" not in p or "text" not in p:
                return {"error": "Each page needs 'page' and 'text' fields"}

        max_length = int(event["input"].get("max_length", 60000))
        max_length_classification = int(event["input"].get("max_length_classification", 8000))

        full_text = combine_pages(pages)
        if not full_text.strip():
            return {"error": "Document text is empty"}

        print(f"[LOG] Received request with {len(pages)} pages, "
              f"{len(full_text)} chars", flush=True)

        return classify_text_with_llm(
            full_text,
            max_length=max_length,
            max_length_classification=max_length_classification,
        )
    except KeyError as e:
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}", flush=True)
        return {"error": str(e)}


if __name__ == '__main__':
    # ===============================
    # LOAD MODEL WITH vLLM
    # (inside __main__ guard so vLLM's spawned child processes
    #  don't re-run initialization)
    # ===============================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    llm = LLM(
        model=MODEL_PATH,
        dtype="float16",
        max_model_len=16384,         # long docs need more context room
        tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,  # reuse KV cache for the shared system prompt
    )

    print("[LOG] Qwen3-8B loaded via vLLM (prefix caching ON)", flush=True)

    runpod.serverless.start({"handler": handler})