from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from open_research.core.config import Settings
from open_research.core.domain import AssetExtractionMethod, AssetProcessingStatus

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".log"}
_JSON_EXTENSIONS = {".json", ".jsonl"}
_CSV_EXTENSIONS = {".csv", ".tsv"}
_HTML_EXTENSIONS = {".html", ".htm"}
_DOCX_EXTENSIONS = {".docx"}
_PDF_EXTENSIONS = {".pdf"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self._parts)


@dataclass(slots=True)
class ExtractedAsset:
    processing_status: AssetProcessingStatus
    extraction_method: AssetExtractionMethod
    extracted_text: str | None
    file_size_bytes: int
    sha256: str
    preview_excerpt: str | None
    content_type: str | None
    file_name: str
    warnings: list[str] = field(default_factory=list)
    ocr_used: bool = False
    page_count: int | None = None
    processing_error: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "processing_status": self.processing_status.value,
            "extraction_method": self.extraction_method.value,
            "ocr_used": self.ocr_used,
            "page_count": self.page_count,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "warnings": self.warnings,
            "preview_excerpt": self.preview_excerpt,
            "processing_error": self.processing_error,
            "extracted_text": self.extracted_text,
        }


def _preview_excerpt(text: str | None, *, limit: int = 280) -> str | None:
    if not text:
        return None
    collapsed = " ".join(text.split())
    return collapsed[:limit] if len(collapsed) > limit else collapsed


def _decode_text(data: bytes) -> str:
    errors: list[Exception] = []
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:  # pragma: no cover - narrow branch
            errors.append(exc)
    raise ValueError("Could not decode text content.") from (errors[-1] if errors else None)


def _extract_json(data: bytes) -> str:
    payload = json.loads(_decode_text(data))
    return json.dumps(payload, indent=2, sort_keys=True)


def _extract_csv(data: bytes, *, delimiter: str = ",") -> str:
    raw = _decode_text(data)
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows = ["\t".join(cell.strip() for cell in row) for row in reader]
    return "\n".join(row for row in rows if row.strip())


def _extract_html(data: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(_decode_text(data))
    return parser.text()


def _extract_pdf(
    data: bytes,
    *,
    max_ocr_pages: int,
    min_text_chars: int,
) -> tuple[str, AssetExtractionMethod, bool, int, list[str]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError("PyMuPDF is required to process PDF uploads.") from exc

    warnings: list[str] = []
    document = fitz.open(stream=data, filetype="pdf")
    page_count = document.page_count
    text_parts = [page.get_text("text").strip() for page in document]
    text = "\n\n".join(part for part in text_parts if part)
    if len(text.strip()) >= min_text_chars:
        return text, AssetExtractionMethod.PDF_TEXT, False, page_count, warnings
    if page_count > max_ocr_pages:
        raise RuntimeError(
            f"PDF requires OCR but has {page_count} pages; the OCR limit is {max_ocr_pages}."
        )
    ocr_text = _ocr_pdf_document(document)
    warnings.append("OCR fallback used because PDF text extraction was insufficient.")
    return ocr_text, AssetExtractionMethod.PDF_OCR, True, page_count, warnings


def _ocr_pdf_document(document) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError(
            "PDF OCR requires Pillow and pytesseract, and the Tesseract binary must be installed."
        ) from exc

    parts: list[str] = []
    for page in document:
        pixmap = page.get_pixmap()
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image).strip()
        if text:
            parts.append(text)
    if not parts:
        raise RuntimeError("OCR could not extract text from the PDF.")
    return "\n\n".join(parts)


def _extract_docx(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
        temp_file.write(data)
        temp_file.flush()
        try:
            import docx2txt  # type: ignore

            text = docx2txt.process(temp_file.name)
            if text and text.strip():
                return text
        except ImportError:
            pass
        try:
            from docx import Document  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency-dependent
            raise RuntimeError(
                "DOCX extraction requires docx2txt or python-docx to be installed."
            ) from exc
        document = Document(temp_file.name)
        paragraphs = [
            paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()
        ]
        text = "\n".join(paragraphs)
        if not text.strip():
            raise RuntimeError("No readable text was found in the DOCX upload.")
        return text


def _extract_image_ocr(data: bytes) -> str:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency-dependent
        raise RuntimeError(
            "Image OCR requires Pillow and pytesseract, and the Tesseract binary must be installed."
        ) from exc
    image = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(image).strip()
    if not text:
        raise RuntimeError("OCR could not extract text from the uploaded image.")
    return text


def extract_uploaded_file(
    *,
    settings: Settings,
    file_name: str,
    content_type: str | None,
    data: bytes,
) -> ExtractedAsset:
    file_size_bytes = len(data)
    sha256 = hashlib.sha256(data).hexdigest()
    suffix = Path(file_name).suffix.lower()
    content_type = content_type or "application/octet-stream"

    if file_size_bytes > settings.max_upload_file_size_bytes:
        return ExtractedAsset(
            processing_status=AssetProcessingStatus.FAILED,
            extraction_method=AssetExtractionMethod.UNKNOWN,
            extracted_text=None,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            preview_excerpt=None,
            content_type=content_type,
            file_name=file_name,
            processing_error=(
                f"File exceeds the maximum upload size of "
                f"{settings.max_upload_file_size_bytes // (1024 * 1024)} MB."
            ),
        )

    try:
        if suffix in _JSON_EXTENSIONS or content_type == "application/json":
            extracted_text = _extract_json(data)
            method = AssetExtractionMethod.JSON
            page_count = None
            ocr_used = False
            warnings = []
        elif suffix in _CSV_EXTENSIONS or content_type in {"text/csv", "text/tab-separated-values"}:
            delimiter = (
                "\t" if suffix == ".tsv" or content_type == "text/tab-separated-values" else ","
            )
            extracted_text = _extract_csv(data, delimiter=delimiter)
            method = AssetExtractionMethod.CSV
            page_count = None
            ocr_used = False
            warnings = []
        elif suffix in _HTML_EXTENSIONS or content_type == "text/html":
            extracted_text = _extract_html(data)
            method = AssetExtractionMethod.HTML
            page_count = None
            ocr_used = False
            warnings = []
        elif suffix in _TEXT_EXTENSIONS or content_type.startswith("text/"):
            extracted_text = _decode_text(data)
            method = AssetExtractionMethod.TEXT
            page_count = None
            ocr_used = False
            warnings = []
        elif suffix in _PDF_EXTENSIONS or content_type == "application/pdf":
            extracted_text, method, ocr_used, page_count, warnings = _extract_pdf(
                data,
                max_ocr_pages=settings.max_ocr_pdf_pages,
                min_text_chars=settings.pdf_text_extraction_min_chars,
            )
        elif suffix in _DOCX_EXTENSIONS or content_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        }:
            extracted_text = _extract_docx(data)
            method = AssetExtractionMethod.DOCX
            page_count = None
            ocr_used = False
            warnings = []
        elif suffix in _IMAGE_EXTENSIONS or content_type.startswith("image/"):
            extracted_text = _extract_image_ocr(data)
            method = AssetExtractionMethod.IMAGE_OCR
            page_count = 1
            ocr_used = True
            warnings = []
        else:
            raise RuntimeError(
                f"Unsupported file type for '{file_name}'. Supported types: "
                "txt, md, json, csv, html, pdf, docx, and common images."
            )
        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise RuntimeError(f"No readable text was extracted from '{file_name}'.")
        return ExtractedAsset(
            processing_status=AssetProcessingStatus.READY,
            extraction_method=method,
            extracted_text=extracted_text,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            preview_excerpt=_preview_excerpt(extracted_text),
            content_type=content_type,
            file_name=file_name,
            warnings=warnings,
            ocr_used=ocr_used,
            page_count=page_count,
        )
    except Exception as exc:
        return ExtractedAsset(
            processing_status=AssetProcessingStatus.FAILED,
            extraction_method=AssetExtractionMethod.UNKNOWN,
            extracted_text=None,
            file_size_bytes=file_size_bytes,
            sha256=sha256,
            preview_excerpt=None,
            content_type=content_type,
            file_name=file_name,
            processing_error=str(exc),
        )
