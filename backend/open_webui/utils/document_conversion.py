"""LibreOffice-backed conversion of office documents to PDF.

Used so we can send the converted PDF natively to multimodal models that
accept PDF content blocks but don't accept .docx/.xlsx/etc.
"""

import logging
import os
import shutil
import subprocess
import tempfile

log = logging.getLogger(__name__)

# Extensions LibreOffice can convert to PDF with reasonable fidelity.
CONVERTIBLE_EXTENSIONS = {
    "doc", "docx", "rtf", "odt",
    "xls", "xlsx", "ods", "csv",
    "ppt", "pptx", "odp",
}

CONVERSION_TIMEOUT_SECONDS = 90


def is_convertible(extension: str) -> bool:
    if not extension:
        return False
    return extension.lower().lstrip(".") in CONVERTIBLE_EXTENSIONS


def convert_to_pdf(source_path: str) -> bytes:
    """Convert an office document at `source_path` to PDF bytes.

    Raises RuntimeError if soffice is missing, the conversion times out,
    or the converter produces no PDF.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError(
            "LibreOffice (soffice) not found on PATH; cannot convert document"
        )

    with tempfile.TemporaryDirectory(prefix="owu-soffice-") as tmpdir:
        try:
            result = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--norestore",
                    "--nologo",
                    "--nofirststartwizard",
                    "--convert-to", "pdf",
                    "--outdir", tmpdir,
                    source_path,
                ],
                check=True,
                capture_output=True,
                timeout=CONVERSION_TIMEOUT_SECONDS,
                env={**os.environ, "HOME": tmpdir},
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"LibreOffice conversion timed out after {CONVERSION_TIMEOUT_SECONDS}s"
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"LibreOffice conversion failed: {e.stderr.decode(errors='replace')}"
            ) from e

        stem = os.path.splitext(os.path.basename(source_path))[0]
        pdf_path = os.path.join(tmpdir, f"{stem}.pdf")
        if not os.path.isfile(pdf_path):
            # Some versions emit a different stem; pick whatever .pdf landed.
            candidates = [p for p in os.listdir(tmpdir) if p.lower().endswith(".pdf")]
            if not candidates:
                raise RuntimeError(
                    f"LibreOffice produced no PDF (stdout={result.stdout!r})"
                )
            pdf_path = os.path.join(tmpdir, candidates[0])

        with open(pdf_path, "rb") as f:
            return f.read()
