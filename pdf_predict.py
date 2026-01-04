from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from predict import load_model_artifacts, load_thresholds, predict_dataframe
from src.utils import safe_mkdirs
from src.utils.pdf_processor import extract_sentences_with_pages, get_pdf_files, safe_report_name


def build_dataframe_from_pdf(
    pdf_path: Path,
    text_col: str,
    enable_ocr: bool,
    ocr_lang: str,
    ocr_dpi: int,
) -> pd.DataFrame:
    """Extract sentences/pages from a PDF and normalize column names for inference."""
    sentences = extract_sentences_with_pages(
        str(pdf_path),
        enable_ocr=enable_ocr,
        ocr_lang=ocr_lang,
        ocr_dpi=ocr_dpi,
    )
    if not sentences:
        raise ValueError(f"No sentences extracted from PDF: {pdf_path}")
    df = pd.DataFrame(sentences)
    df.rename(columns={"sentence": text_col, "page": "source_page"}, inplace=True)
    df["source_pdf"] = safe_report_name(str(pdf_path))
    return df


def parse_args() -> argparse.Namespace:
    """Define CLI for PDF-only batch or single-report prediction runs."""
    parser = argparse.ArgumentParser(
        description="Analyse one or more PDF reports with the finetuned wellbeing classifier."
    )
    parser.add_argument("--model_dir", required=True, help="Directory containing the trained model and tokenizer.")
    parser.add_argument(
        "pdf_paths",
        nargs="*",
        help="Optional PDF files to process. When omitted, PDFs are discovered in --pdf_dir.",
    )

    parser.add_argument(
        "--pdf_dir",
        default="reports",
        help="Directory to scan when no explicit PDF paths are provided.",
    )
    parser.add_argument("--output_dir", default="outputs/pdf_reports", help="Directory to write prediction CSV files.")
    parser.add_argument("--text_col", default="sentence", help="Name to use for the extracted sentence column.")
    parser.add_argument(
        "--enable_ocr",
        action="store_true",
        help="Enable OCR fallback for scanned PDFs (requires Pillow + pytesseract + system tesseract).",
    )
    parser.add_argument("--ocr_lang", default="eng", help="Tesseract language code for OCR.")
    parser.add_argument("--ocr_dpi", type=int, default=300, help="DPI used when rendering PDF pages for OCR.")
    parser.add_argument(
        "--label_cols",
        nargs="+",
        default=["social", "environment", "financial", "maori"],
        help="Ordered list of labels expected by the classifier.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    """Discover PDF files, run predictions, and write per-report CSVs."""
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    safe_mkdirs(output_dir)

    if args.pdf_paths:
        pdf_files: List[Path] = [Path(pdf_path) for pdf_path in args.pdf_paths]
    else:
        search_dir = Path(args.pdf_dir)
        if not search_dir.exists() or not search_dir.is_dir():
            raise ValueError(f"PDF directory not found: {search_dir}")
        discovered = get_pdf_files(str(search_dir))
        if not discovered:
            raise ValueError(f"No PDF files found in directory: {search_dir}")
        pdf_files = [Path(path) for path in discovered]

    tokenizer, model, device = load_model_artifacts(model_dir)
    thresholds = load_thresholds(model_dir, args.label_cols)

    for pdf_path in pdf_files:
        if not pdf_path.exists():
            print(f"Skipping {pdf_path}: file not found.")
            continue

        try:
            df = build_dataframe_from_pdf(
                pdf_path,
                args.text_col,
                enable_ocr=args.enable_ocr,
                ocr_lang=args.ocr_lang,
                ocr_dpi=args.ocr_dpi,
            )
        except ValueError as exc:
            print(f"Skipping {pdf_path}: {exc}")
            continue

        predictions = predict_dataframe(
            df,
            text_col=args.text_col,
            label_cols=args.label_cols,
            tokenizer=tokenizer,
            model=model,
            device=device,
            thresholds=thresholds,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

        report_name = safe_report_name(str(pdf_path)).lower()
        full_output_path = output_dir / f"{report_name}_predictions.csv"
        predictions.to_csv(full_output_path, index=False)

        prob_only = predictions.drop(columns=[col for col in predictions.columns if col.startswith("pred_")])
        prob_output_path = output_dir / f"{report_name}_probabilities.csv"
        prob_only.to_csv(prob_output_path, index=False)

        print(f"Saved predictions for {pdf_path} -> {full_output_path}")
        print(f"Saved probabilities-only view -> {prob_output_path}")


if __name__ == "__main__":
    main()
