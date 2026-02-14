"""
Clean up pre-2000s CJEU judgments from the dataset.

This script filters the dataset to only keep judgments from 2000 onwards,
which have numbered paragraphs suitable for precise alignment.

Removes:
    - Document files (French and German) from data/raw/
    - Entries from alignment_index.json

Usage:
    python -m src.data.cleanup_old_documents
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_year_from_celex(celex_id: str) -> int:
    """Extract year from CELEX identifier.

    CELEX format: 6YYYYCJXXXX where YYYY is the year.

    Args:
        celex_id: CELEX identifier (e.g., "62009CJ0296")

    Returns:
        Year as integer (e.g., 2009)

    Example:
        >>> extract_year_from_celex("62009CJ0296")
        2009
    """
    # Extract characters 1-5 (positions after "6")
    year_str = celex_id[1:5]
    return int(year_str)


def filter_documents_by_year(
    alignment_index: Dict,
    min_year: int = 2000
) -> tuple[Dict, List[str]]:
    """Filter alignment index to keep only documents from min_year onwards.

    Args:
        alignment_index: Full alignment index dictionary
        min_year: Minimum year to keep (default: 2000)

    Returns:
        Tuple of (filtered_index, removed_celex_ids)
    """
    filtered_index = {}
    removed_ids = []

    for celex_id, metadata in alignment_index.items():
        year = extract_year_from_celex(celex_id)

        if year >= min_year:
            filtered_index[celex_id] = metadata
            logger.debug(f"Keeping {celex_id} (year: {year})")
        else:
            removed_ids.append(celex_id)
            logger.debug(f"Removing {celex_id} (year: {year})")

    logger.info(f"Kept {len(filtered_index)} documents (>= {min_year})")
    logger.info(f"Removed {len(removed_ids)} documents (< {min_year})")

    return filtered_index, removed_ids


def delete_document_files(celex_ids: List[str], project_root: Path) -> None:
    """Delete French and German document files for given CELEX IDs.

    Args:
        celex_ids: List of CELEX identifiers to delete
        project_root: Project root directory
    """
    deleted_count = 0
    missing_count = 0

    for celex_id in celex_ids:
        # Delete French file
        fr_path = project_root / "data" / "raw" / "french" / f"{celex_id}.txt"
        if fr_path.exists():
            fr_path.unlink()
            deleted_count += 1
            logger.debug(f"Deleted {fr_path}")
        else:
            missing_count += 1
            logger.warning(f"File not found: {fr_path}")

        # Delete German file
        de_path = project_root / "data" / "raw" / "german" / f"{celex_id}.txt"
        if de_path.exists():
            de_path.unlink()
            deleted_count += 1
            logger.debug(f"Deleted {de_path}")
        else:
            missing_count += 1
            logger.warning(f"File not found: {de_path}")

    logger.info(f"Deleted {deleted_count} files")
    if missing_count > 0:
        logger.warning(f"{missing_count} files were already missing")


def main():
    """Main cleanup function."""
    logger.info("Starting cleanup of pre-2000s documents...")

    # Define paths
    project_root = Path(__file__).parent.parent.parent
    alignment_index_path = (
        project_root / "data" / "raw" / "metadata" / "alignment_index.json"
    )

    # Load alignment index
    logger.info(f"Loading alignment index from {alignment_index_path}")
    with open(alignment_index_path, "r", encoding="utf-8") as f:
        alignment_index = json.load(f)

    logger.info(f"Original index contains {len(alignment_index)} documents")

    # Filter by year
    filtered_index, removed_ids = filter_documents_by_year(
        alignment_index, min_year=2000
    )

    # Delete files for removed documents
    logger.info("Deleting document files for removed entries...")
    delete_document_files(removed_ids, project_root)

    # Save updated alignment index
    logger.info(f"Saving filtered alignment index ({len(filtered_index)} documents)")
    with open(alignment_index_path, "w", encoding="utf-8") as f:
        json.dump(filtered_index, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Documents kept (>= 2000): {len(filtered_index)}")
    logger.info(f"Documents removed (< 2000): {len(removed_ids)}")
    logger.info("=" * 60)
    logger.info("Cleanup complete! Dataset now contains only 2000+ judgments.")


if __name__ == "__main__":
    main()
