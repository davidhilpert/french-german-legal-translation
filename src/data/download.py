"""
Download CJEU preliminary reference rulings from EUR-Lex in French and German.

This module uses the eurlex R package via rpy2 to query and download parallel
legal documents from the European Union's legal database. Documents are aligned
by their CELEX identifiers.

Author: [Your Name]
Date: February 2026
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import pandas as pd
from tqdm import tqdm

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
except ImportError:
    print("ERROR: rpy2 not installed. Install with: pip install rpy2")
    print("Also ensure R is installed with the eurlex package: install.packages('eurlex')")
    sys.exit(1)


# Configure logging
LOG_DIR = Path(__file__).parent.parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "download.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EURLexDownloader:
    """Download and manage CJEU legal documents from EUR-Lex."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        languages: Tuple[str, str] = ("fr", "de"),
        delay: float = 1.0
    ):
        """
        Initialize the EUR-Lex downloader.

        Args:
            output_dir: Root directory for downloaded documents
            languages: Tuple of language codes (source, target)
            delay: Delay in seconds between API requests (rate limiting)
        """
        self.output_dir = Path(output_dir)
        self.languages = languages
        self.delay = delay

        # Create directory structure
        self.french_dir = self.output_dir / "french"
        self.german_dir = self.output_dir / "german"
        self.metadata_dir = self.output_dir / "metadata"

        for dir_path in [self.french_dir, self.german_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized downloader with output directory: {self.output_dir}")
        logger.info(f"Languages: {self.languages[0]} (source) -> {self.languages[1]} (target)")

        # Load R packages
        self._setup_r_environment()

    def _setup_r_environment(self) -> None:
        """Set up R environment and load eurlex package."""
        try:
            logger.info("Loading R packages...")

            # Check if eurlex is installed by checking rownames
            installed_pkgs = ro.r('rownames(installed.packages())')

            if 'eurlex' not in list(installed_pkgs):
                logger.error("eurlex package not found in R library paths")
                logger.error("Please install it with: sudo R -e \"install.packages('eurlex')\"")
                raise ImportError("eurlex R package is not installed")

            # Import eurlex
            self.eurlex = importr('eurlex')
            logger.info("R environment configured successfully")
            logger.info("eurlex package loaded successfully")

        except Exception as e:
            logger.error(f"Failed to set up R environment: {e}")
            logger.error("Please ensure R is installed and eurlex package is available")
            raise

    def query_cjeu_cases(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query CJEU preliminary reference cases from EUR-Lex.

        The query targets Court of Justice (CJEU) case-law, filtering for JUDG
        (judgment) documents only, excluding procedural notices and other document types.
        Results are further filtered by CELEX ID pattern to ensure only CJ (Court of Justice)
        cases are included, not General Court (TJ) or other tribunals.

        Args:
            start_year: Starting year for case retrieval (e.g., 2000)
            end_year: Ending year for case retrieval (e.g., 2023)
            limit: Maximum number of JUDGMENT cases to retrieve (for testing)

        Returns:
            DataFrame with case metadata including CELEX IDs

        Raises:
            Exception: If query fails or R environment has issues
        """
        try:
            logger.info("Querying CJEU judgments...")

            # Build query parameters
            # Note: sector "6" corresponds to case-law in CELEX numbering system
            # We query with a larger limit to ensure we get enough JUDG documents
            # after filtering out INFO_JUDICIAL and other procedural types
            query_limit = (limit * 10) if limit else 1000

            query_params = {
                'resource_type': 'caselaw',
                'sector': 6,  # Case-law sector (CELEX numbers starting with 6)
                'include_date': True,
                'include_date_lodged': True,
                'include_ecli': True,
                'include_court_procedure': True,  # Include procedure type
                'limit': query_limit  # Query more documents to account for filtering
            }

            logger.info(f"Querying up to {query_limit} case-law documents...")
            logger.info("Will filter for JUDG (judgment) documents only...")

            # Create SPARQL query
            logger.info("Creating SPARQL query...")
            query = self.eurlex.elx_make_query(**query_params)

            # Execute query
            logger.info("Executing query against EUR-Lex endpoint...")
            results = self.eurlex.elx_run_query(query)

            # Convert R dataframe to pandas using context manager
            with localconverter(ro.default_converter + pandas2ri.converter):
                df = ro.conversion.rpy2py(results)

            logger.info(f"Query returned {len(df)} case-law documents")
            logger.info(f"Available columns: {list(df.columns)}")

            # Log sample CELEX IDs and types for debugging
            if len(df) > 0:
                if 'celex' in df.columns:
                    logger.info(f"Sample CELEX IDs: {df['celex'].head(5).tolist()}")
                if 'type' in df.columns:
                    type_counts = df['type'].value_counts().to_dict()
                    logger.info(f"Document type distribution: {type_counts}")

            # Filter by document type (keep only JUDG = judgments)
            # Exclude INFO_JUDICIAL (procedural notices), ORDER, OPIN_AG, etc.
            if 'type' in df.columns:
                # Filter for exact match on "JUDG" to exclude INFO_JUDICIAL
                df = df[df['type'] == 'JUDG']
                logger.info(f"Filtered to {len(df)} JUDG (judgment) documents")

                if len(df) == 0:
                    logger.warning("No JUDG documents found! All results were procedural documents.")
                    logger.warning("Consider increasing query limit or checking EUR-Lex data availability.")
                    return df

            # Filter for CJEU cases (Court of Justice)
            # CELEX format: 6YYYYAA0NNN where:
            # - 6 = case law
            # - YYYY = year (e.g., 2019)
            # - AA = court code (CJ=Court of Justice, TJ=General Court, etc.)
            # - 0NNN = case number
            # Examples: 62019CJ0100, 62020CJ0050
            if 'celex' in df.columns:
                # Match CELEX IDs that contain 'CJ' (Court of Justice)
                # Pattern: starts with 6, followed by 4 digits (year), then CJ
                df_filtered = df[df['celex'].str.match(r'^6\d{4}CJ\d+', na=False)]
                logger.info(f"Filtered to {len(df_filtered)} CJEU Court of Justice cases")

                if len(df_filtered) == 0:
                    # Fall back to broader pattern if strict pattern yields no results
                    logger.warning("Strict CELEX pattern matched 0 cases, trying broader filter")
                    df_filtered = df[df['celex'].str.contains(r'6\d{4}C[JA]', na=False, regex=True)]
                    logger.info(f"Broader filter found {len(df_filtered)} CJEU cases (CJ+CA)")
            else:
                df_filtered = df
                logger.warning("No 'celex' column found in results")

            # Apply year filtering if specified
            if start_year or end_year:
                # Try different date column names
                date_col = None
                for col in ['date', 'work_date_document', 'datelodged']:
                    if col in df_filtered.columns:
                        date_col = col
                        break

                if date_col:
                    df_filtered['year'] = pd.to_datetime(
                        df_filtered[date_col],
                        errors='coerce'
                    ).dt.year

                    if start_year:
                        df_filtered = df_filtered[df_filtered['year'] >= start_year]
                    if end_year:
                        df_filtered = df_filtered[df_filtered['year'] <= end_year]

                    logger.info(f"Filtered to {len(df_filtered)} cases between "
                              f"{start_year or 'earliest'} and {end_year or 'latest'}")
                else:
                    logger.warning("No date column found for year filtering")

            # Apply user's requested limit AFTER all filtering
            if limit and len(df_filtered) > limit:
                logger.info(f"Limiting results to requested {limit} cases")
                df_filtered = df_filtered.head(limit)

            # Log sample of final CELEX IDs
            if len(df_filtered) > 0 and 'celex' in df_filtered.columns:
                logger.info(f"Final CELEX sample: {df_filtered['celex'].head(10).tolist()}")

            return df_filtered

        except Exception as e:
            logger.error(f"Error querying CJEU cases: {e}")
            raise

    def download_document_text(
        self,
        celex_id: str,
        language: str,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Download document text in specified language.

        Args:
            celex_id: CELEX identifier (e.g., '62019CJ0100')
            language: ISO 639 language code ('fr', 'de', etc.)
            max_retries: Number of retry attempts on failure

        Returns:
            Document text as string, or None if download fails
        """
        for attempt in range(max_retries):
            try:
                # Construct EUR-Lex URL from CELEX ID
                url = f"http://publications.europa.eu/resource/celex/{celex_id}"

                # Fetch document text with specified language priority
                logger.debug(f"Fetching {language} text for {celex_id} (attempt {attempt + 1})")

                result = self.eurlex.elx_fetch_data(
                    url=url,
                    type='text',
                    language_1=language
                )

                # Convert R character vector to Python string
                text = str(result[0]) if len(result) > 0 else None

                # Check if we got valid text
                if text and len(text) > 100:  # Reasonable minimum length
                    logger.debug(f"Successfully downloaded {len(text)} characters")
                    time.sleep(self.delay)  # Rate limiting
                    return text
                else:
                    logger.warning(f"Text too short or empty for {celex_id} ({language})")
                    return None

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {celex_id} ({language}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.delay * 2)  # Longer delay on error
                else:
                    logger.error(f"Failed to download {celex_id} ({language}) after {max_retries} attempts")
                    return None

    def save_document(
        self,
        celex_id: str,
        text: str,
        language: str
    ) -> Path:
        """
        Save document text to file.

        Args:
            celex_id: CELEX identifier
            text: Document text content
            language: Language code

        Returns:
            Path to saved file
        """
        # Determine output directory based on language
        if language == "fr":
            output_dir = self.french_dir
        elif language == "de":
            output_dir = self.german_dir
        else:
            output_dir = self.output_dir / language
            output_dir.mkdir(exist_ok=True)

        # Create filename: CELEX_ID.txt
        filename = f"{celex_id}.txt"
        filepath = output_dir / filename

        # Write text to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)

        logger.debug(f"Saved {language} document to {filepath}")
        return filepath

    def download_parallel_documents(
        self,
        cases_df: pd.DataFrame,
        max_documents: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Download parallel French-German documents for all cases.

        Args:
            cases_df: DataFrame with case metadata (must include 'celex' column)
            max_documents: Maximum number of documents to download (for testing)

        Returns:
            DataFrame with download status for each case
        """
        if 'celex' not in cases_df.columns:
            logger.error("Cases DataFrame must include 'celex' column")
            raise ValueError("Missing 'celex' column in cases DataFrame")

        # Limit documents if specified
        if max_documents:
            cases_df = cases_df.head(max_documents)
            logger.info(f"Limited to {max_documents} documents for download")

        # Track download results
        results = []

        logger.info(f"Starting download of {len(cases_df)} parallel documents...")

        for _, row in tqdm(cases_df.iterrows(), total=len(cases_df), desc="Downloading"):
            celex_id = row['celex']
            result = {
                'celex_id': celex_id,
                'french_success': False,
                'german_success': False,
                'french_path': None,
                'german_path': None,
                'french_length': 0,
                'german_length': 0,
                'timestamp': datetime.now().isoformat()
            }

            # Download French version
            logger.info(f"Downloading {celex_id} (French)...")
            fr_text = self.download_document_text(celex_id, 'fr')

            if fr_text:
                fr_path = self.save_document(celex_id, fr_text, 'fr')
                result['french_success'] = True
                result['french_path'] = str(fr_path)
                result['french_length'] = len(fr_text)

            # Download German version
            logger.info(f"Downloading {celex_id} (German)...")
            de_text = self.download_document_text(celex_id, 'de')

            if de_text:
                de_path = self.save_document(celex_id, de_text, 'de')
                result['german_success'] = True
                result['german_path'] = str(de_path)
                result['german_length'] = len(de_text)

            # Log status
            if result['french_success'] and result['german_success']:
                logger.info(f"✓ {celex_id}: Both languages downloaded successfully")
            elif result['french_success'] or result['german_success']:
                logger.warning(f"⚠ {celex_id}: Only one language downloaded")
            else:
                logger.error(f"✗ {celex_id}: Both downloads failed")

            results.append(result)

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Log summary statistics
        total = len(results_df)
        both_success = results_df[results_df['french_success'] & results_df['german_success']].shape[0]
        fr_only = results_df[results_df['french_success'] & ~results_df['german_success']].shape[0]
        de_only = results_df[~results_df['french_success'] & results_df['german_success']].shape[0]
        both_failed = results_df[~results_df['french_success'] & ~results_df['german_success']].shape[0]

        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total cases processed: {total}")
        logger.info(f"Both languages downloaded: {both_success} ({both_success/total*100:.1f}%)")
        logger.info(f"French only: {fr_only}")
        logger.info(f"German only: {de_only}")
        logger.info(f"Both failed: {both_failed}")
        logger.info("=" * 60)

        return results_df

    def save_metadata(
        self,
        cases_df: pd.DataFrame,
        results_df: pd.DataFrame,
        filename: str = "download_metadata.csv"
    ) -> Path:
        """
        Save comprehensive metadata for downloaded documents.

        Args:
            cases_df: Original cases DataFrame from query
            results_df: Download results DataFrame
            filename: Output filename

        Returns:
            Path to saved metadata file
        """
        # Merge case metadata with download results
        metadata = cases_df.merge(
            results_df,
            left_on='celex',
            right_on='celex_id',
            how='left'
        )

        # Save to CSV
        metadata_path = self.metadata_dir / filename
        metadata.to_csv(metadata_path, index=False, encoding='utf-8')

        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Metadata contains {len(metadata)} entries")

        return metadata_path

    def create_alignment_index(
        self,
        results_df: pd.DataFrame,
        filename: str = "alignment_index.json"
    ) -> Path:
        """
        Create JSON index mapping CELEX IDs to parallel document paths.

        Args:
            results_df: Download results DataFrame
            filename: Output filename

        Returns:
            Path to saved index file
        """
        # Filter for successful parallel downloads
        aligned = results_df[
            results_df['french_success'] & results_df['german_success']
        ]

        # Create alignment mapping
        alignment_index = {}
        for _, row in aligned.iterrows():
            alignment_index[row['celex_id']] = {
                'french_path': row['french_path'],
                'german_path': row['german_path'],
                'french_length': row['french_length'],
                'german_length': row['german_length'],
                'timestamp': row['timestamp']
            }

        # Save to JSON
        index_path = self.metadata_dir / filename
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(alignment_index, f, indent=2, ensure_ascii=False)

        logger.info(f"Alignment index saved to {index_path}")
        logger.info(f"Index contains {len(alignment_index)} aligned document pairs")

        return index_path


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download CJEU preliminary reference rulings from EUR-Lex"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for downloaded documents'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Starting year for case retrieval (e.g., 2000)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='Ending year for case retrieval (e.g., 2023)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit query results (for testing)'
    )
    parser.add_argument(
        '--max-documents',
        type=int,
        help='Maximum number of documents to download (for testing)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between API requests in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    try:
        # Initialize downloader
        downloader = EURLexDownloader(
            output_dir=args.output_dir,
            languages=('fr', 'de'),
            delay=args.delay
        )

        # Query CJEU cases
        cases_df = downloader.query_cjeu_cases(
            start_year=args.start_year,
            end_year=args.end_year,
            limit=args.limit
        )

        if len(cases_df) == 0:
            logger.warning("No cases found matching criteria")
            return

        # Download parallel documents
        results_df = downloader.download_parallel_documents(
            cases_df,
            max_documents=args.max_documents
        )

        # Save metadata and alignment index
        downloader.save_metadata(cases_df, results_df)
        downloader.create_alignment_index(results_df)

        logger.info("Download process completed successfully!")

    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
