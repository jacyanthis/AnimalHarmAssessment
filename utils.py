#!/usr/bin/env python3
"""
Common utilities for the AHA benchmark.
"""

import csv
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

# Default logging format
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

class FileUtils:
    """File system utilities."""
    
    @staticmethod
    def mkd(directory: Path) -> None:
        """Create directory if it doesn't exist."""
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_json(file_path: Path) -> Any:
        """Read JSON file with UTF-8 encoding."""
        return json.loads(file_path.read_text(encoding='utf-8'))

    @staticmethod
    def write_json(file_path: Path, data: Any) -> None:
        """Write data to JSON file with UTF-8 encoding."""
        file_path.write_text(json.dumps(data), encoding='utf-8')

    @staticmethod
    def get_latest_file(directory: Path, pattern: str) -> Optional[Path]:
        """Get most recently modified file matching pattern in directory."""
        files = list(directory.glob(pattern))
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

class DataFrameUtils:
    """Pandas DataFrame utilities."""
    
    @staticmethod
    def read_csv_as_df(csv_path: Path) -> pd.DataFrame:
        """Read CSV file into pandas DataFrame."""
        return pd.read_csv(csv_path)

    @staticmethod
    def combine_csv_files(csv_files: List[Path]) -> pd.DataFrame:
        """Combine multiple CSV files into a single DataFrame."""
        dfs = [pd.read_csv(f) for f in csv_files]
        return pd.concat(dfs, ignore_index=True)

class TimeUtils:
    """Time-related utilities."""
    
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    TIMESTAMP_PATTERN = r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\+\d{2}-\d{2})"
    
    @classmethod
    def generate_timestamp(cls) -> str:
        """Generate current timestamp string."""
        return datetime.now().strftime(cls.TIMESTAMP_FORMAT)

    @classmethod
    def extract_timestamp(cls, log_filename: str) -> str:
        """Extract timestamp from log filename or generate new one."""
        match = re.search(cls.TIMESTAMP_PATTERN, log_filename)
        return match.group(1) if match else cls.generate_timestamp()

def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging with specified level."""
    logging.basicConfig(level=level, format=LOG_FORMAT)

# For backward compatibility
mkd = FileUtils.mkd
read_json = FileUtils.read_json
write_json = FileUtils.write_json
get_latest_file = FileUtils.get_latest_file
read_csv_as_df = DataFrameUtils.read_csv_as_df
combine_csv_files = DataFrameUtils.combine_csv_files
generate_timestamp = TimeUtils.generate_timestamp
extract_timestamp = TimeUtils.extract_timestamp
