#!/usr/bin/env python3
"""
Spatial Data Downloader for Water Risk Application

Downloads watershed data and political boundaries for California, Nevada, 
Oregon, and Washington from authoritative sources.

Data Sources:
- USGS Watershed Boundary Dataset (WBD) - HUC-12 watersheds
- Census Bureau TIGER/Line - Political boundaries
- USGS PAD-US - Protected areas
- State GIS Portals - Special districts and state-specific data

Usage:
    python download_spatial_data.py [--all] [--category CATEGORY] [--dataset DATASET] [--list]
    
Examples:
    python download_spatial_data.py --all                    # Download everything
    python download_spatial_data.py --category watershed     # Download only watershed data
    python download_spatial_data.py --category political     # Download only political boundaries
    python download_spatial_data.py --dataset watershed_huc12  # Download specific dataset
    python download_spatial_data.py --list                   # List available datasets
"""

import argparse
import hashlib
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from config import (
    DATA_DIR,
    DATA_SOURCES,
    DOWNLOAD_SETTINGS,
    RAW_DATA_DIR,
    STATES_OF_INTEREST,
)


class SpatialDataDownloader:
    """Downloads and manages spatial datasets for the water risk application."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the downloader with a target data directory."""
        self.data_dir = data_dir or RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.download_log = []
        
    def _get_dataset_path(self, dataset_key: str, dataset_info: dict) -> Path:
        """Get the download path for a dataset."""
        category = dataset_info.get("category", "other")
        state = dataset_info.get("state", "national")
        
        # Create category/state subdirectory structure
        if state != "national":
            subdir = self.data_dir / category / state.lower()
        else:
            subdir = self.data_dir / category / "national"
        
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Extract filename from URL
        url = dataset_info["url"]
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # Handle URLs with query parameters (like ScienceBase)
        if not filename or filename == "get":
            filename = f"{dataset_key}.zip"
        
        return subdir / filename

    def _download_file(
        self, 
        url: str, 
        dest_path: Path, 
        description: str = "",
        show_progress: bool = True
    ) -> bool:
        """
        Download a file from URL to destination path with progress bar.
        
        Returns True if download was successful, False otherwise.
        """
        settings = DOWNLOAD_SETTINGS
        
        for attempt in range(settings["max_retries"]):
            try:
                # Make request with streaming
                response = requests.get(
                    url,
                    stream=True,
                    timeout=settings["timeout"],
                    verify=settings["verify_ssl"],
                    headers={
                        "User-Agent": "WaterRiskApp/1.0 (Environmental Data Research)"
                    }
                )
                response.raise_for_status()
                
                # Get total file size if available
                total_size = int(response.headers.get("content-length", 0))
                
                # Download with progress bar
                desc = description or dest_path.name
                with open(dest_path, "wb") as f:
                    if show_progress and total_size > 0:
                        with tqdm(
                            total=total_size,
                            unit="iB",
                            unit_scale=True,
                            desc=desc[:40],
                            ncols=80,
                        ) as pbar:
                            for chunk in response.iter_content(
                                chunk_size=settings["chunk_size"]
                            ):
                                size = f.write(chunk)
                                pbar.update(size)
                    else:
                        for chunk in response.iter_content(
                            chunk_size=settings["chunk_size"]
                        ):
                            f.write(chunk)
                
                return True
                
            except requests.exceptions.RequestException as e:
                print(f"\n  ⚠ Attempt {attempt + 1}/{settings['max_retries']} failed: {e}")
                if attempt < settings["max_retries"] - 1:
                    print(f"  Retrying in {settings['retry_delay']} seconds...")
                    time.sleep(settings["retry_delay"])
                else:
                    print(f"  ✗ Failed to download after {settings['max_retries']} attempts")
                    return False
        
        return False

    def _extract_zip(self, zip_path: Path, extract_dir: Optional[Path] = None) -> bool:
        """Extract a zip file to the specified directory."""
        if extract_dir is None:
            extract_dir = zip_path.parent / zip_path.stem
        
        try:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"  ✓ Extracted to: {extract_dir}")
            return True
        except zipfile.BadZipFile:
            print(f"  ✗ Invalid zip file: {zip_path}")
            return False
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            return False

    def download_dataset(
        self, 
        dataset_key: str, 
        force: bool = False,
        extract: bool = True
    ) -> bool:
        """
        Download a single dataset by its key.
        
        Args:
            dataset_key: The key of the dataset in DATA_SOURCES
            force: If True, re-download even if file exists
            extract: If True, extract zip files after download
            
        Returns:
            True if download was successful, False otherwise
        """
        if dataset_key not in DATA_SOURCES:
            print(f"✗ Unknown dataset: {dataset_key}")
            return False
        
        dataset_info = DATA_SOURCES[dataset_key]
        dest_path = self._get_dataset_path(dataset_key, dataset_info)
        extract_dir = dest_path.parent / dest_path.stem
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_info['name']}")
        print(f"Source: {dataset_info['source']}")
        print(f"Format: {dataset_info['format'].upper()}")
        if "size_estimate" in dataset_info:
            print(f"Estimated size: {dataset_info['size_estimate']}")
        print(f"{'='*60}")
        
        # Check if already downloaded and extracted
        if not force:
            if extract_dir.exists() and any(extract_dir.iterdir()):
                print(f"  ✓ Already downloaded and extracted: {extract_dir}")
                self.download_log.append({
                    "dataset": dataset_key,
                    "status": "skipped",
                    "path": str(extract_dir),
                    "reason": "already exists"
                })
                return True
            elif dest_path.exists():
                print(f"  ✓ Zip file exists: {dest_path}")
                if extract:
                    return self._extract_zip(dest_path, extract_dir)
                return True
        
        # Download the file
        print(f"  Downloading from: {dataset_info['url'][:70]}...")
        success = self._download_file(
            url=dataset_info["url"],
            dest_path=dest_path,
            description=dataset_info["name"]
        )
        
        if success:
            print(f"  ✓ Downloaded: {dest_path}")
            self.download_log.append({
                "dataset": dataset_key,
                "status": "downloaded",
                "path": str(dest_path),
                "timestamp": datetime.now().isoformat()
            })
            
            # Extract if it's a zip file
            if extract and dest_path.suffix.lower() == ".zip":
                self._extract_zip(dest_path, extract_dir)
        else:
            self.download_log.append({
                "dataset": dataset_key,
                "status": "failed",
                "url": dataset_info["url"]
            })
        
        return success

    def download_by_category(
        self, 
        category: str, 
        force: bool = False,
        extract: bool = True
    ) -> dict:
        """
        Download all datasets in a category.
        
        Categories: watershed, political, protected, special_district
        """
        results = {"success": [], "failed": [], "skipped": []}
        
        matching_datasets = {
            k: v for k, v in DATA_SOURCES.items() 
            if v.get("category") == category
        }
        
        if not matching_datasets:
            print(f"No datasets found for category: {category}")
            return results
        
        print(f"\n{'#'*60}")
        print(f"# Downloading {len(matching_datasets)} datasets in category: {category}")
        print(f"{'#'*60}")
        
        for dataset_key in matching_datasets:
            if self.download_dataset(dataset_key, force=force, extract=extract):
                results["success"].append(dataset_key)
            else:
                results["failed"].append(dataset_key)
        
        return results

    def download_all(
        self, 
        force: bool = False, 
        extract: bool = True,
        priority_max: int = 99
    ) -> dict:
        """
        Download all configured datasets.
        
        Args:
            force: Re-download even if files exist
            extract: Extract zip files after download
            priority_max: Only download datasets with priority <= this value
        """
        results = {"success": [], "failed": [], "skipped": []}
        
        # Sort by priority
        sorted_datasets = sorted(
            DATA_SOURCES.items(),
            key=lambda x: x[1].get("priority", 99)
        )
        
        filtered_datasets = [
            (k, v) for k, v in sorted_datasets 
            if v.get("priority", 99) <= priority_max
        ]
        
        print(f"\n{'#'*60}")
        print(f"# Downloading {len(filtered_datasets)} datasets")
        print(f"{'#'*60}")
        
        for dataset_key, _ in filtered_datasets:
            if self.download_dataset(dataset_key, force=force, extract=extract):
                results["success"].append(dataset_key)
            else:
                results["failed"].append(dataset_key)
        
        return results

    def list_datasets(self, category: Optional[str] = None):
        """List available datasets, optionally filtered by category."""
        print("\n" + "=" * 80)
        print("AVAILABLE SPATIAL DATASETS")
        print("=" * 80)
        
        categories = {}
        for key, info in DATA_SOURCES.items():
            cat = info.get("category", "other")
            if category and cat != category:
                continue
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((key, info))
        
        for cat, datasets in sorted(categories.items()):
            print(f"\n{'─'*40}")
            print(f"Category: {cat.upper()}")
            print(f"{'─'*40}")
            
            for key, info in sorted(datasets, key=lambda x: x[1].get("priority", 99)):
                state = info.get("state", "National")
                print(f"\n  [{key}]")
                print(f"    Name: {info['name']}")
                print(f"    Source: {info['source']}")
                print(f"    Format: {info['format'].upper()}")
                print(f"    Scope: {state}")
                print(f"    Priority: {info.get('priority', 'N/A')}")
                if "size_estimate" in info:
                    print(f"    Size: {info['size_estimate']}")
        
        print("\n" + "=" * 80)

    def print_summary(self, results: dict):
        """Print a summary of download results."""
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"  ✓ Successful: {len(results['success'])}")
        print(f"  ✗ Failed: {len(results['failed'])}")
        
        if results["failed"]:
            print("\nFailed datasets:")
            for ds in results["failed"]:
                print(f"  - {ds}")
        
        print("\nData directory:", self.data_dir)
        print("=" * 60)


def main():
    """Main entry point for the download script."""
    parser = argparse.ArgumentParser(
        description="Download spatial datasets for the Water Risk application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                       Download all datasets
  %(prog)s --category watershed        Download watershed data only
  %(prog)s --category political        Download political boundaries only
  %(prog)s --category protected        Download protected areas only
  %(prog)s --dataset watershed_huc12   Download specific dataset
  %(prog)s --list                      List all available datasets
  %(prog)s --list --category political List datasets in a category
        """
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--category", 
        type=str,
        choices=["watershed", "political", "protected", "special_district"],
        help="Download datasets by category"
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        help="Download a specific dataset by key"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--no-extract", 
        action="store_true",
        help="Don't extract zip files after download"
    )
    parser.add_argument(
        "--priority", 
        type=int,
        default=99,
        help="Only download datasets with priority <= this value (default: 99)"
    )
    parser.add_argument(
        "--data-dir", 
        type=str,
        help="Custom data directory (default: ./data/raw)"
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    data_dir = Path(args.data_dir) if args.data_dir else None
    downloader = SpatialDataDownloader(data_dir=data_dir)
    
    # Handle listing
    if args.list:
        downloader.list_datasets(category=args.category)
        return 0
    
    # Handle downloads
    extract = not args.no_extract
    
    if args.dataset:
        success = downloader.download_dataset(
            args.dataset, 
            force=args.force, 
            extract=extract
        )
        return 0 if success else 1
    
    if args.category:
        results = downloader.download_by_category(
            args.category, 
            force=args.force, 
            extract=extract
        )
        downloader.print_summary(results)
        return 0 if not results["failed"] else 1
    
    if args.all:
        results = downloader.download_all(
            force=args.force, 
            extract=extract,
            priority_max=args.priority
        )
        downloader.print_summary(results)
        return 0 if not results["failed"] else 1
    
    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

