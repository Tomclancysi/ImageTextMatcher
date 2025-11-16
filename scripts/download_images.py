"""
Robust image downloader script for dataset.csv
Features:
- Resume interrupted downloads
- Retry failed downloads
- Progress tracking
- Duplicate detection
- Error handling and logging
"""

import argparse
import csv
import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Set, Dict, List, Optional
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import json


class ImageDownloader:
    def __init__(self,
                 output_dir: str = "data/images",
                 max_retries: int = 3,
                 timeout: int = 30,
                 chunk_size: int = 8192,
                 resume_file: str = "download_progress.json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.resume_file = Path(resume_file)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('download.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.downloaded_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.load_progress()
        
        self.stats = {
            'total': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': time.time()
        }

    def load_progress(self):
        """Load download progress from resume file."""
        if self.resume_file.exists():
            try:
                with open(self.resume_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.downloaded_urls = set(data.get('downloaded', []))
                    self.failed_urls = set(data.get('failed', []))
                    self.logger.info(f"Loaded progress: {len(self.downloaded_urls)} downloaded, {len(self.failed_urls)} failed")
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")

    def save_progress(self):
        """Save download progress to resume file."""
        try:
            data = {
                'downloaded': list(self.downloaded_urls),
                'failed': list(self.failed_urls),
                'timestamp': time.time()
            }
            with open(self.resume_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Could not save progress: {e}")

    def get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL."""
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        
        if not filename or '.' not in filename:
            filename = f"image_{hashlib.md5(url.encode()).hexdigest()[:8]}.jpg"
        
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        return filename

    def is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image."""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def download_image(self, url: str, output_path: Path) -> bool:
        """Download a single image with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"Downloading {url} (attempt {attempt + 1})")
                
                response = self.session.get(
                    url,
                    timeout=self.timeout,
                    stream=True,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                )
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                    self.logger.warning(f"Unexpected content type for {url}: {content_type}")
                
                total_size = int(response.headers.get('content-length', 0))
                with open(output_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name, leave=False) as pbar:
                            for chunk in response.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=self.chunk_size):
                            if chunk:
                                f.write(chunk)
                
                if output_path.exists() and output_path.stat().st_size > 0:
                    if self.is_valid_image(output_path):
                        self.logger.info(f"Successfully downloaded: {output_path.name}")
                        return True
                    else:
                        self.logger.warning(f"Downloaded file is not a valid image: {output_path.name}")
                        output_path.unlink(missing_ok=True)
                        return False
                else:
                    self.logger.warning(f"Downloaded file is empty: {output_path.name}")
                    output_path.unlink(missing_ok=True)
                    return False
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"All attempts failed for {url}")
                    return False
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {url}: {e}")
                return False
        
        return False

    def process_url(self, url: str) -> bool:
        """Process a single URL."""
        if url in self.downloaded_urls:
            self.stats['skipped'] += 1
            return True
        
        if url in self.failed_urls:
            self.stats['failed'] += 1
            return False
        
        filename = self.get_filename_from_url(url)
        output_path = self.output_dir / filename
        
        if output_path.exists() and self.is_valid_image(output_path):
            self.downloaded_urls.add(url)
            self.stats['skipped'] += 1
            return True
        
        success = self.download_image(url, output_path)
        
        if success:
            self.downloaded_urls.add(url)
            self.failed_urls.discard(url)
            self.stats['downloaded'] += 1
        else:
            self.failed_urls.add(url)
            self.stats['failed'] += 1
        
        return success

    def download_from_csv(self, csv_file: str, start_index: int = 0, max_images: Optional[int] = None):
        """Download images from CSV file."""
        self.logger.info(f"Starting download from {csv_file}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                urls = [row['url'] for row in reader if row.get('url')]
        except Exception as e:
            self.logger.error(f"Could not read CSV file: {e}")
            return
        
        total_urls = len(urls)
        self.stats['total'] = total_urls
        
        if max_images:
            urls = urls[:max_images]
        
        self.logger.info(f"Found {len(urls)} URLs to process")
        
        with tqdm(total=len(urls), desc="Downloading images") as pbar:
            for i, url in enumerate(urls[start_index:], start=start_index):
                try:
                    self.process_url(url)
                    pbar.update(1)
                    
                    if (i + 1) % 10 == 0:
                        self.save_progress()
                        
                except KeyboardInterrupt:
                    self.logger.info("Download interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Error processing URL {url}: {e}")
                    continue
        
        self.save_progress()
        self.print_stats()

    def print_stats(self):
        """Print download statistics."""
        elapsed = time.time() - self.stats['start_time']
        self.logger.info("=" * 50)
        self.logger.info("DOWNLOAD STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total URLs: {self.stats['total']}")
        self.logger.info(f"Downloaded: {self.stats['downloaded']}")
        self.logger.info(f"Skipped (already exists): {self.stats['skipped']}")
        self.logger.info(f"Failed: {self.stats['failed']}")
        self.logger.info(f"Time elapsed: {elapsed:.2f} seconds")
        if self.stats['downloaded'] > 0:
            self.logger.info(f"Average time per download: {elapsed/self.stats['downloaded']:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Download images from dataset.csv")
    parser.add_argument("--csv", default="dataset.csv", help="CSV file with image URLs")
    parser.add_argument("--output", default="data/images", help="Output directory for images")
    parser.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--max", type=int, help="Maximum number of images to download")
    parser.add_argument("--retries", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--resume", default="download_progress.json", help="Resume file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found")
        sys.exit(1)
    
    downloader = ImageDownloader(
        output_dir=args.output,
        max_retries=args.retries,
        timeout=args.timeout,
        resume_file=args.resume
    )
    
    try:
        downloader.download_from_csv(args.csv, args.start, args.max)
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Progress saved.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
