"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   JDIH Sumatera Utara — LLM Database Builder                               ║
║   Penulis : Fauzan Nur Ahmadi, USU                                          ║
║   Tujuan  : Scraping & ekstraksi teks peraturan untuk corpus LLM            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Mendukung dua sumber utama:
  1. https://jdih.sumutprov.go.id          (JDIH Resmi Prov. Sumut)
  2. https://peraturan.bpk.go.id           (JDIH BPK – mirror nasional)

Pipeline:
  Crawl daftar peraturan  →  Download PDF  →  Ekstraksi teks  →  Chunking
  →  Simpan JSON/CSV/JSONL  →  Laporan ringkasan
"""

import os
import re
import json
import time
import csv
import hashlib
import logging
import argparse
import urllib.parse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich import print as rprint

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("jdih_scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("jdih_scraper")
console = Console()

# ─── Konstanta ────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
}

JENIS_PERATURAN = {
    "perda":   "Peraturan Daerah Provinsi",
    "pergub":  "Peraturan Gubernur",
    "sk":      "Surat Keputusan",
    "se":      "Surat Edaran",
    "instruksi": "Instruksi Gubernur",
}

# Mapping URL sumber ke tipe
SUMBER_URL = {
    "jdih_sumut": {
        "base":       "https://jdih.sumutprov.go.id",
        "daftar":     "https://jdih.sumutprov.go.id/produk-hukum",
        "param_jenis":"jenis",
        "param_page": "page",
    },
    "bpk": {
        "base":       "https://peraturan.bpk.go.id",
        "daftar":     "https://peraturan.bpk.go.id/Home/Peraturan?kodePropinsi=12",  # kode Sumut = 12
        "param_page": "page",
    }
}

DELAY_ANTAR_REQUEST = 1.5   # detik — menghormati server
MAX_RETRY           = 3
CHUNK_SIZE_CHARS    = 1500  # ukuran chunk untuk LLM (±1500 karakter ≈ ~375 token)
CHUNK_OVERLAP_CHARS = 200   # overlap antar chunk


# ─── Data Model ───────────────────────────────────────────────────────────────
@dataclass
class DokumenHukum:
    """Representasi satu dokumen hukum yang di-scrape."""
    id:                str  = ""      # hash MD5 URL
    sumber:            str  = ""      # 'jdih_sumut' | 'bpk'
    jenis:             str  = ""      # perda / pergub / sk / se
    judul:             str  = ""
    nomor:             str  = ""
    tahun:             str  = ""
    tanggal_ditetapkan:str  = ""
    tanggal_diundangkan:str = ""
    tentang:           str  = ""      # subjek/topik singkat
    status:            str  = ""      # berlaku / dicabut / diubah
    url_detail:        str  = ""
    url_pdf:           str  = ""
    path_pdf:          str  = ""
    teks_penuh:        str  = ""      # hasil ekstraksi PDF
    sha256_pdf:        str  = ""
    jumlah_halaman:    int  = 0
    jumlah_karakter:   int  = 0
    timestamp_scrape:  str  = ""
    error:             str  = ""      # kosong jika sukses


@dataclass
class ChunkDokumen:
    """Satu chunk teks untuk LLM corpus."""
    chunk_id:      str = ""
    doc_id:        str = ""
    jenis:         str = ""
    judul:         str = ""
    nomor:         str = ""
    tahun:         str = ""
    chunk_index:   int = 0
    total_chunks:  int = 0
    teks:          str = ""
    karakter:      int = 0


# ─── Utilitas ─────────────────────────────────────────────────────────────────
def safe_get(url: str, session: requests.Session,
             timeout: int = 30, stream: bool = False) -> Optional[requests.Response]:
    """GET dengan retry dan rate-limiting."""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout, stream=stream)
            resp.raise_for_status()
            time.sleep(DELAY_ANTAR_REQUEST)
            return resp
        except requests.RequestException as e:
            log.warning(f"[Attempt {attempt}/{MAX_RETRY}] GET {url} gagal: {e}")
            if attempt < MAX_RETRY:
                time.sleep(DELAY_ANTAR_REQUEST * attempt * 2)
    return None


def hash_url(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def bersihkan_teks(teks: str) -> str:
    """Membersihkan teks hasil OCR/ekstraksi PDF."""
    # Hapus karakter kontrol
    teks = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', teks)
    # Normalisasi whitespace berlebih
    teks = re.sub(r'\n{3,}', '\n\n', teks)
    teks = re.sub(r'[ \t]{2,}', ' ', teks)
    # Hapus header/footer berulang (nomor halaman, nama instansi)
    teks = re.sub(r'\n\s*\d+\s*\n', '\n', teks)
    return teks.strip()


def chunking_teks(teks: str, doc: DokumenHukum) -> list[ChunkDokumen]:
    """
    Chunking berbasis struktur hukum:
    Prioritas split pada batas pasal/ayat, baru fallback ke karakter.
    Menghasilkan list ChunkDokumen.
    """
    # Coba split pada batas pasal ("Pasal X", "BAB", "BAGIAN")
    pola_pasal = re.compile(
        r'(?=(?:Pasal\s+\d+|BAB\s+[IVXLCDM]+|BAGIAN\s+\w+))',
        re.IGNORECASE
    )
    segmen = pola_pasal.split(teks)
    segmen = [s.strip() for s in segmen if s.strip()]

    # Gabungkan segmen kecil, pecah segmen besar
    chunks_raw = []
    buffer = ""
    for seg in segmen:
        if len(buffer) + len(seg) < CHUNK_SIZE_CHARS:
            buffer += "\n" + seg
        else:
            if buffer:
                chunks_raw.append(buffer.strip())
            # Jika segmen sendiri terlalu besar, pecah per karakter dengan overlap
            if len(seg) > CHUNK_SIZE_CHARS:
                start = 0
                while start < len(seg):
                    end = min(start + CHUNK_SIZE_CHARS, len(seg))
                    chunks_raw.append(seg[start:end].strip())
                    start = end - CHUNK_OVERLAP_CHARS
            else:
                buffer = seg
    if buffer:
        chunks_raw.append(buffer.strip())

    # Buat objek ChunkDokumen
    result = []
    total = len(chunks_raw)
    for i, teks_chunk in enumerate(chunks_raw):
        if not teks_chunk:
            continue
        chunk = ChunkDokumen(
            chunk_id    = f"{doc.id}_c{i:04d}",
            doc_id      = doc.id,
            jenis       = doc.jenis,
            judul       = doc.judul,
            nomor       = doc.nomor,
            tahun       = doc.tahun,
            chunk_index = i,
            total_chunks= total,
            teks        = teks_chunk,
            karakter    = len(teks_chunk),
        )
        result.append(chunk)
    return result


# ─── Ekstraksi PDF ─────────────────────────────────────────────────────────────
def ekstrak_teks_pdf(path_pdf: str) -> tuple[str, int]:
    """
    Ekstraksi teks dari PDF menggunakan PyMuPDF.
    Mengembalikan (teks_bersih, jumlah_halaman).
    Jika PDF hasil scan (teks kosong), beri peringatan.
    """
    try:
        doc = fitz.open(path_pdf)
        halaman = len(doc)
        teks_halaman = []
        for page in doc:
            teks_halaman.append(page.get_text("text"))
        doc.close()
        teks = "\n".join(teks_halaman)
        teks = bersihkan_teks(teks)
        if len(teks.strip()) < 100:
            log.warning(f"PDF tampaknya hasil scan (teks minimal): {path_pdf}")
        return teks, halaman
    except Exception as e:
        log.error(f"Gagal ekstrak PDF {path_pdf}: {e}")
        return "", 0


# ─── Scraper: JDIH Sumut ──────────────────────────────────────────────────────
class JDIHSumutScraper:
    """
    Scraper untuk https://jdih.sumutprov.go.id
    Struktur: halaman daftar → halaman detail → unduh PDF
    """

    BASE = "https://jdih.sumutprov.go.id"

    def __init__(self, session: requests.Session, output_dir: Path):
        self.session = session
        self.output_dir = output_dir
        (output_dir / "pdf").mkdir(parents=True, exist_ok=True)

    def crawl_daftar(self, jenis: str = "perda",
                     tahun_mulai: int = 2000,
                     tahun_akhir: int = datetime.now().year,
                     maks_halaman: int = 50) -> list[DokumenHukum]:
        """
        Crawl halaman daftar produk hukum.
        Mengembalikan list DokumenHukum dengan metadata awal (tanpa teks penuh).
        """
        dokumen_list = []
        base_url = f"{self.BASE}/produk-hukum"
        halaman = 1

        console.print(f"\n[bold cyan]Crawling JDIH Sumut — jenis: {jenis.upper()}[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed} dokumen"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Mengambil daftar...", total=None)

            while halaman <= maks_halaman:
                params = {
                    "jenis":  jenis,
                    "page":   halaman,
                    "tahun_awal":  tahun_mulai,
                    "tahun_akhir": tahun_akhir,
                }
                url = f"{base_url}?" + urllib.parse.urlencode(params)
                resp = safe_get(url, self.session)
                if resp is None:
                    log.warning(f"Halaman {halaman} tidak dapat diakses, berhenti.")
                    break

                soup = BeautifulSoup(resp.text, "lxml")
                items = self._parse_daftar(soup, jenis)

                if not items:
                    log.info(f"Halaman {halaman}: tidak ada item, crawl selesai.")
                    break

                dokumen_list.extend(items)
                progress.update(task,
                    description=f"Halaman {halaman} — {len(dokumen_list)} dokumen",
                    completed=len(dokumen_list))

                # Cek apakah ada halaman berikutnya
                if not self._ada_halaman_berikutnya(soup):
                    break
                halaman += 1

        console.print(f"[green]✓ Total {len(dokumen_list)} dokumen ditemukan[/green]")
        return dokumen_list

    def _parse_daftar(self, soup: BeautifulSoup, jenis: str) -> list[DokumenHukum]:
        """Parse satu halaman daftar produk hukum JDIH Sumut."""
        hasil = []

        # Selector umum untuk daftar peraturan JDIH Sumut
        # (diperbarui jika struktur HTML berubah)
        items = (
            soup.select("div.produk-hukum-item") or
            soup.select("table.table tbody tr") or
            soup.select("ul.list-produk li") or
            soup.select("div.card")
        )

        for item in items:
            try:
                doc = DokumenHukum()
                doc.sumber = "jdih_sumut"
                doc.jenis  = jenis
                doc.timestamp_scrape = datetime.now().isoformat()

                # Ambil judul dan URL detail
                link = (
                    item.select_one("a.judul") or
                    item.select_one("h4 a") or
                    item.select_one("td:nth-child(2) a") or
                    item.select_one("a")
                )
                if link:
                    doc.judul = link.get_text(strip=True)
                    href = link.get("href", "")
                    doc.url_detail = href if href.startswith("http") else self.BASE + href

                # Nomor dan tahun dari judul (pola umum: "Perda No. 5 Tahun 2023")
                m = re.search(r'[Nn]o(?:mor)?\.?\s*(\d+)\s*[Tt]ahun\s*(\d{4})', doc.judul)
                if m:
                    doc.nomor = m.group(1)
                    doc.tahun = m.group(2)

                # Tentang
                tentang_el = (
                    item.select_one("p.tentang") or
                    item.select_one("td:nth-child(3)")
                )
                if tentang_el:
                    doc.tentang = tentang_el.get_text(strip=True)

                # Tanggal
                tgl_el = item.select_one("span.tanggal, td.tanggal, .date")
                if tgl_el:
                    doc.tanggal_ditetapkan = tgl_el.get_text(strip=True)

                # Status
                status_el = item.select_one("span.status, .badge")
                if status_el:
                    doc.status = status_el.get_text(strip=True).lower()
                else:
                    doc.status = "berlaku"  # default

                if doc.url_detail:
                    doc.id = hash_url(doc.url_detail)
                    hasil.append(doc)

            except Exception as e:
                log.debug(f"Gagal parse item: {e}")

        return hasil

    def _ada_halaman_berikutnya(self, soup: BeautifulSoup) -> bool:
        """Cek apakah ada link halaman berikutnya."""
        pagination = soup.select_one("ul.pagination")
        if not pagination:
            return False
        next_btn = pagination.select_one("li.next:not(.disabled), a[rel='next']")
        return next_btn is not None

    def ambil_detail_dan_pdf(self, doc: DokumenHukum) -> DokumenHukum:
        """Kunjungi halaman detail untuk mendapatkan URL PDF, lalu unduh."""
        if not doc.url_detail:
            doc.error = "Tidak ada URL detail"
            return doc

        resp = safe_get(doc.url_detail, self.session)
        if resp is None:
            doc.error = "Gagal mengakses halaman detail"
            return doc

        soup = BeautifulSoup(resp.text, "lxml")

        # Cari link PDF
        pdf_link = (
            soup.select_one("a[href$='.pdf']") or
            soup.select_one("a.btn-download") or
            soup.select_one("a[href*='download']") or
            soup.select_one("a[href*='/pdf/']")
        )
        if pdf_link:
            href = pdf_link.get("href", "")
            doc.url_pdf = href if href.startswith("http") else self.BASE + href

        # Perkaya metadata dari halaman detail
        if not doc.tentang:
            tentang_el = soup.select_one("h1, h2, .judul-dokumen, .document-title")
            if tentang_el:
                doc.tentang = tentang_el.get_text(strip=True)

        tgl_ditetapkan = soup.select_one(".tanggal-ditetapkan, [data-field='tanggal_ditetapkan']")
        if tgl_ditetapkan:
            doc.tanggal_ditetapkan = tgl_ditetapkan.get_text(strip=True)

        tgl_diundangkan = soup.select_one(".tanggal-diundangkan, [data-field='tanggal_diundangkan']")
        if tgl_diundangkan:
            doc.tanggal_diundangkan = tgl_diundangkan.get_text(strip=True)

        status_el = soup.select_one(".status-peraturan, .badge-status")
        if status_el:
            doc.status = status_el.get_text(strip=True).lower()

        # Unduh PDF
        if doc.url_pdf:
            doc = self._unduh_pdf(doc)

        return doc

    def _unduh_pdf(self, doc: DokumenHukum) -> DokumenHukum:
        """Unduh file PDF dan simpan ke disk."""
        nama_file = f"{doc.jenis}_{doc.nomor or 'xx'}_{doc.tahun or 'xxxx'}_{doc.id}.pdf"
        path = self.output_dir / "pdf" / nama_file

        # Lewati jika sudah ada dan valid
        if path.exists() and path.stat().st_size > 1024:
            doc.path_pdf = str(path)
            doc.sha256_pdf = sha256_file(str(path))
            log.debug(f"PDF sudah ada, dilewati: {nama_file}")
            return doc

        resp = safe_get(doc.url_pdf, self.session, timeout=60, stream=True)
        if resp is None:
            doc.error = f"Gagal unduh PDF: {doc.url_pdf}"
            return doc

        try:
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            doc.path_pdf  = str(path)
            doc.sha256_pdf = sha256_file(str(path))
            log.info(f"PDF diunduh: {nama_file} ({path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            doc.error = f"Gagal simpan PDF: {e}"
            path.unlink(missing_ok=True)

        return doc


# ─── Scraper: BPK (fallback) ──────────────────────────────────────────────────
class BPKScraper:
    """
    Scraper untuk https://peraturan.bpk.go.id
    Digunakan sebagai fallback / sumber pelengkap untuk Perda Sumut.
    Kode propinsi Sumatera Utara = 12
    """

    BASE = "https://peraturan.bpk.go.id"

    def __init__(self, session: requests.Session, output_dir: Path):
        self.session = session
        self.output_dir = output_dir
        (output_dir / "pdf").mkdir(parents=True, exist_ok=True)

    def crawl_daftar(self, jenis_kode: str = "perda",
                     tahun: Optional[int] = None,
                     maks_halaman: int = 30) -> list[DokumenHukum]:
        """Crawl daftar peraturan Sumatera Utara dari database BPK."""
        dokumen_list = []
        kode_propinsi = "12"  # Sumatera Utara
        halaman = 1

        console.print(f"\n[bold yellow]Crawling BPK (Sumut) — jenis: {jenis_kode.upper()}[/bold yellow]")

        while halaman <= maks_halaman:
            params = {
                "kodePropinsi": kode_propinsi,
                "jenis":        jenis_kode,
                "page":         halaman,
            }
            if tahun:
                params["tahun"] = tahun

            url = f"{self.BASE}/Home/Peraturan?" + urllib.parse.urlencode(params)
            resp = safe_get(url, self.session)
            if resp is None:
                break

            soup = BeautifulSoup(resp.text, "lxml")
            items = self._parse_daftar(soup, jenis_kode)
            if not items:
                break

            dokumen_list.extend(items)
            log.info(f"BPK halaman {halaman}: +{len(items)} ({len(dokumen_list)} total)")

            next_page = soup.select_one("a[aria-label='Next']")
            if not next_page:
                break
            halaman += 1

        console.print(f"[green]✓ BPK: {len(dokumen_list)} dokumen[/green]")
        return dokumen_list

    def _parse_daftar(self, soup: BeautifulSoup, jenis: str) -> list[DokumenHukum]:
        hasil = []
        rows = soup.select("table.table tbody tr")
        for row in rows:
            cols = row.select("td")
            if len(cols) < 3:
                continue
            try:
                doc = DokumenHukum()
                doc.sumber = "bpk"
                doc.jenis  = jenis
                doc.timestamp_scrape = datetime.now().isoformat()

                link = cols[1].select_one("a") if len(cols) > 1 else None
                if link:
                    doc.judul = link.get_text(strip=True)
                    href = link.get("href", "")
                    doc.url_detail = href if href.startswith("http") else self.BASE + href
                    doc.id = hash_url(doc.url_detail)

                # Nomor & tahun
                m = re.search(r'[Nn]o\.?\s*(\d+)\s*[Tt]ahun\s*(\d{4})', doc.judul)
                if m:
                    doc.nomor = m.group(1)
                    doc.tahun = m.group(2)

                doc.tentang = cols[2].get_text(strip=True) if len(cols) > 2 else ""
                if len(cols) > 3:
                    doc.tanggal_ditetapkan = cols[3].get_text(strip=True)
                doc.status = "berlaku"

                if doc.url_detail:
                    hasil.append(doc)
            except Exception as e:
                log.debug(f"BPK parse error: {e}")
        return hasil

    def ambil_pdf(self, doc: DokumenHukum) -> DokumenHukum:
        """Ambil URL PDF dari halaman detail BPK lalu unduh."""
        if not doc.url_detail:
            return doc
        resp = safe_get(doc.url_detail, self.session)
        if resp is None:
            doc.error = "Gagal akses detail BPK"
            return doc

        soup = BeautifulSoup(resp.text, "lxml")
        pdf_link = soup.select_one("a.btn[href$='.pdf'], a[href*='/download/'], a[title='Download']")
        if pdf_link:
            href = pdf_link.get("href", "")
            doc.url_pdf = href if href.startswith("http") else self.BASE + href

        if doc.url_pdf:
            nama = f"bpk_{doc.jenis}_{doc.nomor or 'x'}_{doc.tahun or 'x'}_{doc.id}.pdf"
            path = self.output_dir / "pdf" / nama
            if not (path.exists() and path.stat().st_size > 1024):
                resp_pdf = safe_get(doc.url_pdf, self.session, timeout=60, stream=True)
                if resp_pdf:
                    with open(path, "wb") as f:
                        for chunk in resp_pdf.iter_content(8192):
                            f.write(chunk)
            if path.exists():
                doc.path_pdf   = str(path)
                doc.sha256_pdf = sha256_file(str(path))

        return doc


# ─── Penyimpanan Output ───────────────────────────────────────────────────────
class DatabaseBuilder:
    """Membangun corpus LLM dari daftar DokumenHukum yang sudah diproses."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

    def proses_dokumen(self, dokumen_list: list[DokumenHukum]) -> list[ChunkDokumen]:
        """Ekstrak teks dari semua PDF dan lakukan chunking."""
        all_chunks = []

        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console
        ) as prog:
            task = prog.add_task("Ekstraksi & chunking PDF...", total=len(dokumen_list))

            for doc in dokumen_list:
                prog.update(task,
                    description=f"Proses: {doc.judul[:50]}...",
                    advance=1)

                if not doc.path_pdf or not Path(doc.path_pdf).exists():
                    doc.error = doc.error or "PDF tidak tersedia"
                    continue

                teks, halaman = ekstrak_teks_pdf(doc.path_pdf)
                doc.teks_penuh      = teks
                doc.jumlah_halaman  = halaman
                doc.jumlah_karakter = len(teks)

                if teks:
                    chunks = chunking_teks(teks, doc)
                    all_chunks.extend(chunks)

        return all_chunks

    def simpan_metadata_csv(self, dokumen_list: list[DokumenHukum]) -> Path:
        """Simpan metadata semua dokumen ke CSV."""
        path = self.output_dir / "metadata_dokumen.csv"
        fieldnames = [
            "id", "sumber", "jenis", "judul", "nomor", "tahun",
            "tanggal_ditetapkan", "tentang", "status",
            "url_detail", "url_pdf", "path_pdf",
            "sha256_pdf", "jumlah_halaman", "jumlah_karakter",
            "timestamp_scrape", "error"
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for doc in dokumen_list:
                row = asdict(doc)
                row.pop("teks_penuh", None)  # teks disimpan terpisah
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        console.print(f"[green]✓ Metadata CSV: {path}[/green]")
        return path

    def simpan_corpus_jsonl(self, dokumen_list: list[DokumenHukum]) -> Path:
        """
        Simpan corpus teks penuh dalam format JSONL.
        Format ini standar untuk fine-tuning LLM.
        Setiap baris = satu dokumen dengan teks penuh.
        """
        path = self.output_dir / "corpus_teks_penuh.jsonl"
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for doc in dokumen_list:
                if not doc.teks_penuh:
                    continue
                record = {
                    "id":       doc.id,
                    "jenis":    doc.jenis,
                    "judul":    doc.judul,
                    "nomor":    doc.nomor,
                    "tahun":    doc.tahun,
                    "tentang":  doc.tentang,
                    "status":   doc.status,
                    "sumber":   doc.sumber,
                    "teks":     doc.teks_penuh,
                    "metadata": {
                        "url":       doc.url_detail,
                        "halaman":   doc.jumlah_halaman,
                        "karakter":  doc.jumlah_karakter,
                        "sha256":    doc.sha256_pdf,
                        "timestamp": doc.timestamp_scrape,
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        console.print(f"[green]✓ Corpus JSONL ({count} dokumen): {path}[/green]")
        return path

    def simpan_chunks_jsonl(self, chunks: list[ChunkDokumen]) -> Path:
        """
        Simpan chunks dalam format JSONL — siap untuk RAG indexing.
        Setiap baris = satu chunk dengan metadata lengkap.
        """
        path = self.output_dir / "corpus_chunks_rag.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        console.print(f"[green]✓ RAG Chunks JSONL ({len(chunks)} chunks): {path}[/green]")
        return path

    def simpan_ringkasan(self, dokumen_list: list[DokumenHukum],
                          chunks: list[ChunkDokumen]) -> Path:
        """Simpan laporan ringkasan scraping."""
        path = self.output_dir / "ringkasan_scraping.json"

        # Hitung statistik
        berhasil = [d for d in dokumen_list if d.teks_penuh]
        gagal    = [d for d in dokumen_list if d.error]
        per_jenis = {}
        for d in berhasil:
            per_jenis[d.jenis] = per_jenis.get(d.jenis, 0) + 1

        total_karakter = sum(d.jumlah_karakter for d in berhasil)
        total_halaman  = sum(d.jumlah_halaman  for d in berhasil)

        ringkasan = {
            "timestamp":         datetime.now().isoformat(),
            "total_dokumen":     len(dokumen_list),
            "berhasil":          len(berhasil),
            "gagal":             len(gagal),
            "per_jenis":         per_jenis,
            "total_halaman_pdf": total_halaman,
            "total_karakter":    total_karakter,
            "total_chunks_rag":  len(chunks),
            "rata_karakter_per_chunk": (
                total_karakter // len(chunks) if chunks else 0
            ),
            "error_summary": [
                {"id": d.id, "judul": d.judul, "error": d.error}
                for d in gagal
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(ringkasan, f, ensure_ascii=False, indent=2)

        return path, ringkasan


# ─── CLI Orchestrator ─────────────────────────────────────────────────────────
def tampilkan_banner():
    console.print(Panel.fit(
        "[bold white]JDIH Sumatera Utara — LLM Database Builder[/bold white]\n"
        "[dim]Scraping · Ekstraksi PDF · Chunking · Export JSONL[/dim]\n"
        "[cyan]Fauzan Nur Ahmadi · Universitas Sumatera Utara[/cyan]",
        border_style="bold blue",
        padding=(1, 4)
    ))


def tampilkan_ringkasan(ringkasan: dict):
    tbl = Table(title="Ringkasan Scraping", show_header=True,
                header_style="bold cyan", border_style="blue")
    tbl.add_column("Metrik", style="white", width=30)
    tbl.add_column("Nilai",  style="bold green", justify="right")
    tbl.add_row("Total Dokumen Ditemukan", str(ringkasan["total_dokumen"]))
    tbl.add_row("Berhasil Diproses",       str(ringkasan["berhasil"]))
    tbl.add_row("Gagal",                   str(ringkasan["gagal"]))
    tbl.add_row("Total Halaman PDF",       str(ringkasan["total_halaman_pdf"]))
    tbl.add_row("Total Karakter Teks",     f"{ringkasan['total_karakter']:,}")
    tbl.add_row("Total Chunks (RAG)",      str(ringkasan["total_chunks_rag"]))
    tbl.add_row("Rata-rata Karakter/Chunk",f"{ringkasan['rata_karakter_per_chunk']:,}")
    for jenis, jumlah in ringkasan["per_jenis"].items():
        tbl.add_row(f"  • {JENIS_PERATURAN.get(jenis, jenis)}", str(jumlah))
    console.print(tbl)


def main():
    parser = argparse.ArgumentParser(
        description="JDIH Sumatera Utara LLM Corpus Builder"
    )
    parser.add_argument(
        "--sumber", choices=["jdih_sumut", "bpk", "keduanya"],
        default="jdih_sumut",
        help="Sumber scraping (default: jdih_sumut)"
    )
    parser.add_argument(
        "--jenis", nargs="+",
        default=["perda", "pergub"],
        choices=list(JENIS_PERATURAN.keys()),
        help="Jenis peraturan yang di-scrape"
    )
    parser.add_argument(
        "--tahun-mulai", type=int, default=2000,
        help="Tahun awal filter (default: 2000)"
    )
    parser.add_argument(
        "--tahun-akhir", type=int, default=datetime.now().year,
        help=f"Tahun akhir filter (default: {datetime.now().year})"
    )
    parser.add_argument(
        "--maks-halaman", type=int, default=10,
        help="Maks halaman daftar per jenis (default: 10, 1 halaman ≈ 20 dokumen)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./jdih_corpus",
        help="Direktori output (default: ./jdih_corpus)"
    )
    parser.add_argument(
        "--skip-pdf", action="store_true",
        help="Lewati pengunduhan PDF (hanya metadata)"
    )
    parser.add_argument(
        "--mode-uji", action="store_true",
        help="Mode uji: hanya 3 dokumen per jenis"
    )
    args = parser.parse_args()

    tampilkan_banner()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    semua_dokumen = []

    # ── Crawl Daftar ──
    for jenis in args.jenis:
        maks = 1 if args.mode_uji else args.maks_halaman

        if args.sumber in ("jdih_sumut", "keduanya"):
            scraper = JDIHSumutScraper(session, output_dir)
            daftar = scraper.crawl_daftar(
                jenis        = jenis,
                tahun_mulai  = args.tahun_mulai,
                tahun_akhir  = args.tahun_akhir,
                maks_halaman = maks
            )
            if args.mode_uji:
                daftar = daftar[:3]

            if not args.skip_pdf:
                console.print(f"\n[bold]Mengunduh PDF & detail untuk {len(daftar)} dokumen...[/bold]")
                for doc in tqdm(daftar, desc=f"Detail+PDF {jenis}"):
                    scraper.ambil_detail_dan_pdf(doc)

            semua_dokumen.extend(daftar)

        if args.sumber in ("bpk", "keduanya"):
            scraper_bpk = BPKScraper(session, output_dir)
            daftar_bpk  = scraper_bpk.crawl_daftar(jenis_kode=jenis, maks_halaman=maks)
            if args.mode_uji:
                daftar_bpk = daftar_bpk[:3]

            if not args.skip_pdf:
                for doc in tqdm(daftar_bpk, desc=f"BPK PDF {jenis}"):
                    scraper_bpk.ambil_pdf(doc)

            semua_dokumen.extend(daftar_bpk)

    if not semua_dokumen:
        console.print("[red]Tidak ada dokumen yang berhasil di-crawl.[/red]")
        return

    console.print(f"\n[bold]Total dokumen: {len(semua_dokumen)}[/bold]")

    # ── Bangun Database ──
    builder = DatabaseBuilder(output_dir)

    if not args.skip_pdf:
        chunks = builder.proses_dokumen(semua_dokumen)
    else:
        chunks = []

    builder.simpan_metadata_csv(semua_dokumen)
    builder.simpan_corpus_jsonl(semua_dokumen)
    if chunks:
        builder.simpan_chunks_jsonl(chunks)

    _, ringkasan = builder.simpan_ringkasan(semua_dokumen, chunks)

    console.print("\n")
    tampilkan_ringkasan(ringkasan)
    console.print(f"\n[bold green]✓ Selesai! Output tersimpan di: {output_dir.resolve()}[/bold green]")
    console.print("[dim]File yang dihasilkan:[/dim]")
    console.print(f"  [cyan]metadata_dokumen.csv[/cyan]      — metadata semua dokumen")
    console.print(f"  [cyan]corpus_teks_penuh.jsonl[/cyan]   — corpus teks penuh (fine-tuning)")
    console.print(f"  [cyan]corpus_chunks_rag.jsonl[/cyan]   — chunks siap RAG indexing")
    console.print(f"  [cyan]ringkasan_scraping.json[/cyan]   — statistik & laporan")
    console.print(f"  [cyan]pdf/[/cyan]                      — seluruh file PDF asli")


if __name__ == "__main__":
    main()
