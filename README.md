# JDIH Sumatera Utara — LLM Database Builder
## Panduan Lengkap Penggunaan

**Penulis:** Fauzan Nur Ahmadi, S.Kom., M.Cs., Universitas Sumatera Utara  
**Tujuan:** Membangun corpus hukum dari JDIH Sumatera Utara untuk pelatihan dan retrieval LLM

---

## Gambaran Sistem

Aplikasi ini terdiri dari dua skrip utama yang bekerja secara berurutan:

```
jdih_sumut_scraper.py   →   jdih_rag_indexer.py
       │                            │
  Crawl + Download PDF        Embedding + FAISS
  Ekstraksi Teks               Index RAG siap pakai
  Export JSONL/CSV
```

---

## Instalasi

```bash
# 1. Buat virtual environment (disarankan)
python3 -m venv env
source env/bin/activate          # Linux/Mac
env\Scripts\activate             # Windows

# 2. Install dependensi
pip install -r requirements.txt
```

---

## Langkah 1: Scraping & Ekstraksi (jdih_sumut_scraper.py)

### Mode Uji (3 dokumen saja — untuk verifikasi)

```bash
python jdih_sumut_scraper.py --mode-uji
```

### Scraping penuh Perda + Pergub (semua tahun)

```bash
python jdih_sumut_scraper.py \
    --sumber jdih_sumut \
    --jenis perda pergub \
    --tahun-mulai 2000 \
    --tahun-akhir 2025 \
    --maks-halaman 50 \
    --output-dir ./jdih_corpus
```

### Scraping dari kedua sumber (JDIH Sumut + BPK)

```bash
python jdih_sumut_scraper.py \
    --sumber keduanya \
    --jenis perda pergub sk \
    --maks-halaman 30 \
    --output-dir ./jdih_corpus_lengkap
```

### Hanya metadata (tanpa unduh PDF)

```bash
python jdih_sumut_scraper.py --skip-pdf --jenis perda
```

### Argumen Lengkap

| Argumen | Default | Keterangan |
|---|---|---|
| `--sumber` | `jdih_sumut` | `jdih_sumut` / `bpk` / `keduanya` |
| `--jenis` | `perda pergub` | `perda` `pergub` `sk` `se` `instruksi` |
| `--tahun-mulai` | 2000 | Tahun awal filter |
| `--tahun-akhir` | tahun ini | Tahun akhir filter |
| `--maks-halaman` | 10 | Maks halaman list (≈20 dok/halaman) |
| `--output-dir` | `./jdih_corpus` | Direktori output |
| `--skip-pdf` | False | Lewati download PDF |
| `--mode-uji` | False | Hanya 3 dok/jenis |

---

## Struktur Output Langkah 1

```
jdih_corpus/
├── pdf/                          ← File PDF asli
│   ├── perda_5_2023_abc123.pdf
│   └── pergub_12_2022_def456.pdf
│
├── metadata_dokumen.csv          ← Metadata semua dokumen
├── corpus_teks_penuh.jsonl       ← Teks penuh (untuk fine-tuning)
├── corpus_chunks_rag.jsonl       ← Chunks siap RAG
└── ringkasan_scraping.json       ← Statistik scraping
```

### Format corpus_teks_penuh.jsonl (fine-tuning LLM)

```json
{
  "id": "abc123",
  "jenis": "perda",
  "judul": "Peraturan Daerah Provinsi Sumatera Utara Nomor 5 Tahun 2023",
  "nomor": "5",
  "tahun": "2023",
  "tentang": "Retribusi Daerah",
  "status": "berlaku",
  "sumber": "jdih_sumut",
  "teks": "BAB I KETENTUAN UMUM\n\nPasal 1\n...",
  "metadata": {
    "url": "https://jdih.sumutprov.go.id/...",
    "halaman": 45,
    "karakter": 87234,
    "sha256": "a1b2c3...",
    "timestamp": "2025-05-15T10:30:00"
  }
}
```

### Format corpus_chunks_rag.jsonl (RAG indexing)

```json
{
  "chunk_id": "abc123_c0042",
  "doc_id": "abc123",
  "jenis": "perda",
  "judul": "Peraturan Daerah... Nomor 5 Tahun 2023",
  "nomor": "5",
  "tahun": "2023",
  "chunk_index": 42,
  "total_chunks": 87,
  "teks": "Pasal 15\nSanksi administratif bagi...",
  "karakter": 1487
}
```

---

## Langkah 2: Membangun FAISS Index (jdih_rag_indexer.py)

```bash
# Install dependensi tambahan
pip install sentence-transformers faiss-cpu

# Build index
python jdih_rag_indexer.py \
    --chunks ./jdih_corpus/corpus_chunks_rag.jsonl \
    --output ./jdih_corpus/faiss_index \
    --model LazarusNLP/IndoNanoSE-Base

# Test retrieval setelah selesai
python jdih_rag_indexer.py \
    --chunks ./jdih_corpus/corpus_chunks_rag.jsonl \
    --output ./jdih_corpus/faiss_index \
    --test-query "sanksi pidana pelanggaran retribusi daerah"
```

---

## Langkah 3: Integrasi dengan LLM (RAG Pipeline)

```python
from jdih_rag_indexer import JDIHRetriever

# Inisialisasi retriever
retriever = JDIHRetriever("./jdih_corpus/faiss_index")

# Cari chunk relevan untuk query
def cari_pasal_relevan(query: str, jenis_filter=None):
    return retriever.cari(query, top_k=5, filter_jenis=jenis_filter)

# Contoh: deteksi konflik — cari pasal terkait sanksi
hasil = cari_pasal_relevan(
    "batas maksimum sanksi pidana dalam Perda",
    jenis_filter=["perda"]
)

# Gunakan hasil sebagai konteks dalam prompt LLM
konteks = "\n\n---\n\n".join([
    f"[{r['jenis'].upper()} No.{r['nomor']}/{r['tahun']}]\n{r['teks_preview']}"
    for r in hasil
])

prompt = f"""Anda adalah asisten analisis hukum. Berdasarkan dokumen berikut:

{konteks}

Pertanyaan: {query}
Analisis apakah terdapat konflik normatif dengan UU No. 12 Tahun 2011 Pasal 15.
"""

# Kirimkan prompt ke LLM pilihan Anda
```

---

## Catatan Penting

### Etika Scraping
- Scraper menggunakan delay 1.5 detik antar request untuk menghormati server JDIH.
- Jangan menjalankan beberapa instance scraper secara bersamaan.
- Corpus yang dibangun untuk keperluan riset akademik sesuai doktrin fair use.

### PDF Scan vs PDF Digital
- Banyak Perda lama tersedia hanya sebagai PDF hasil scan (gambar).
- Untuk PDF scan, ekstraksi teks otomatis akan menghasilkan teks kosong.
- Solusi: integrasikan OCR (pytesseract + Tesseract dengan bahasa Indonesia).
- Peringatan PDF scan akan muncul di log: `jdih_scraper.log`

### Pemilihan Model Embedding
- `LazarusNLP/IndoNanoSE-Base` — direkomendasikan, terlatih untuk Bahasa Indonesia
- `paraphrase-multilingual-MiniLM-L12-v2` — fallback jika model Indo tidak tersedia
- Hindari model berbasis Bahasa Inggris murni untuk teks hukum Indonesia

### Pembaruan Berkala
Untuk menjaga corpus tetap mutakhir, jadwalkan scraping berkala:
```bash
# Contoh crontab: scraping setiap Senin dini hari
0 2 * * 1 cd /path/to/project && python jdih_sumut_scraper.py \
    --tahun-mulai $(date +%Y) --maks-halaman 5 >> scraping_cron.log 2>&1
```

---

## Troubleshooting

**Error: Connection refused / Timeout**  
JDIH Sumut kadang tidak responsif. Tingkatkan delay: edit `DELAY_ANTAR_REQUEST = 3.0`

**PDF terunduh tapi teks kosong**  
PDF adalah hasil scan. Aktifkan OCR dengan pytesseract.

**ImportError: No module named 'fitz'**  
`pip install PyMuPDF --break-system-packages`

**FAISS index lambat untuk corpus besar (>50k chunks)**  
Ganti ke `IndexIVFFlat` dengan `nlist=256` dan gunakan GPU dengan `faiss-gpu`.
