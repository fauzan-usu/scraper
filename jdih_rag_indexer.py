"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  JDIH Corpus → FAISS Vector Index Builder                                  ║
║  Langkah lanjutan setelah jdih_sumut_scraper.py                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Skrip ini:
  1. Membaca corpus_chunks_rag.jsonl yang dihasilkan scraper
  2. Membuat embedding menggunakan sentence-transformers
     (model: "LazarusNLP/IndoNanoSE-Base" — multilingual, bagus untuk Indo)
  3. Membangun FAISS index dan menyimpannya ke disk
  4. Menyediakan fungsi retrieval untuk integrasi RAG

Instalasi dependensi:
  pip install sentence-transformers faiss-cpu --break-system-packages

Catatan:
  Untuk GPU, gunakan faiss-gpu daripada faiss-cpu.
  Jika server terbatas, ganti model ke paraphrase-multilingual-MiniLM-L12-v2
"""

import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def build_faiss_index(
    chunks_jsonl_path: str = "./jdih_corpus/corpus_chunks_rag.jsonl",
    output_dir:        str = "./jdih_corpus/faiss_index",
    model_name:        str = "LazarusNLP/IndoNanoSE-Base",
    batch_size:        int = 64
):
    """
    Membangun FAISS index dari corpus chunks JDIH.

    Parameter
    ---------
    chunks_jsonl_path : path ke file corpus_chunks_rag.jsonl
    output_dir        : direktori untuk menyimpan index
    model_name        : nama model embedding (HuggingFace)
    batch_size        : ukuran batch untuk encoding

    File Output
    -----------
    faiss_index/
    ├── index.faiss       — FAISS index (IVF Flat untuk recall tinggi)
    ├── metadata.pkl      — metadata setiap chunk (id, judul, nomor, dst)
    └── config.json       — konfigurasi index
    """

    # Impor dengan informasi yang jelas jika belum terinstal
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Jalankan dulu:\n"
            "  pip install sentence-transformers faiss-cpu --break-system-packages"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Muat chunks ──
    print(f"Memuat chunks dari {chunks_jsonl_path}...")
    chunks = []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))

    if not chunks:
        raise ValueError("Tidak ada chunks ditemukan di JSONL!")

    print(f"  → {len(chunks)} chunks dimuat")
    teks_list = [c["teks"] for c in chunks]

    # ── Buat embedding ──
    print(f"Memuat model embedding: {model_name}")
    model = SentenceTransformer(model_name)

    print("Membuat embedding (proses ini memakan waktu)...")
    embeddings = model.encode(
        teks_list,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True   # normalisasi untuk cosine similarity
    )
    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    print(f"  → Dimensi embedding: {dim}")

    # ── Bangun FAISS index ──
    # Gunakan IndexFlatIP (Inner Product = cosine similarity setelah normalisasi)
    # untuk corpus kecil (<100k). Untuk corpus besar, pakai IndexIVFFlat.
    if len(chunks) < 10000:
        index = faiss.IndexFlatIP(dim)
    else:
        # IVF dengan 100 cluster untuk corpus menengah
        nlist = min(100, len(chunks) // 39)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)

    index.add(embeddings)
    print(f"  → Index dibangun: {index.ntotal} vektor")

    # ── Simpan ke disk ──
    faiss_path = output_path / "index.faiss"
    faiss.write_index(index, str(faiss_path))

    meta_path = output_path / "metadata.pkl"
    metadata = [
        {
            "chunk_id":    c["chunk_id"],
            "doc_id":      c["doc_id"],
            "jenis":       c["jenis"],
            "judul":       c["judul"],
            "nomor":       c["nomor"],
            "tahun":       c["tahun"],
            "chunk_index": c["chunk_index"],
            "total_chunks":c["total_chunks"],
            "karakter":    c["karakter"],
            # Simpan juga potongan teks untuk snippet
            "teks_preview": c["teks"][:300],
        }
        for c in chunks
    ]
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    config = {
        "model_name":   model_name,
        "embedding_dim":dim,
        "total_chunks": len(chunks),
        "index_type":   type(index).__name__,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✓ FAISS index tersimpan di: {output_path}")
    return str(faiss_path)


class JDIHRetriever:
    """
    Retriever untuk JDIH corpus menggunakan FAISS.
    Digunakan dalam pipeline RAG.

    Contoh penggunaan:
    ------------------
    retriever = JDIHRetriever("./jdih_corpus/faiss_index")
    hasil = retriever.cari("sanksi pelanggaran retribusi daerah", top_k=5)
    for r in hasil:
        print(r["skor"], r["judul"], r["teks_preview"])
    """

    def __init__(self, index_dir: str):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Jalankan: pip install sentence-transformers faiss-cpu")

        index_path = Path(index_dir)

        with open(index_path / "config.json") as f:
            self.config = json.load(f)

        print(f"Memuat model: {self.config['model_name']}...")
        self.model = SentenceTransformer(self.config["model_name"])
        self.index = faiss.read_index(str(index_path / "index.faiss"))

        with open(index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        print(f"✓ Retriever siap — {self.config['total_chunks']} chunks terindeks")

    def cari(self, query: str, top_k: int = 5,
              filter_jenis: Optional[list] = None) -> list[dict]:
        """
        Cari chunks yang relevan dengan query.

        Parameter
        ---------
        query        : pertanyaan atau teks pasal yang ingin dibandingkan
        top_k        : jumlah hasil yang dikembalikan
        filter_jenis : filter berdasarkan jenis peraturan ['perda', 'pergub', dll]

        Return
        ------
        list of dict dengan kunci: skor, chunk_id, doc_id, jenis, judul,
                                   nomor, tahun, teks_preview
        """
        import faiss
        import numpy as np

        vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Ambil lebih banyak dulu jika ada filter
        k_internal = top_k * 5 if filter_jenis else top_k
        k_internal = min(k_internal, self.index.ntotal)

        scores, indices = self.index.search(vec, k_internal)

        hasil = []
        for skor, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            if filter_jenis and meta["jenis"] not in filter_jenis:
                continue
            hasil.append({
                "skor":         float(skor),
                "chunk_id":     meta["chunk_id"],
                "doc_id":       meta["doc_id"],
                "jenis":        meta["jenis"],
                "judul":        meta["judul"],
                "nomor":        meta["nomor"],
                "tahun":        meta["tahun"],
                "chunk_index":  meta["chunk_index"],
                "total_chunks": meta["total_chunks"],
                "teks_preview": meta["teks_preview"],
            })
            if len(hasil) >= top_k:
                break

        return hasil


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index dari corpus JDIH")
    parser.add_argument("--chunks", default="./jdih_corpus/corpus_chunks_rag.jsonl")
    parser.add_argument("--output", default="./jdih_corpus/faiss_index")
    parser.add_argument("--model",  default="LazarusNLP/IndoNanoSE-Base")
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--test-query", type=str,
        help="Jalankan test retrieval setelah index selesai")
    args = parser.parse_args()

    build_faiss_index(
        chunks_jsonl_path=args.chunks,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch,
    )

    if args.test_query:
        print(f"\nTest retrieval: '{args.test_query}'")
        retriever = JDIHRetriever(args.output)
        results = retriever.cari(args.test_query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Skor: {r['skor']:.4f}")
            print(f"     {r['jenis'].upper()} No.{r['nomor']} Tahun {r['tahun']}")
            print(f"     {r['judul']}")
            print(f"     {r['teks_preview'][:200]}...")
