# pip install pymupdf numpy sentence-transformers transformers torch

from __future__ import annotations
import re
import os
from dataclasses import dataclass
from typing import List, Dict
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch

PDF_PATH = "doencas_respiratorias_cronicas.pdf"

#modelo treinado em portugues
#MODEL_QA = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"

#modelo multilíngue 
#MODEL_QA = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"

#modelo treinado em ingles
MODEL_QA = "deepset/roberta-base-squad2"

MODEL_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TARGET_CHARS = 1200
OVERLAP_CHARS = 150
QA_MAX_LENGTH = 384
QA_STRIDE = 128
TOP_K = 3

HEADER_FOOTER_PATTERNS = [
    r"CADERNOS DE ATENÇÃO BÁSICA.*",
    r"Ministério da Saúde.*",
    r"^\s*\d+\s*$",
]
_re_header_footer = re.compile("|".join(f"(?:{p})" for p in HEADER_FOOTER_PATTERNS), re.IGNORECASE | re.MULTILINE)

RE_CHAPTER = re.compile(r"(?m)^\s*(\d{1,2})\s+[A-ZÁ-Ú][A-ZÁ-Ú0-9\-()/:,.; ]+$")
RE_SECTION = re.compile(r"(?m)^\s*(\d{1,2}\.\d+(?:\.\d+)?)\s+.+$")

@dataclass
class Block:
    title: str
    level: str
    content: str

def pdf_to_pages_text(path: str) -> List[str]:
    doc = fitz.open(path)
    pages = [doc[i].get_text("text") for i in range(len(doc))]
    doc.close()
    return pages

def clean_page_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = _re_header_footer.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def clean_document_pages(pages: List[str]) -> List[str]:
    return [clean_page_text(p) for p in pages]

def split_semantic_blocks(pages: List[str]) -> List[Block]:
    full = "\n".join(pages)
    markers = []
    for m in RE_CHAPTER.finditer(full):
        markers.append((m.start(), 'chapter', m.group(0).strip()))
    for m in RE_SECTION.finditer(full):
        markers.append((m.start(), 'section', m.group(0).strip()))
    markers.sort(key=lambda x: x[0])
    blocks: List[Block] = []
    if not markers:
        blocks.append(Block(title="DOCUMENTO", level="text", content=full))
        return blocks
    for i, (pos, level, title) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(full)
        blocks.append(Block(title=title, level=level, content=full[pos:end].strip()))
    return blocks

SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-ZÁ-Ú0-9])")

def split_sentences(text: str) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    return sents if sents else [text]

def pack_sentences_with_overlap(sents: List[str], target_chars: int, overlap_chars: int) -> List[str]:
    chunks, buf = [], ""
    for s in sents:
        if not buf:
            buf = s
            continue
        if len(buf) + 1 + len(s) <= target_chars:
            buf = f"{buf} {s}"
        else:
            chunks.append(buf)
            buf = (buf[-overlap_chars:] + " " + s).strip() if overlap_chars > 0 else s
    if buf:
        chunks.append(buf)
    return chunks

def make_chunks(pages: List[str]) -> List[str]:
    blocks = split_semantic_blocks(pages)
    chunks: List[str] = []
    for b in blocks:
        sents = split_sentences(b.content)
        packed = pack_sentences_with_overlap(sents, TARGET_CHARS, OVERLAP_CHARS)
        if packed:
            packed[0] = f"{b.title}\n\n" + packed[0]
        chunks.extend(packed)
    return [c.strip() for c in chunks if c.strip()]

def build_embedder(model_name: str = MODEL_EMB) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=("cuda" if torch.cuda.is_available() else "cpu"))

def embed_corpus(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def retrieve_top_k(embedder: SentenceTransformer, query: str, corpus_embs: np.ndarray, k: int) -> List[int]:
    q = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = q @ corpus_embs.T
    return np.argsort(sims)[::-1][:k].tolist()

def answer_with_windows(question: str, long_context: str, model_name: str = MODEL_QA) -> Dict:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    inputs = tokenizer(
        question,
        long_context,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=QA_MAX_LENGTH,
        stride=QA_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=False,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    best = {"score": -1.0}
    num_windows = inputs["input_ids"].shape[0]
    for i in range(num_windows):
        
        input_slice = {k: v[i].unsqueeze(0) for k, v in inputs.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
        with torch.no_grad():
            out = model(**input_slice)
        start_probs = torch.nn.functional.softmax(out.start_logits, dim=-1)
        end_probs = torch.nn.functional.softmax(out.end_logits, dim=-1)
        start_idx, end_idx = start_probs.argmax(dim=-1).item(), end_probs.argmax(dim=-1).item()
        if end_idx >= start_idx:
            score = (start_probs[0, start_idx] * end_probs[0, end_idx]).item()
            ans = tokenizer.decode(input_slice["input_ids"][0][start_idx:end_idx+1])
            if score > best["score"]:
                best = {"answer": ans, "score": score, "start": int(start_idx), "end": int(end_idx)}
    return best

def build_corpus_from_pdf(pdf_path: str) -> List[str]:
    return make_chunks(clean_document_pages(pdf_to_pages_text(pdf_path)))

def qa_over_pdf(pdf_path: str, question: str, top_k: int = TOP_K) -> Dict:
    corpus = build_corpus_from_pdf(pdf_path)
    embedder = build_embedder(MODEL_EMB)
    embs = embed_corpus(embedder, corpus)
    idxs = retrieve_top_k(embedder, question, embs, k=top_k)
    selected = [corpus[i] for i in idxs]
    selected_context = "\n\n".join(selected)
    result = answer_with_windows(question, selected_context, MODEL_QA)
    result["selected_chunk_ids"] = idxs
    result["selected_preview"] = [s[:200] + ("..." if len(s) > 200 else "") for s in selected]
    return result

if __name__ == "__main__":
    if os.path.exists(PDF_PATH):
        #question = "De que forma o diagnóstico da rinite alérgica é realizado e qual a importância do diagnóstico diferencial?"
        question = "Para que servem os medicamentos de atenção básica?"
        #question = "Como a asma é classificada de acordo com a gravidade?"
        out = qa_over_pdf(PDF_PATH, question, top_k=TOP_K)
        print("Resposta:", out.get("answer"))
        print("Score:", round(out.get("score", 0.0), 4))
        
        for i, prev in enumerate(out.get("selected_preview", [])):
            print(f"[{i}] {prev}")
    else:
        print(f"PDF não encontrado: {PDF_PATH}")
