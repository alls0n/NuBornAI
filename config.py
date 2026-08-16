

SOURCES = [

    "documents/newbornnotes.pdf",
    "documents/caring-for-your-newborn-handbook.pdf", 
    "documents/Healthy Baby Guide BMC.pdf", 
    "documents/newbornhandbookcoverbooklet.pdf"
]

EMBED_MODEL = "all-mpnet-base-v2"   # sentence-transformers model for dense retrieval
OLLAMA_MODEL = "mistral"            # local model served via `ollama run mistral`

INDEX_PATH = "data/newborn_rag_index.faiss"
META_PATH = "data/newborn_rag_meta.pkl"

FORCE_REBUILD = False   # True will re-parse SOURCES to rebuild the index 
                         # False will load INDEX_PATH / META_PATH from disk 

CHUNK_WORDS = 300
OVERLAP_WORDS = 50

TOP_K = 8     # candidates considered before final trimming
FINAL_K = 5   # chunks actually sent to the model as context
RRF_K = 60    # reciprocal rank fusion constant

# cross-encoder reranking 
USE_RERANKER = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 20   # how many hybrid candidates to feed the reranker 

EVAL_K = 5    # top-k used when scoring retrieval quality in eval/evaluate.py

SYSTEM_PROMPT = """\
You are a warm, careful assistant for new parents of a newborn. Answer using
ONLY the document excerpts provided below.

Rules:
1. If the excerpts contain the answer, explain it clearly and in plain language.
2. Cite the source in brackets, e.g. [some_article.pdf].
3. If the answer is NOT in the excerpts, say so plainly: "I couldn't find this
   in the documents provided — please check with your pediatrician."
4. Never guess or make up medical facts, dosages, or numbers.
5. For anything urgent-sounding (fever, breathing trouble, injury), remind the
   parent this is not a substitute for medical care and to contact their
   pediatrician or emergency services if concerned.
"""
