# app.py — local web server for NewbornAI.

from flask import Flask, jsonify, render_template, request

import config
from rag_engine import engine

app = Flask(__name__)


@app.route("/")
def index(): #main chat 
    return render_template("index.html")


@app.route("/api/status")
def status(): #reports statuts of the RAG engine 
    return jsonify({
        "ready": engine.ready,
        "error": engine.error,
        "log": engine.status_log,
        "sources": engine.sources_summary() if engine.ready else [],
        "chunk_count": engine.index.ntotal if engine.ready and engine.index else 0,
        "ollama_model": config.OLLAMA_MODEL,
        "reranker_model": config.RERANK_MODEL if engine.reranker is not None else None,
    })


@app.route("/api/chat", methods=["POST"])
def chat(): #handles the message from user 
    if not engine.ready:
        return jsonify({"error": "Index isn't ready yet."}), 503

    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"error": "Empty message."}), 400  #there needs to be input from user 

    try:
        result = engine.ask(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500 #bad query does not crash 


@app.route("/api/reset", methods=["POST"])
def reset(): #clearing conversation 
    engine.reset()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("loading sources and building/loading the index")
    engine.initialize()
    for line in engine.status_log:
        print(line)
    print("\nOpen http://127.0.0.1:5050 in your browser.") #the port you can open in brower to view frontend 
    app.run(host="127.0.0.1", port=5050, debug=False)
