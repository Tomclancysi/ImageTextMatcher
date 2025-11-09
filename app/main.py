import os

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for, abort, session

from .services.index_service import IndexService
from .services.text_correction_service import TextCorrectionService


def create_app() -> Flask:
    image_root = os.environ.get("ITM_IMAGE_ROOT", os.path.join(os.getcwd(), "data", "images"))
    index_dir = os.environ.get("ITM_INDEX_DIR", os.path.join(os.getcwd(), "data", "index"))
    model_name = os.environ.get("ITM_MODEL_NAME", "openai/clip-vit-base-patch32")

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
    app.config["INDEX_SERVICE"] = IndexService(image_root=image_root, index_dir=index_dir, model_name=model_name)
    app.config["TEXT_CORRECTION_SERVICE"] = TextCorrectionService(language="en")

    @app.get("/")
    def home():
        current_top_k = int(session.get("top_k", 10))
        return render_template("index.html", current_top_k=current_top_k)

    @app.post("/search")
    def search():
        query = request.form.get("query", "").strip()
        try:
            top_k = int(request.form.get("top_k", session.get("top_k", 10)))
        except ValueError:
            top_k = int(session.get("top_k", 10))

        session["top_k"] = max(1, min(100, top_k))

        if not query:
            return redirect(url_for("home"))

        # Text correction
        correction_service: TextCorrectionService = app.config["TEXT_CORRECTION_SERVICE"]
        corrected_query, suggestions = correction_service.correct_text(query)
        
        index_service: IndexService = app.config["INDEX_SERVICE"]
        results = index_service.search(query=corrected_query, top_k=session["top_k"])
        
        return render_template("results.html", 
                             query=query, 
                             corrected_query=corrected_query,
                             suggestions=suggestions,
                             results=results, 
                             current_top_k=session["top_k"])

    @app.get("/api/search")
    def api_search():
        query = request.args.get("q", "").strip()
        k_arg = request.args.get("k")
        if k_arg is not None:
            try:
                session["top_k"] = max(1, min(100, int(k_arg)))
            except ValueError:
                pass
        top_k = int(session.get("top_k", 10))
        if not query:
            return jsonify({"error": "missing q"}), 400
        
        # Text correction for API
        correction_service: TextCorrectionService = app.config["TEXT_CORRECTION_SERVICE"]
        corrected_query, suggestions = correction_service.correct_text(query)
        
        index_service: IndexService = app.config["INDEX_SERVICE"]
        results = index_service.search(query=corrected_query, top_k=top_k)
        return jsonify({
            "query": query,
            "corrected_query": corrected_query,
            "suggestions": suggestions,
            "top_k": top_k,
            "results": [{"path": r.path, "score": r.score, "url": url_for("image", path=r.path)} for r in results],
        })

    @app.get("/image")
    def image():
        image_path = request.args.get("path")
        if not image_path:
            abort(404)
        index_service: IndexService = app.config["INDEX_SERVICE"]
        abs_root = os.path.abspath(index_service.image_root)
        abs_path = os.path.abspath(image_path)
        if not abs_path.startswith(abs_root):
            abort(403)
        if not os.path.exists(abs_path):
            abort(404)
        return send_file(abs_path)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
