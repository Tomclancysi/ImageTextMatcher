import os

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for, abort, session

from .services.index_service import IndexService
from .services.text_correction_service import TextCorrectionService


def create_app() -> Flask:
    image_root = os.environ.get("ITM_IMAGE_ROOT", os.path.join(os.getcwd(), "data", "images"))
    index_dir = os.environ.get("ITM_INDEX_DIR", os.path.join(os.getcwd(), "data", "index"))
    dataset_csv = os.environ.get("ITM_DATASET_CSV", os.path.join(os.getcwd(), "data", "dataset_en.csv"))
    model_name = os.environ.get("ITM_MODEL_NAME", "openai/clip-vit-base-patch32")
    default_method = os.environ.get("ITM_METHOD", "clip")

    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
    
    # 添加自定义过滤器：提取文件名
    @app.template_filter('basename')
    def basename_filter(path):
        return os.path.basename(path)
    
    # 创建多个IndexService实例，支持不同方法
    app.config["INDEX_SERVICES"] = {
        "clip": IndexService(image_root=image_root, index_dir=index_dir, method="clip", model_name=model_name, dataset_csv=dataset_csv),
        "vse": IndexService(image_root=image_root, index_dir=index_dir, method="vse", model_name=model_name, dataset_csv=dataset_csv),
        "scan": IndexService(image_root=image_root, index_dir=index_dir, method="scan", model_name=model_name, dataset_csv=dataset_csv),
    }
    app.config["DEFAULT_METHOD"] = default_method
    app.config["TEXT_CORRECTION_SERVICE"] = TextCorrectionService(language="en")
    
    def get_index_service(method: str = None):
        """获取指定方法的IndexService"""
        if method is None:
            method = session.get("method", default_method)
        method = method.lower()
        if method not in app.config["INDEX_SERVICES"]:
            method = default_method
        return app.config["INDEX_SERVICES"][method]

    @app.get("/")
    def home():
        current_top_k = int(session.get("top_k", 10))
        current_method = session.get("method", default_method)
        return render_template("index.html", current_top_k=current_top_k, current_method=current_method)

    @app.post("/search")
    def search():
        query = request.form.get("query", "").strip()
        try:
            top_k = int(request.form.get("top_k", session.get("top_k", 10)))
        except ValueError:
            top_k = int(session.get("top_k", 10))
        
        method = request.form.get("method", session.get("method", default_method))
        method = method.lower()
        if method not in app.config["INDEX_SERVICES"]:
            method = default_method

        session["top_k"] = max(1, min(100, top_k))
        session["method"] = method

        if not query:
            return redirect(url_for("home"))

        # Text correction
        correction_service: TextCorrectionService = app.config["TEXT_CORRECTION_SERVICE"]
        corrected_query, suggestions = correction_service.correct_text(query)
        
        index_service = get_index_service(method)
        results = index_service.search(query=corrected_query, top_k=session["top_k"])
        
        return render_template("results.html", 
                             query=query, 
                             corrected_query=corrected_query,
                             suggestions=suggestions,
                             results=results, 
                             current_top_k=session["top_k"],
                             method=method)

    @app.get("/api/search")
    def api_search():
        query = request.args.get("q", "").strip()
        k_arg = request.args.get("k")
        method_arg = request.args.get("method", session.get("method", default_method))
        method_arg = method_arg.lower()
        if method_arg not in app.config["INDEX_SERVICES"]:
            method_arg = default_method
        
        if k_arg is not None:
            try:
                session["top_k"] = max(1, min(100, int(k_arg)))
            except ValueError:
                pass
        top_k = int(session.get("top_k", 10))
        session["method"] = method_arg
        
        if not query:
            return jsonify({"error": "missing q"}), 400
        
        # Text correction for API
        correction_service: TextCorrectionService = app.config["TEXT_CORRECTION_SERVICE"]
        corrected_query, suggestions = correction_service.correct_text(query)
        
        index_service = get_index_service(method_arg)
        results = index_service.search(query=corrected_query, top_k=top_k)
        return jsonify({
            "query": query,
            "corrected_query": corrected_query,
            "suggestions": suggestions,
            "top_k": top_k,
            "method": method_arg,
            "results": [{"path": r.path, "score": r.score, "description": r.description, "url": url_for("image", path=r.path)} for r in results],
        })

    @app.get("/image")
    def image():
        image_path = request.args.get("path")
        if not image_path:
            abort(404)
        # 使用默认服务获取image_root
        index_service = get_index_service()
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
