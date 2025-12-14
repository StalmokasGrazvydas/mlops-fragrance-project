import json
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid
import joblib

model = None


def init():
    """
    Azure ML calls init() once when the container starts.
    It must load the model from the mounted AZUREML_MODEL_DIR.
    """
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR")

    if not model_dir:
        raise RuntimeError(
            "AZUREML_MODEL_DIR is not set. Azure ML should set this during deployment."
        )

    # Common patterns: model.pkl at root, or inside a folder
    candidates = [
        Path(model_dir) / "model.pkl",
        Path(model_dir) / "model_output" / "model.pkl",
        Path(model_dir) / "model" / "model.pkl",
    ]

    model_path = None
    for c in candidates:
        if c.exists():
            model_path = c
            break

    if model_path is None:
        # Fallback: find first .pkl anywhere
        pkls = list(Path(model_dir).rglob("*.pkl"))
        if pkls:
            model_path = pkls[0]

    if model_path is None:
        raise FileNotFoundError(f"No .pkl model found under AZUREML_MODEL_DIR={model_dir}")

    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")


def _best_effort_log_request_local(rec: dict) -> None:
    """
    Best-effort request logging to JSONL locally. Never breaks inference if logging fails.

    If INFERENCE_LOG_DIR is set, we append a JSON line to:
      <INFERENCE_LOG_DIR>/requests-YYYY-MM-DD.jsonl
    """
    log_dir = os.environ.get("INFERENCE_LOG_DIR")
    if not log_dir:
        return

    try:
        os.makedirs(log_dir, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fname = os.path.join(log_dir, f"requests-{day}.jsonl")

        with open(fname, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Local logging failed: {e}")


def _best_effort_log_request_blob(rec: dict) -> None:
    """
    Optional: log JSONL lines to Azure Blob Storage.
    Requires:
      AZURE_STORAGE_CONNECTION_STRING
      INFERENCE_LOG_CONTAINER (e.g. inference-logs)
    """
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container = os.environ.get("INFERENCE_LOG_CONTAINER")
    if not conn_str or not container:
        return

    try:
        from azure.storage.blob import BlobServiceClient

        blob_service = BlobServiceClient.from_connection_string(conn_str)
        blob_name = f"requests-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        client = blob_service.get_blob_client(container=container, blob=blob_name)

        line = json.dumps(rec, ensure_ascii=False) + "\n"

        # Prefer Append Blob semantics
        try:
            if not client.exists():
                client.create_append_blob()
            client.append_block(line.encode("utf-8"))
        except Exception:
            # Fallback: overwrite (not ideal, but acceptable for a low-volume demo)
            client.upload_blob(line.encode("utf-8"), overwrite=True)

    except Exception as e:
        print(f"Blob logging failed: {e}")


def run(raw_data):
    """
    Azure ML calls run() per request.
    Accepts dict or JSON string and returns {"predictions": [<label>]}.
    """
    global model
    if model is None:
        raise RuntimeError("Model is not loaded. init() was not called.")

    if raw_data is None:
        return {"error": "Empty request body."}

    # Accept either dict or JSON string
    if isinstance(raw_data, (bytes, bytearray)):
        raw_data = raw_data.decode("utf-8")

    if isinstance(raw_data, str):
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {str(e)}"}
    elif isinstance(raw_data, dict):
        data = raw_data
    else:
        try:
            data = dict(raw_data)
        except Exception:
            return {"error": "Unsupported input type. Provide JSON with a 'text' field."}

    # text = data.get("text", "")
    text = data.get("text") or data.get("reviewText") or data.get("review_text") or ""

    if not isinstance(text, str) or not text.strip():
        return {"error": "Input must be JSON with non-empty 'text' field."}

    rec = {
        "id": str(uuid.uuid4()),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "text": text,
    }

    # Log request (best effort; never breaks inference)
    _best_effort_log_request_local(rec)  # optional local file logging
    _best_effort_log_request_blob(rec)   # optional Azure Blob logging

    pred = model.predict([text])[0]
    return {"predictions": [pred]}
