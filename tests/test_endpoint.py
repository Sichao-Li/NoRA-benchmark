from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from threading import Thread

import pytest

from nora.models import ChatCompletionsModel, ModelInput


@pytest.mark.parametrize("finish", ["stop", "length"])
@pytest.mark.parametrize("media,filename,part_type,mime", [
    ("frames", "frame_all_prev.jpg", "image_url", "image/jpeg"),
    ("video", "video_prev.mp4", "video_url", "video/mp4"),
])
def test_compatible_endpoint_roundtrip(tmp_path, monkeypatch, finish, media, filename, part_type, mime):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"finish_reason": finish,
                "message": {"content": "Chosen action: wait."}}]}).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.delenv("NORA_API_KEY", raising=False)
    image = tmp_path / filename
    image.write_bytes(b"test-image")
    model = ChatCompletionsModel(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="custom")
    try:
        item = ModelInput("x", image, "system", "task", "direct", media)
        if finish == "stop":
            assert model(item) == "Chosen action: wait."
        else:
            with pytest.raises(ValueError, match="incomplete"):
                model(item)
        path, body = received[0]
        assert path == "/v1/chat/completions"
        assert body["model"] == "custom"
        content = body["messages"][1]["content"][1]
        assert content["type"] == part_type
        assert content[part_type]["url"].startswith(f"data:{mime};base64,")
        assert "labels" not in json.dumps(body)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
