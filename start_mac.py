# start_mac.py
import os, sys, time, webbrowser, pathlib

# Env guards
os.environ.setdefault("FLASK_SKIP_DOTENV", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("PORT", "5002")

# Add bundled ffmpeg (optional; see packaging step)
bundle = getattr(sys, "_MEIPASS", str(pathlib.Path(__file__).resolve().parent))
ffmpeg_dir = str(pathlib.Path(bundle) / "assets" / "bin" / "macos")
if os.path.exists(ffmpeg_dir):
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

from main import app

def run():
    port = int(os.getenv("PORT", "5002"))
    import threading
    t = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, debug=False), daemon=True)
    t.start()
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{port}")
    t.join()

if __name__ == "__main__":
    run()
