#!/usr/bin/env bash
set -euo pipefail

echo "=== FoodChat Mac bootstrap ==="

# Move into the directory where this script lives
cd "$(dirname "$0")"

# --- 0) Xcode Command Line Tools (for building Python if needed) ---
if ! xcode-select -p >/dev/null 2>&1; then
  echo "• Xcode Command Line Tools are required. A dialog will open; please install and then re-run this script."
  xcode-select --install || true
  exit 1
fi

# --- 1) Homebrew ---
if ! command -v brew >/dev/null 2>&1; then
  echo "• Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Add Homebrew to PATH (Apple Silicon vs Intel)
  if [ -d "/opt/homebrew/bin" ]; then
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    echo 'eval "$(/usr/local/bin/brew shellenv)"' >> "$HOME/.zprofile"
    eval "$(/usr/local/bin/brew shellenv)"
  fi
else
  # Make sure brew is in PATH for this shell
  if [ -d "/opt/homebrew/bin" ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -d "/usr/local/bin" ]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi

echo "• Homebrew ready: $(brew --version | head -n1)"

# --- 2) pyenv + Python 3.10.11 ---
PY_VER="3.10.11"
if ! command -v pyenv >/dev/null 2>&1; then
  echo "• Installing pyenv..."
  brew install pyenv
fi
echo "• Ensuring Python $PY_VER is available..."
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install -s "$PY_VER"
pyenv local "$PY_VER"

# --- 3) Virtual env ---
if [ ! -d ".venv" ]; then
  echo "• Creating virtual environment (.venv)..."
  python -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -V
pip install --upgrade pip setuptools wheel

# --- 4) Python dependencies ---
if [ -f "requirements.txt" ]; then
  echo "• Installing Python dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "! requirements.txt not found; skipping."
fi

# On macOS, torch wheel selection is automatic. If missing, uncomment next line:
# pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# --- 5) Ensure data folder exists & pre-download Whisper model ---
mkdir -p data/whisper
export FOODCHAT_ASSETS_DIR="$(pwd)/data"
python - <<'PY'
import os, whisper
root=os.environ.get("FOODCHAT_ASSETS_DIR","data")
os.makedirs(os.path.join(root,"whisper"), exist_ok=True)
print("• Downloading/validating Whisper 'base' model into", os.path.join(root,"whisper"))
_ = whisper.load_model("base", download_root=os.path.join(root,"whisper"))
print("✓ Whisper model ready.")
PY

# --- 6) Create a double-click launcher (Run FoodChat.command) ---
cat > "Run FoodChat.command" <<'CMD'
#!/usr/bin/env bash
cd "$(dirname "$0")"
# Activate venv
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi
# Point the app to local assets/models
export FOODCHAT_ASSETS_DIR="$(pwd)/data"
# Launch the app
python main.py
read -n 1 -s -r -p "Press any key to close this window..."
CMD
chmod +x "Run FoodChat.command"

echo "=== Setup complete ==="
echo "Next time, just double-click: Run FoodChat.command"
