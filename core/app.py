# app.py - Voice Gmail Assistant MVP (server-side Whisper STT + Gmail API)
import os
import json
import uuid
import time
import base64
import requests
from pathlib import Path
from email.message import EmailMessage

from flask import (
    Flask, url_for, redirect, session, request, jsonify, render_template,
    send_from_directory
)
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user, UserMixin
)
from dotenv import load_dotenv

# STT / TTS
import speech_recognition as sr
import pyttsx3
import whisper  # openai-whisper
from bs4 import BeautifulSoup

def gmail_get_message_full(access_token: str, msg_id: str):
    """
    Return full message JSON for msg_id (format=full).
    """
    url = f"{GMAIL_API_BASE}/users/me/messages/{msg_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"format": "full"}  # full includes payload and parts
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _extract_text_from_payload(payload):
    """
    Recursively extract text content from a Gmail payload.
    Prefer text/plain; if only text/html, convert to text.
    Returns concatenated string.
    """
    parts = []
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {}) or {}

    # helper to decode base64url (may lack padding)
    def _b64u_decode(data_b64u: str) -> bytes:
        if not data_b64u:
            return b""
        s = data_b64u.replace("-", "+").replace("_", "/")
        # pad length to multiple of 4
        padding = 4 - (len(s) % 4) if (len(s) % 4) else 0
        s += "=" * padding
        return base64.b64decode(s)

    # If this part is plain text
    if mime_type == "text/plain":
        raw = body.get("data") or ""
        text = _b64u_decode(raw).decode(errors="ignore")
        if text:
            parts.append(text.strip())

    # If html, convert
    elif mime_type == "text/html":
        raw = body.get("data") or ""
        html = _b64u_decode(raw).decode(errors="ignore")
        if html:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n")
            parts.append(text.strip())

    # If multipart, iterate
    if payload.get("parts"):
        for p in payload.get("parts"):
            sub = _extract_text_from_payload(p)
            if sub:
                parts.append(sub)

    return "\n\n".join([p for p in parts if p]).strip()


def gmail_get_message_subject_and_body(access_token: str, msg_id: str):
    """
    Fetch message (full) and extract Subject and body text (plain).
    Returns (subject, body_text)
    """
    msg = gmail_get_message_full(access_token, msg_id)
    headers = msg.get("payload", {}).get("headers", [])
    subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(no subject)")
    # Extract body from payload
    body_text = _extract_text_from_payload(msg.get("payload", {}))
    # Fallback to snippet if body empty
    if not body_text:
        body_text = msg.get("snippet", "") or ""
    return subject, body_text

load_dotenv()

# --- Config ---
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("Warning: GOOGLE_CLIENT_ID / SECRET missing in environment.")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = APP_SECRET
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///demo.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# DB + auth
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

oauth = OAuth(app)

# --- OAuth registration (include Gmail scopes for Google) ---
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.send"
    },
)

oauth.register(
    name="microsoft",
    client_id=MS_CLIENT_ID,
    client_secret=MS_CLIENT_SECRET,
    server_metadata_url="https://login.microsoftonline.com/organizations/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile offline_access User.Read"},
)

# --- Whisper model (load once) ---
# Choose model size: "tiny", "base", "small", "medium", "large"
# smaller => faster but less accurate. "small" is a good compromise for demo.
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
print("Loading Whisper model:", WHISPER_MODEL_NAME)
WHISPER_MODEL = whisper.load_model(WHISPER_MODEL_NAME)


# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(50), nullable=False)
    provider_id = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False, unique=True)
    name = db.Column(db.String(200))
    language = db.Column(db.String(10), default="en")
    preferences = db.Column(db.Text, default="{}")  # JSON string (stores tokens etc.)

    def get_preferences(self):
        try:
            return json.loads(self.preferences or "{}")
        except Exception:
            return {}

    def set_preferences(self, obj: dict):
        self.preferences = json.dumps(obj or {})


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


with app.app_context():
    db.create_all()

def find_or_create_user(provider: str, provider_id: str, email: str, name: str):
    user = User.query.filter_by(email=email).first()
    if user:
        user.provider = provider
        user.provider_id = provider_id
        user.name = name or user.name
        db.session.commit()
        return user
    user = User(provider=provider, provider_id=provider_id, email=email, name=name)
    db.session.add(user)
    db.session.commit()
    return user

# TTS file directory
TTS_DIR = Path(app.static_folder) / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Helper: Token storage & refresh -----------------
def save_google_token_to_user(user: User, token: dict):
    prefs = user.get_preferences()
    prefs["google_token"] = token
    user.set_preferences(prefs)
    db.session.commit()

def get_google_token_for_user(user: User):
    prefs = user.get_preferences()
    return prefs.get("google_token")

def ensure_google_token_valid(user: User):
    """
    Ensure the google token is valid. If expired and refresh_token present, refresh using Authlib client.
    Returns token dict or None on failure.
    """
    token = get_google_token_for_user(user)
    if not token:
        return None
    # token may have 'expires_at' (unix). If expired or near expiry, refresh.
    expires_at = token.get("expires_at")  # may exist
    now = int(time.time())
    if expires_at and expires_at - now < 60:
        # refresh
        try:
            client = oauth.create_client("google")
            refreshed = client.refresh_token(
                token_endpoint=client.server_metadata["token_endpoint"],
                refresh_token=token.get("refresh_token"),
            )
            # merge/replace token fields
            token.update(refreshed)
            save_google_token_to_user(user, token)
            return token
        except Exception as e:
            print("Failed to refresh google token:", e)
            return token  # return original (may still work if not fully expired)
    return token

def fetch_last_n_messages_for_user(user: User, n=5):
    """
    Returns list of dicts: [{"id": id, "subject": subject, "body": body_text}, ...]
    """
    token = ensure_google_token_valid(user)
    if not token:
        raise RuntimeError("No Google token available for user")
    access_token = token.get("access_token")
    ids = gmail_list_message_ids(access_token, max_results=n)
    out = []
    for mid in ids:
        subj, body = gmail_get_message_subject_and_body(access_token, mid)
        out.append({"id": mid, "subject": subj, "body": body})
    return out

# ----------------- Gmail helpers (REST calls) -----------------
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"

def gmail_list_message_ids(access_token: str, max_results=5):
    url = f"{GMAIL_API_BASE}/users/me/messages"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"maxResults": max_results, "labelIds": "INBOX"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return [m["id"] for m in data.get("messages", [])]

def gmail_get_message_metadata(access_token: str, msg_id: str):
    url = f"{GMAIL_API_BASE}/users/me/messages/{msg_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"format": "metadata", "metadataHeaders": ["Subject", "From", "Date"]}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_last_n_subjects_for_user(user: User, n=5):
    token = ensure_google_token_valid(user)
    if not token:
        raise RuntimeError("No Google token available for user")
    access_token = token.get("access_token")
    ids = gmail_list_message_ids(access_token, max_results=n)
    subjects = []
    for mid in ids:
        meta = gmail_get_message_metadata(access_token, mid)
        headers = meta.get("payload", {}).get("headers", [])
        subj = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(no subject)")
        subjects.append(subj)
    return subjects

def gmail_send_message_for_user(user: User, to_email: str, subject: str, body_text: str):
    token = ensure_google_token_valid(user)
    if not token:
        raise RuntimeError("No Google token available for user")
    access_token = token.get("access_token")
    msg = EmailMessage()
    msg["To"] = to_email
    msg["From"] = user.email
    msg["Subject"] = subject
    msg.set_content(body_text)
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    url = f"{GMAIL_API_BASE}/users/me/messages/send"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json={"raw": raw}, timeout=10)
    r.raise_for_status()
    return r.json()

# ----------------- ROUTES -----------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("voice_page"))
    return render_template("index.html")

@app.route("/login/<provider>")
def login(provider):
    if provider not in ("google", "microsoft"):
        return "Unknown provider", 400
    redirect_uri = url_for("auth_callback", provider=provider, _external=True)
    client = oauth.create_client(provider)
    # For Google request offline access to get refresh token
    if provider == "google":
        return client.authorize_redirect(redirect_uri, access_type="offline", prompt="consent")
    return client.authorize_redirect(redirect_uri)

@app.route("/auth/<provider>/callback")
def auth_callback(provider):
    if provider not in ("google", "microsoft"):
        return "Unknown provider", 400
    client = oauth.create_client(provider)
    token = client.authorize_access_token()
    # token is a dict containing access_token, refresh_token, expires_at etc.
    # userinfo
    userinfo = token.get("userinfo")
    if not userinfo:
        try:
            userinfo = client.userinfo()
        except Exception:
            userinfo = {}
    provider_id = userinfo.get("sub") or userinfo.get("oid") or token.get("id_token")
    email = userinfo.get("email") or token.get("email")
    name = userinfo.get("name") or userinfo.get("preferred_username") or email
    if not email:
        return "Unable to retrieve email from provider", 400
    user = find_or_create_user(provider, provider_id, email, name)
    # Persist Google token to user's preferences (prototype)
    if provider == "google":
        try:
            save_google_token_to_user(user, token)
        except Exception as e:
            print("Failed to save google token:", e)
    login_user(user)
    return redirect(url_for("profile_page"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/profile", methods=["GET", "PUT"])
@login_required
def profile():
    if request.method == "GET":
        return jsonify({
            "email": current_user.email,
            "name": current_user.name,
            "language": current_user.language,
            "preferences": current_user.get_preferences()
        })
    data = request.get_json() or {}
    lang = data.get("language")
    prefs = data.get("preferences")
    if lang:
        current_user.language = lang[:10]
    if isinstance(prefs, dict):
        current_user.set_preferences(prefs)
    db.session.commit()
    return jsonify({"status": "ok"})

@app.route("/profile-page")
@login_required
def profile_page():
    return render_template("profile.html")

@app.route("/update-profile", methods=["POST"])
@login_required
def update_profile():
    data = request.get_json()
    language = data.get("language")
    if language:
        current_user.language = language
        db.session.commit()
    return jsonify({"message": "Profile updated successfully."})

@app.route("/voice")
@login_required
def voice_page():
    return render_template("voice.html")

# Server-side voice endpoint using Whisper
@app.route("/server-voice", methods=["POST"])
@login_required
def server_voice():
    """
    Listen on local microphone (re-init per request), save WAV, transcribe with Whisper,
    process command (including Gmail calls), generate TTS WAV, and return JSON with audio URL.
    """
    recognizer = sr.Recognizer()
    language = getattr(current_user, "language", "en") or "en"
    lang_code = "en-IN" if language.startswith("en") else language

    # 1) Capture from mic
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=12)
    except sr.WaitTimeoutError:
        return jsonify({"error": "timeout", "message": "No speech detected (timeout)."}), 400
    except Exception as e:
        return jsonify({"error": "mic_error", "message": f"Microphone error: {e}"}), 500

    # Save WAV to temp file for Whisper
    uid = uuid.uuid4().hex
    wav_path = Path("tmp")
    wav_path.mkdir(exist_ok=True)
    wav_file = wav_path / f"rec_{uid}.wav"
    with open(wav_file, "wb") as f:
        f.write(audio.get_wav_data())

    # 2) Transcribe with Whisper (model loaded at startup)
    try:
        # Use language parameter if desired; whisper auto-detects; forcing may improve
        res = WHISPER_MODEL.transcribe(str(wav_file), language="en")
        recognized_text = (res.get("text") or "").strip()
    except Exception as e:
        recognized_text = ""
        print("Whisper transcription failed:", e)
        # fallback to Google STT (optional)
        try:
            recognized_text = recognizer.recognize_google(audio, language=lang_code)
        except Exception:
            return jsonify({"error":"stt_failed","message":"STT failed"}), 500

    # remove temp wav file (optional)
    try:
        wav_file.unlink(missing_ok=True)
    except Exception:
        pass

    # 3) Command logic: handle Gmail commands specially
    cmd = recognized_text.lower().strip()
    response_text = ""
    audio_url = None

    if not cmd:
        response_text = "Please say something."
    elif any(kw in cmd for kw in ("check inbox", "latest email", "read my latest email", "read latest email")):
        # Fetch latest full email
        try:
            msgs = fetch_last_n_messages_for_user(current_user, n=1)
            if msgs:
                m = msgs[0]
                # truncate body for speech: make sure we don't read megabytes
                body = (m["body"] or "").strip()
                # Prefer first 800 characters or first 2 sentences
                if body:
                    # try to break into sentences
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', body)
                    preview = " ".join(sentences[:2]) if len(sentences) >= 2 else body[:800]
                    response_text = f"Latest email from {m.get('subject','(no subject)')}. {preview}"
                else:
                    response_text = f"Latest email subject: {m.get('subject','(no subject)')}. Email body is empty."
            else:
                response_text = "Could not find any recent emails."
        except Exception as e:
            response_text = f"Failed to fetch emails: {e}"
    elif any(kw in cmd for kw in ("read my emails", "read emails", "read latest five")):
        try:
            msgs = fetch_last_n_messages_for_user(current_user, n=5)
            if msgs:
                # join subject and first-line body preview
                lines = []
                for m in msgs:
                    b = (m.get("body") or "").strip()
                    # first sentence or 120 chars
                    if b:
                        first = b.splitlines()[0]
                        first = first if len(first) <= 160 else first[:157] + "..."
                    else:
                        first = "(no body)"
                    lines.append(f"{m.get('subject','(no subject)')} — {first}")
                response_text = "Here are the last five emails: " + " | ".join(lines)
            else:
                response_text = "No emails found."
        except Exception as e:
            response_text = f"Failed to fetch emails: {e}"
    elif "send email" in cmd or "compose email" in cmd:
        # minimal flow for demo: ask for recipient and subject/body via typed UI would be better.
        response_text = "Send email via voice is not implemented in this demo. Use the profile to send."
    elif "hello" in cmd:
        response_text = "Hello. How can I assist you today?"
    elif "exit" in cmd:
        response_text = "Goodbye."
    else:
        response_text = "This feature will be available in the next milestone."

    # 4) TTS: generate WAV file (pyttsx3, fresh engine)
    tts_uid = uuid.uuid4().hex
    tts_filename = f"tts_{tts_uid}.wav"
    tts_filepath = TTS_DIR / tts_filename

    try:
        try:
            engine = pyttsx3.init(driverName="sapi5")
        except Exception:
            engine = pyttsx3.init()
        # Optional: tune voice/rate/volume from preferences
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        engine.save_to_file(response_text, str(tts_filepath))
        engine.runAndWait()
        time.sleep(0.12)
        audio_url = url_for("tts_file", fname=tts_filename, _external=True)
    except Exception as e:
        print("TTS failed:", e)
        audio_url = None

    return jsonify({
        "recognized": recognized_text,
        "response": response_text,
        "audio_url": audio_url
    })

# Serve TTS static file
@app.route("/static/tts/<path:fname>")
def tts_file(fname):
    return send_from_directory(str(TTS_DIR), fname, as_attachment=False)

@app.route("/process-command", methods=["POST"])
@login_required
def process_command():
    data = request.get_json() or {}
    command = (data.get("command") or "").lower().strip()
    if not command:
        return jsonify({"response": "Please say something."})
    # simple mapping - uses gmail helper where needed
    if "check inbox" in command or "latest email" in command:
        try:
            subjects = fetch_last_n_subjects_for_user(current_user, n=1)
            if subjects:
                return jsonify({"response": f"Latest email subject: {subjects[0]}"})
            return jsonify({"response": "No recent emails found."})
        except Exception as e:
            return jsonify({"response": f"Failed to fetch emails: {e}"})
    if "read email" in command:
        try:
            subjects = fetch_last_n_subjects_for_user(current_user, n=5)
            if subjects:
                text = "Last five email subjects: " + " | ".join(subjects)
                return jsonify({"response": text})
            return jsonify({"response": "No emails found."})
        except Exception as e:
            return jsonify({"response": f"Failed to fetch emails: {e}"})
    if "send email" in command:
        return jsonify({"response": "To send email, use the send UI or implement voice-compose flow."})
    if "hello" in command:
        return jsonify({"response": "Hello. How can I help?"})
    return jsonify({"response": "This feature will be available in the next milestone."})

@app.route("/gmail/fetch", methods=["GET"])
@login_required
def gmail_fetch_endpoint():
    try:
        msgs = fetch_last_n_messages_for_user(current_user, n=5)
        # Return subject + first 500 chars of body to keep payload small
        for m in msgs:
            m["body_preview"] = (m.get("body") or "")[:500]
            m.pop("body", None)
        return jsonify({"messages": msgs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/gmail/send", methods=["POST"])
@login_required
def gmail_send_endpoint():
    data = request.get_json() or {}
    to = data.get("to")
    subject = data.get("subject", "")
    body = data.get("body", "")
    if not to:
        return jsonify({"error": "recipient required"}), 400
    try:
        res = gmail_send_message_for_user(current_user, to, subject, body)
        return jsonify({"ok": True, "result": res})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/voice-demo")
@login_required
def voice_demo():
    return jsonify({
        "message": "You are authorized to run the voice demo.",
        "user": {"email": current_user.email, "language": current_user.language}
    })

@app.route("/whoami")
def whoami():
    return jsonify({"auth": current_user.is_authenticated, "email": getattr(current_user, "email", None)})

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)