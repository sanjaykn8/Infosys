# app.py
import os
import json
import uuid
import time
from pathlib import Path
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

# Local STT/TTS libs
import speech_recognition as sr
import pyttsx3

load_dotenv()

# --- Config ---
APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
MS_CLIENT_ID = os.getenv("MS_CLIENT_ID")
MS_CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = APP_SECRET
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///demo.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# OAuth registration (unchanged)
oauth = OAuth(app)
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"}
)
oauth.register(
    name="microsoft",
    client_id=MS_CLIENT_ID,
    client_secret=MS_CLIENT_SECRET,
    server_metadata_url="https://login.microsoftonline.com/organizations/v2.0/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile offline_access User.Read"}
)

# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(50), nullable=False)
    provider_id = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False, unique=True)
    name = db.Column(db.String(200))
    language = db.Column(db.String(10), default="en")
    preferences = db.Column(db.Text, default="{}")

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

# ensure tts folder exists
TTS_DIR = Path(app.static_folder) / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- routes (unchanged oauth/profile basics) -----------------
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
    return oauth.create_client(provider).authorize_redirect(redirect_uri)

@app.route("/auth/<provider>/callback")
def auth_callback(provider):
    if provider not in ("google", "microsoft"):
        return "Unknown provider", 400
    client = oauth.create_client(provider)
    token = client.authorize_access_token()
    userinfo = token.get("userinfo")
    if not userinfo:
        userinfo = client.userinfo()
    provider_id = userinfo.get("sub") or userinfo.get("oid")
    email = userinfo.get("email")
    name = userinfo.get("name") or userinfo.get("preferred_username")
    if not email:
        return "Unable to retrieve email from provider", 400
    user = find_or_create_user(provider, provider_id, email, name)
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

# ----------------- server-side STT + TTS endpoint -----------------
@app.route("/server-voice", methods=["POST"])
@login_required
def server_voice():
    """
    Listens on the local microphone, returns recognized text, computed response,
    and a URL to a generated TTS file (WAV).
    This re-initializes audio engines per request (works more reliably on Windows).
    """
    # STT: local microphone (re-init per request)
    recognizer = sr.Recognizer()
    language = getattr(current_user, "language", "en") or "en"
    lang_code = "en-IN" if language.startswith("en") else language

    try:
        with sr.Microphone() as source:
            # short ambient calibration
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            # listen (adjust timeouts as needed)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
    except sr.WaitTimeoutError:
        return jsonify({"error": "timeout", "message": "No speech detected (timeout)."}), 400
    except Exception as e:
        return jsonify({"error": "mic_error", "message": f"Microphone error: {e}"}), 500

    try:
        # Using Google Web Speech API via SpeechRecognition (requires internet)
        recognized_text = recognizer.recognize_google(audio, language=lang_code)
    except sr.UnknownValueError:
        return jsonify({"error": "unrecognized", "message": "Speech not recognized."}), 400
    except sr.RequestError as e:
        return jsonify({"error": "api_error", "message": f"STT request error: {e}"}), 502

    # Process command (same logic as /process-command)
    cmd = recognized_text.lower().strip()
    if not cmd:
        response_text = "Please say something."
    elif "check inbox" in cmd:
        response_text = "Inbox module will be added in the next milestone."
    elif "send email" in cmd:
        response_text = "Send email feature is under development."
    elif "read email" in cmd:
        response_text = "Reading email functionality is coming soon."
    elif "hello" in cmd:
        response_text = "Hello. How can I assist you today?"
    elif "exit" in cmd:
        response_text = "Goodbye."
    else:
        response_text = "This feature will be available in the next milestone."

    # TTS: create audio file using fresh pyttsx3 engine per request
    uid = uuid.uuid4().hex
    filename = f"tts_{uid}.wav"
    filepath = TTS_DIR / filename

    try:
        engine = pyttsx3.init(driverName="sapi5")  # windows SAPI5
    except Exception:
        engine = pyttsx3.init()

    # optional: set voice/rate/volume from user preferences
    rate = 150
    volume = 1.0
    try:
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
    except Exception:
        pass

    # save to file and run
    try:
        engine.save_to_file(response_text, str(filepath))
        engine.runAndWait()
        # some windows installs need a tiny sleep to flush file
        time.sleep(0.15)
    except Exception as e:
        # fallback: do not crash — return response text without audio
        return jsonify({
            "recognized": recognized_text,
            "response": response_text,
            "audio_url": None,
            "note": f"TTS failed: {e}"
        })

    audio_url = url_for("tts_file", fname=filename, _external=True)
    return jsonify({"recognized": recognized_text, "response": response_text, "audio_url": audio_url})

# serve tts files
@app.route("/static/tts/<path:fname>")
def tts_file(fname):
    return send_from_directory(str(TTS_DIR), fname, as_attachment=False)

# existing /process-command (text-only)
@app.route("/process-command", methods=["POST"])
@login_required
def process_command():
    data = request.get_json() or {}
    command = (data.get("command") or "").lower().strip()
    if not command:
        return jsonify({"response": "Please say something."})
    if "check inbox" in command:
        response = "Inbox module will be added in the next milestone."
    elif "send email" in command:
        response = "Send email feature is under development."
    elif "read email" in command:
        response = "Reading email functionality is coming soon."
    elif "hello" in command:
        response = "Hello. How can I assist you today?"
    elif "exit" in command:
        response = "Goodbye."
    else:
        response = "This feature will be available in the next milestone."
    return jsonify({"response": response})

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
    # ensure host matches redirect URIs you registered
    app.run(host="localhost", port=5000, debug=True)
