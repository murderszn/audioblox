"""
AI Voice Assistant Server
------------------------
This Flask server handles the backend processing for the AI Voice Assistant.
It integrates with Google's Gemini API for chat functionality and Google Cloud
Text-to-Speech for voice synthesis.

Key Features:
- Chat session management
- Voice synthesis
- API integration
- CORS handling for local development
- Async task processing
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import time
import io
import threading
from queue import Queue
import re
import hashlib
from functools import lru_cache
import wave
import json
import asyncio
from flask_sock import Sock

# Initialize Flask app with CORS support
app = Flask(__name__, static_folder='.', static_url_path='')
sock = Sock(app)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:7777", "http://127.0.0.1:7777"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Session-ID"]
    }
})

# Load environment variables from .env file
load_dotenv()

# Eagerly initialize GenAI client at import time for lower latency
genai_client = None

def _init_genai_client():
    global genai_client
    if genai_client is None:
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENERATIVE_LANGUAGE_API_KEY')
        if api_key:
            genai_client = genai.Client(api_key=api_key)
    return genai_client

# Pre-initialize at module load (after load_dotenv)
_init_genai_client()

def get_genai_client():
    if genai_client is None:
        _init_genai_client()
    if genai_client is None:
        raise Exception("GOOGLE_API_KEY is not configured in the environment variables.")
    return genai_client

# Task processing queue and patterns
task_queue = Queue()
PRINCIPAL_COMMANDS = {
    r"(?i)as principal,?\s*(approve|deny|review)\s+(.+)": "Administrative",
    r"(?i)as principal,?\s*(schedule|arrange)\s+(.+)": "Scheduling",
    r"(?i)as principal,?\s*(contact|email|call)\s+(.+)": "Communication",
    r"(?i)as principal,?\s*(implement|establish)\s+(.+)": "Policy",
    r"(?i)as principal,?\s*(evaluate|assess)\s+(.+)": "Evaluation",
    r"(?i)as principal,?\s*(authorize|permit)\s+(.+)": "Authorization"
}

def process_tasks():
    """Background task processor"""
    while True:
        try:
            session_id, task = task_queue.get()
            if session_id in chat_sessions:
                session = chat_sessions[session_id]
                if 'tasks' not in session:
                    session['tasks'] = []
                session['tasks'].append(task)
                print(f"Processed task for session {session_id}: {task}")
            task_queue.task_done()
        except Exception as e:
            print(f"Error processing task: {e}")

# Start task processing thread
task_thread = threading.Thread(target=process_tasks, daemon=True)
task_thread.start()

def check_principal_command(text):
    """
    Check if text contains a principal command and categorize it
    
    Args:
        text (str): The message text to check
        
    Returns:
        tuple: (command_type, command_content) or (None, None) if no command found
    """
    for pattern, category in PRINCIPAL_COMMANDS.items():
        match = re.match(pattern, text)
        if match:
            action = match.group(1)
            content = match.group(2)
            return category, f"{action.capitalize()}: {content}"
    return None, None

def clean_text_for_speech(text):
    """
    Fast text cleaning for speech synthesis.
    Removes markdown formatting that shouldn't be spoken.
    """
    # Fast path: strip markdown in one pass
    text = re.sub(r'\*\*(.+?)\*\*|\*(.+?)\*|\_(.+?)\_', lambda m: m.group(1) or m.group(2) or m.group(3), text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# TTS cache: maps (text_hash, voice) -> raw PCM bytes
_tts_cache = {}
_tts_cache_lock = threading.Lock()
_TTS_CACHE_MAX = 200

def _tts_cache_key(text, voice_name):
    return (hashlib.md5(text.encode()).hexdigest(), voice_name)

def generate_speech(text, voice_name="Puck"):
    """
    Generate speech from text using Gemini's native TTS model.
    Returns raw PCM audio bytes (16-bit, 24kHz mono).
    Uses in-memory cache for repeated sentences.
    """
    cleaned_text = clean_text_for_speech(text)
    cache_key = _tts_cache_key(cleaned_text, voice_name)

    # Check cache first
    with _tts_cache_lock:
        if cache_key in _tts_cache:
            return _tts_cache[cache_key]

    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=cleaned_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
        )

        audio_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data:
                audio_data = part.inline_data.data
                break

        if not audio_data:
            raise Exception("No audio data returned from Gemini TTS")

        # Cache the result (evict oldest if at capacity)
        with _tts_cache_lock:
            if len(_tts_cache) >= _TTS_CACHE_MAX:
                # Remove first inserted key (FIFO eviction)
                _tts_cache.pop(next(iter(_tts_cache)))
            _tts_cache[cache_key] = audio_data

        return audio_data
    except Exception as e:
        raise Exception(f"Failed to generate speech: {str(e)}")


def generate_speech_wav(text, voice_name="Puck"):
    """Generate speech and return as WAV format bytes."""
    pcm_data = generate_speech(text, voice_name)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_data)
    return wav_buf.getvalue()

def init_gemini(api_key):
    """
    Initialize Gemini chat model with specific configuration.
    
    Args:
        api_key (str): The Gemini API key
        
    Returns:
        google.genai.chats.Chat: Initialized chat session
        
    Raises:
        Exception: If initialization fails
    """
    try:
        client = get_genai_client()
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=150,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                ]
            ),
            history=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="I want you to act as an advanced real-time conversational voice and text assistant. Be warm, natural, engaging, and highly conversational. You are frequently used as an acting practice partner, customer service agent, or general conversational assistant. When we are practicing acting or roleplaying, fully adapt to your character, deliver complete lines, express appropriate emotions, and feel free to generate longer, descriptive script segments, lines, or monologues as needed. Otherwise, keep your responses naturally conversational and moderate (1 to 3 sentences max) so that our live conversation flows quickly and naturally. Never use markdown formatting, bullet points, or list structures. Respond naturally like a human would in a live voice conversation."
                    )]
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="Understood! I will be a warm, natural, and engaging conversational partner. By default, I'll keep my answers moderate and conversational (1 to 3 sentences) so the dialogue flows smoothly, but I'll adapt fully to deliver complete script lines, monologues, or customer service roleplay whenever we are practicing scenes. What scenario or character would you like to start with?"
                    )]
                )
            ]
        )
        
        return chat
    except Exception as e:
        raise Exception(f"Failed to initialize Gemini: {str(e)}")

# Global store for active chat sessions
chat_sessions = {}

# Route handlers
@app.route('/')
def root():
    """Serve the main application page"""
    return app.send_static_file('index.html')

@app.route('/test', methods=['GET', 'OPTIONS'])
def test_connection():
    """Test endpoint for checking server connectivity"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/init', methods=['POST', 'OPTIONS'])
def init_session():
    """Initialize a new chat session"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        # Use API key from environment variables
        api_key = os.getenv('GOOGLE_GENERATIVE_LANGUAGE_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            return jsonify({'error': 'API key not found in environment variables'}), 400

        # Initialize Gemini chat
        try:
            chat = init_gemini(api_key)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        # Create new session
        session_id = str(time.time())
        chat_sessions[session_id] = {
            'chat': chat
        }
        
        return jsonify({
            'session_id': session_id,
            'status': 'initialized'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/synthesize', methods=['POST', 'OPTIONS'])
def synthesize_speech():
    """Generate speech from text. Returns raw PCM (16-bit, 24kHz, mono)."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        session_id = request.headers.get('Session-ID')
        voice_name = 'Puck'
        if session_id and session_id in chat_sessions:
            voice_name = chat_sessions[session_id].get('voice', 'Puck')

        pcm_data = generate_speech(text, voice_name=voice_name)

        return Response(
            pcm_data,
            mimetype='audio/pcm',
            headers={
                'Content-Type': 'audio/pcm',
                'X-Sample-Rate': '24000',
                'X-Channels': '1',
                'X-Bits-Per-Sample': '16'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/synthesize_wav', methods=['POST', 'OPTIONS'])
def synthesize_speech_wav():
    """Generate speech from text. Returns WAV format."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Text is required'}), 400

        session_id = request.headers.get('Session-ID')
        voice_name = 'Puck'
        if session_id and session_id in chat_sessions:
            voice_name = chat_sessions[session_id].get('voice', 'Puck')

        wav_data = generate_speech_wav(text, voice_name=voice_name)

        return Response(
            wav_data,
            mimetype='audio/wav',
            headers={'Content-Disposition': 'attachment; filename=response.wav'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/synthesize_batch', methods=['POST', 'OPTIONS'])
def synthesize_batch():
    """Synthesize multiple sentences in one call. Returns concatenated raw PCM."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.json
        sentences = data.get('sentences', [])
        if not sentences:
            return jsonify({'error': 'Sentences list is required'}), 400

        session_id = request.headers.get('Session-ID')
        voice_name = 'Puck'
        if session_id and session_id in chat_sessions:
            voice_name = chat_sessions[session_id].get('voice', 'Puck')

        # Synthesize all sentences and concatenate PCM
        all_pcm = b''
        for sentence in sentences:
            if sentence and sentence.strip():
                pcm = generate_speech(sentence.strip(), voice_name=voice_name)
                all_pcm += pcm

        return Response(
            all_pcm,
            mimetype='audio/pcm',
            headers={
                'X-Sample-Rate': '24000',
                'X-Channels': '1',
                'X-Bits-Per-Sample': '16'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/settings', methods=['POST', 'OPTIONS'])
def update_settings():
    """Update settings for the active session"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        session_id = request.headers.get('Session-ID')
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        data = request.json
        voice = data.get('voice')
        if voice:
            chat_sessions[session_id]['voice'] = voice
            
        return jsonify({
            'status': 'success',
            'voice': chat_sessions[session_id].get('voice', 'Puck')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle chat messages and generate responses as a text stream"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        # Validate session
        session_id = request.headers.get('Session-ID')
        if not session_id or session_id not in chat_sessions:
            return jsonify({'error': 'Invalid session'}), 400

        # Get message
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Check for principal commands
        command_type, command_content = check_principal_command(message)
        
        if command_type:
            # Add task to processing queue
            task = {
                'type': command_type,
                'content': command_content,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'pending'
            }
            task_queue.put((session_id, task))

        # Get chat session
        session = chat_sessions[session_id]
        chat_obj = session['chat']

        def generate():
            full_response = []
            try:
                for chunk in chat_obj.send_message_stream(message):
                    text_part = chunk.text
                    if text_part:
                        full_response.append(text_part)
                        yield text_part
            except Exception as e:
                yield f"\n[STREAM_ERROR: {str(e)}]"
            
            # Save history
            if 'history' not in session:
                session['history'] = []
            session['history'].append({
                'user': message,
                'ai': "".join(full_response)
            })

        return app.response_class(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── Gemini Live API WebSocket Proxy ────────────────────────────────────────

@sock.route('/ws/live')
def live_api_proxy(ws):
    """
    WebSocket proxy for Gemini Live API.
    Client sends: {"type": "audio", "data": "<base64 PCM>"}
                    {"type": "text",  "text": "message"}
                    {"type": "end_of_turn"}
                    {"type": "config", "voice": "Puck"}
    Server sends: {"type": "audio", "data": "<base64 PCM>"}
                   {"type": "text",  "text": "..."}
                   {"type": "turn_complete"}
                   {"type": "interrupted"}
    """
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENERATIVE_LANGUAGE_API_KEY')
    if not api_key:
        ws.send(json.dumps({"type": "error", "text": "API key not configured"}))
        return

    # Get voice from session or default
    session_id = request.args.get('session_id', '')
    voice_name = 'Puck'
    if session_id and session_id in chat_sessions:
        voice_name = chat_sessions[session_id].get('voice', 'Puck')

    client_to_gemini = Queue(maxsize=64)
    gemini_to_client = Queue(maxsize=64)

    def run_gemini_live():
        """Run the async Gemini Live session in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def gemini_session():
            client = get_genai_client()
            model = "gemini-2.5-flash-native-audio-latest"
            config = {
                "response_modalities": ["AUDIO"],
                "speech_config": types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
                "system_instruction": types.Content(
                    parts=[types.Part.from_text(
                        text="You are a warm, natural, and engaging voice assistant. Keep responses conversational and concise (1-3 sentences). Never use markdown or formatting. Respond naturally like a human in a live voice conversation."
                    )]
                ),
            }

            try:
                async with client.aio.live.connect(model=model, config=config) as session:
                    gemini_to_client.put(json.dumps({"type": "connected"}))

                    async def receive_from_gemini():
                        try:
                            async for response in session.receive():
                                if response.data is not None:
                                    import base64
                                    audio_b64 = base64.b64encode(response.data).decode()
                                    gemini_to_client.put(json.dumps({
                                        "type": "audio",
                                        "data": audio_b64
                                    }))
                                if response.text is not None:
                                    gemini_to_client.put(json.dumps({
                                        "type": "text",
                                        "text": response.text
                                    }))
                                if response.turn_complete:
                                    gemini_to_client.put(json.dumps({
                                        "type": "turn_complete"
                                    }))
                                if response.interrupted:
                                    gemini_to_client.put(json.dumps({
                                        "type": "interrupted"
                                    }))
                        except Exception as e:
                            gemini_to_client.put(json.dumps({
                                "type": "error",
                                "text": f"Receive error: {str(e)}"
                            }))

                    async def send_to_gemini():
                        try:
                            while True:
                                msg = client_to_gemini.get()
                                if msg is None:
                                    break
                                import base64
                                if msg["type"] == "audio":
                                    audio_bytes = base64.b64decode(msg["data"])
                                    await session.send_realtime_input(
                                        audio=types.Blob(
                                            data=audio_bytes,
                                            mime_type="audio/pcm;rate=16000"
                                        )
                                    )
                                elif msg["type"] == "text":
                                    await session.send_client_content(
                                        turns=[types.Content(
                                            role="user",
                                            parts=[types.Part.from_text(text=msg["text"])]
                                        )],
                                        turn_complete=True
                                    )
                                elif msg["type"] == "end_of_turn":
                                    # Send empty turn marker
                                    pass
                        except Exception as e:
                            gemini_to_client.put(json.dumps({
                                "type": "error",
                                "text": f"Send error: {str(e)}"
                            }))

                    # Use asyncio.wait with return_when=FIRST_COMPLETED to ensure when either sender or receiver
                    # ends (e.g. on disconnect), the other is cancelled and we clean up properly.
                    done, pending = await asyncio.wait(
                        [
                            asyncio.create_task(send_to_gemini()),
                            asyncio.create_task(receive_from_gemini())
                        ],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
            except Exception as e:
                gemini_to_client.put(json.dumps({
                    "type": "error",
                    "text": f"Connection error: {str(e)}"
                }))

        try:
            loop.run_until_complete(gemini_session())
        except Exception as e:
            gemini_to_client.put(json.dumps({
                "type": "error",
                "text": f"Fatal: {str(e)}"
            }))
        finally:
            loop.close()

    # Start Gemini session in background thread
    gemini_thread = threading.Thread(target=run_gemini_live, daemon=True)
    gemini_thread.start()

    # Wait for connection confirmation
    try:
        conn_msg = gemini_to_client.get(timeout=10)
        conn_data = json.loads(conn_msg)
        if conn_data.get("type") == "error":
            ws.send(conn_msg)
            return
    except Exception:
        ws.send(json.dumps({"type": "error", "text": "Connection timeout"}))
        return

    # Bidirectional proxy: client <-> Gemini
    def client_to_gemini_proxy():
        """Read from client WebSocket, forward to Gemini."""
        try:
            while True:
                raw = ws.receive()
                if raw is None:
                    client_to_gemini.put(None)
                    break
                msg = json.loads(raw)
                client_to_gemini.put(msg)
        except Exception:
            client_to_gemini.put(None)

    # Start client reader in background
    reader_thread = threading.Thread(target=client_to_gemini_proxy, daemon=True)
    reader_thread.start()

    # Main loop: forward Gemini responses to client
    try:
        while True:
            msg = gemini_to_client.get()
            ws.send(msg)
            data = json.loads(msg)
            if data.get("type") == "error":
                break
    except Exception:
        pass
    finally:
        client_to_gemini.put(None)


if __name__ == '__main__':
    print("Starting server on http://localhost:7777")
    print("Make sure to access the application through http://localhost:7777")
    app.run(host='localhost', port=7777, debug=True) 