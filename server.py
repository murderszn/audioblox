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

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import time
import io
import asyncio
import threading
from queue import Queue
import re
import wave

# Initialize Flask app with CORS support
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:7777", "http://127.0.0.1:7777"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Session-ID"]
    }
})

# Load environment variables from .env file
load_dotenv()

# Lazy initializer for GenAI client
genai_client = None

def get_genai_client():
    global genai_client
    if genai_client is None:
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENERATIVE_LANGUAGE_API_KEY')
        if not api_key:
            raise Exception("GOOGLE_API_KEY is not configured in the environment variables.")
        genai_client = genai.Client(api_key=api_key)
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
    Clean text by removing markdown formatting and other symbols that shouldn't be spoken.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text ready for speech synthesis
    """
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)      # Italic
    text = re.sub(r'\_(.+?)\_', r'\1', text)      # Underscore emphasis
    
    # Remove markdown links
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    
    # Remove markdown code blocks and inline code
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    
    # Remove markdown headers
    text = re.sub(r'#{1,6}\s+', '', text)
    
    # Remove bullet points and numbering
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def generate_speech(text, voice_name="Puck"):
    """
    Generate high-quality speech from text using Google's native Gemini conversational TTS model.
    
    Args:
        text (str): The text to convert to speech
        voice_name (str): The name of the voice configuration to use
        
    Returns:
        bytes: The audio content in WAV format
        
    Raises:
        Exception: If speech generation fails
    """
    try:
        # Clean the text before synthesis
        cleaned_text = clean_text_for_speech(text)
        
        # Call Gemini TTS Model
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
            
        # Convert raw linear PCM (16-bit, 24kHz mono) to WAV format
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, 'wb') as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16-bit
            wav_file.setframerate(24000)  # 24kHz
            wav_file.writeframes(audio_data)
            
        return wav_buf.getvalue()
    except Exception as e:
        raise Exception(f"Failed to generate speech: {str(e)}")

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
                top_p=1.0,
                top_k=1,
                max_output_tokens=2048,
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
    """Generate speech from text"""
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

        audio_content = generate_speech(text, voice_name=voice_name)
        
        return send_file(
            io.BytesIO(audio_content),
            mimetype='audio/wav',
            as_attachment=True,
            download_name='response.wav'
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

if __name__ == '__main__':
    print("Starting server on http://localhost:7777")
    print("Make sure to access the application through http://localhost:7777")
    app.run(host='localhost', port=7777, debug=True) 