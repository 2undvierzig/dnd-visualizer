import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import collections
import threading
import time
import os
from datetime import datetime
import math

# --- Konfiguration ---
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16 kHz ist Standard für Whisper
CHUNK_SIZE = 1024 # Größe jedes Audio-Chunks
BUFFER_DURATION = 8 # Sekunden: Erhöht von 3 auf 8 für längere, kohärentere Texte
OVERLAP_DURATION = 2 # Sekunden: Überlappung zwischen Transkriptionen für besseren Kontext

# Datei für Transkriptionen
TRANSCRIPT_FILE = "transcript.txt"

# Debug-Modus
DEBUG = True

# Lautstärkepegel-Variable (global)
current_volume_level = 0  # 0 = niemand spricht, 1-4 = spricht (leise bis sehr laut)
volume_lock = threading.Lock()

# Schwellenwerte für Lautstärkepegel (diese können je nach Mikrofon angepasst werden)
VOLUME_THRESHOLDS = {
    0: 0.005,   # Unter diesem Wert = Stille (niemand spricht)
    1: 0.02,    # Leise sprechen
    2: 0.05,    # Normale Lautstärke
    3: 0.1,     # Laute Stimme
    4: 0.2      # Sehr laut (Schreien/sehr nah am Mikrofon)
}

def debug_print(message):
    """Debug-Ausgabe mit Zeitstempel"""
    if DEBUG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[DEBUG {timestamp}] {message}")

# Modell laden - verwende CPU da es in Tests funktioniert hat
debug_print("Lade Whisper-Modell...")
try:
    model = WhisperModel("base", device="cpu", compute_type="float32")
    debug_print("✓ Whisper-Modell erfolgreich geladen (base/cpu/float32)")
except Exception as e:
    debug_print(f"FEHLER beim Laden des Modells: {e}")
    # Fallback auf tiny
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="float32")
        debug_print("✓ Fallback auf tiny Modell erfolgreich")
    except Exception as e2:
        debug_print(f"KRITISCHER FEHLER: Kein Modell konnte geladen werden: {e2}")
        exit(1)

# Ein Puffer für Audio-Daten
audio_buffer = collections.deque()
buffer_lock = threading.Lock()
transcribing = False # Flag, um Mehrfach-Transkriptionen zu verhindern
file_lock = threading.Lock() # Lock für Thread-sichere Dateischreibung

def calculate_volume_level(audio_data):
    """Berechnet den Lautstärkepegel von Audio-Daten und gibt einen Wert von 0-4 zurück."""
    global current_volume_level
    
    # Konvertiere Audio-Daten zu numpy array und normalisiere
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Berechne RMS (Root Mean Square) als Maß für die Lautstärke
    rms = np.sqrt(np.mean(audio_np**2))
    
    # Bestimme Lautstärkepegel basierend auf Schwellenwerten
    new_level = 0
    if rms >= VOLUME_THRESHOLDS[4]:
        new_level = 4  # Sehr laut
    elif rms >= VOLUME_THRESHOLDS[3]:
        new_level = 3  # Laut
    elif rms >= VOLUME_THRESHOLDS[2]:
        new_level = 2  # Normal
    elif rms >= VOLUME_THRESHOLDS[1]:
        new_level = 1  # Leise
    else:
        new_level = 0  # Stille
    
    # Thread-sichere Aktualisierung der globalen Variable
    with volume_lock:
        current_volume_level = new_level
    
    return new_level, rms

def get_current_volume_level():
    """Gibt den aktuellen Lautstärkepegel thread-sicher zurück."""
    with volume_lock:
        return current_volume_level

def write_to_transcript(text):
    """Schreibt den transkribierten Text in die transcript.txt Datei."""
    with file_lock:
        try:
            with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] {text}\n")
                f.flush()  # Sofort in die Datei schreiben
            debug_print(f"Text gespeichert: {text}")
            print(f"[{timestamp}] Text gespeichert: {text}")
        except Exception as e:
            debug_print(f"Fehler beim Schreiben in die Datei: {e}")

def initialize_transcript_file():
    """Initialisiert die Transkript-Datei mit einem Header."""
    with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"=== Transkriptions-Session gestartet am {start_time} ===\n\n")
    debug_print(f"Transkript-Datei '{TRANSCRIPT_FILE}' initialisiert.")

def record_audio():
    """Nimmt Audio vom Mikrofon auf und fügt es dem Puffer hinzu."""
    debug_print("Starte Audio-Aufnahme Thread...")
    
    p = pyaudio.PyAudio()
    
    # Debug: Zeige ausgewähltes Gerät
    default_input = p.get_default_input_device_info()
    debug_print(f"Verwende Eingabegerät: {default_input['name']}")
    debug_print(f"Standard Sample Rate: {default_input['defaultSampleRate']}")
    
    stream = p.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    debug_print("Audio-Stream geöffnet. Beginne Aufnahme...")
    print("Starte Audioaufnahme. Sprechen Sie jetzt...")

    chunk_counter = 0
    volume_debug_counter = 0
    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # Berechne Lautstärkepegel für diesen Chunk
            volume_level, rms_value = calculate_volume_level(data)
            volume_debug_counter += 1
            
            # Debug: Zeige Lautstärke alle 25 Chunks (ca. alle 1,6 Sekunden bei 16kHz)
            if volume_debug_counter % 25 == 0:
                level_names = ["STILLE", "LEISE", "NORMAL", "LAUT", "SEHR LAUT"]
                debug_print(f"Lautstärkepegel: {volume_level} ({level_names[volume_level]}) - RMS: {rms_value:.4f}")
            
            with buffer_lock:
                audio_buffer.append(data)
                chunk_counter += 1
                
                # Debug: Zeige Puffer-Status alle 50 Chunks
                if chunk_counter % 50 == 0:
                    debug_print(f"Chunk {chunk_counter} hinzugefügt. Puffer-Größe: {len(audio_buffer)}")
                
                # FIXED: Verwende math.ceil für korrekte Chunk-Anzahl
                max_chunks = math.ceil(BUFFER_DURATION * RATE / CHUNK_SIZE)
                while len(audio_buffer) > max_chunks:
                    audio_buffer.popleft()
                    
    except KeyboardInterrupt:
        debug_print("Audio-Aufnahme durch Benutzer beendet.")
        print("Audioaufnahme beendet.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        debug_print("Audio-Resources freigegeben.")

def transcribe_audio():
    """Transkribiert den Audio-Puffer in regelmäßigen Abständen und schreibt in Datei."""
    global transcribing
    debug_print("Starte Transkriptions-Thread...")
    
    transcription_counter = 0
    
    while True:
        # Reduzierte Überprüfungsfrequenz für längere Chunks
        time.sleep(BUFFER_DURATION - OVERLAP_DURATION) # Überprüfe alle 6 Sekunden (8-2)
        transcription_counter += 1
        debug_print(f"Transkriptions-Check #{transcription_counter}")
        
        if transcribing:
            debug_print("Transkription bereits aktiv, überspringe...")
            continue

        with buffer_lock:
            # FIXED: Berechne required_bytes basierend auf tatsächlichen Chunks
            max_chunks = math.ceil(BUFFER_DURATION * RATE / CHUNK_SIZE)
            required_buffer_length_bytes = max_chunks * CHUNK_SIZE * 2  # 2 bytes per sample
            
            current_buffer_length_bytes = sum(len(chunk) for chunk in audio_buffer)

            debug_print(f"Puffer-Status: {current_buffer_length_bytes} bytes vorhanden, {required_buffer_length_bytes} bytes benötigt")
            debug_print(f"Puffer-Chunks: {len(audio_buffer)} / {max_chunks}")

            # Reduziere Schwellenwert für bessere Responsivität 
            min_required_bytes = int(required_buffer_length_bytes * 0.8)  # 80% der Zielgröße

            if current_buffer_length_bytes < min_required_bytes:
                debug_print(f"Nicht genug Audio für Transkription (benötigt mindestens {min_required_bytes} bytes).")
                continue # Nicht genug Audio für eine aussagekräftige Transkription

            debug_print("Ausreichend Audio vorhanden, starte Transkription...")

            # Kombinieren Sie die Chunks zu einem einzigen Numpy-Array
            audio_data = b''.join(list(audio_buffer))
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            debug_print(f"Audio-Array erstellt: {len(audio_np)} samples ({len(audio_np)/RATE:.2f}s)")
            debug_print(f"Audio-Level (RMS): {np.sqrt(np.mean(audio_np**2)):.4f}")

            # VERBESSERT: Behalte Überlappung anstatt kompletter Löschung
            chunks_to_remove = math.ceil((BUFFER_DURATION - OVERLAP_DURATION) * RATE / CHUNK_SIZE)
            debug_print(f"Entferne {chunks_to_remove} Chunks, behalte {OVERLAP_DURATION}s Überlappung")
            
            for _ in range(min(chunks_to_remove, len(audio_buffer))):
                audio_buffer.popleft()
            
            debug_print(f"Audio-Puffer teilweise geleert. Verbleibende Chunks: {len(audio_buffer)}")
            
        transcribing = True # Setzen Sie das Flag vor der Transkription
        debug_print("Transkription gestartet...")

        try:
            # Erweiterte Transkriptions-Parameter für bessere Qualität
            segments, info = model.transcribe(
                audio_np, 
                beam_size=5, 
                language="de", 
                initial_prompt="Hallo",
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500)  # Kürzere Pausen ignorieren
            )
            
            debug_print(f"Transkription abgeschlossen. Sprache: {info.language} (Wahrscheinlichkeit: {info.language_probability:.2f})")
            
            # Sammle alle Segmente zu einem zusammenhängenden Text
            full_text_parts = []
            segment_count = 0
            for segment in segments:
                segment_count += 1
                text = segment.text.strip()
                debug_print(f"Segment {segment_count}: '{text}' ({segment.start:.2f}s - {segment.end:.2f}s)")
                if text:
                    full_text_parts.append(text)
            
            # Kombiniere alle Segmente zu einem Text
            if full_text_parts:
                full_text = " ".join(full_text_parts)
                write_to_transcript(full_text)
            else:
                debug_print("Keine Segmente gefunden - möglicherweise Stille oder zu leise")
                
        except Exception as e:
            debug_print(f"FEHLER bei der Transkription: {e}")
            print(f"Fehler bei der Transkription: {e}")
        finally:
            transcribing = False # Setzen Sie das Flag zurück
            debug_print("Transkription beendet, Flag zurückgesetzt.")

if __name__ == "__main__":
    debug_print("=== STT PROGRAMM STARTET ===")
    
    # Initialisiere die Transkript-Datei
    initialize_transcript_file()
    
    print(f"Transkriptionen werden in '{TRANSCRIPT_FILE}' gespeichert.")
    print("Lautstärkepegel wird in Echtzeit erfasst: 0=Stille, 1=Leise, 2=Normal, 3=Laut, 4=Sehr Laut")
    debug_print(f"Debug-Modus ist aktiviert.")
    debug_print(f"Konfiguration: RATE={RATE}, CHUNK_SIZE={CHUNK_SIZE}, BUFFER_DURATION={BUFFER_DURATION}")
    debug_print(f"Lautstärke-Schwellenwerte: {VOLUME_THRESHOLDS}")
    
    # Starten Sie den Aufnahme-Thread
    record_thread = threading.Thread(target=record_audio)
    record_thread.daemon = True # Der Thread wird beendet, wenn das Hauptprogramm beendet wird
    record_thread.start()
    debug_print("Audio-Thread gestartet.")

    # Starten Sie den Transkriptions-Thread
    transcribe_thread = threading.Thread(target=transcribe_audio)
    transcribe_thread.daemon = True
    transcribe_thread.start()
    debug_print("Transkriptions-Thread gestartet.")

    # Halten Sie das Hauptprogramm am Laufen
    try:
        debug_print("Hauptprogramm läuft. Drücken Sie Ctrl+C zum Beenden.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        debug_print("Programm durch Benutzer beendet.")
        print("Programm beendet.")
        # Schreibe Ende-Marker in die Datei
        with file_lock:
            with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n=== Session beendet am {end_time} ===\n")