import os
import tempfile
import whisper
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
import json
from datetime import datetime, timedelta
import subprocess
import ffmpeg
import threading
import time
import psutil
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import shutil
from pathlib import Path
import hashlib
import math
import traceback
from collections import Counter

# Configurare director de date local
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 * 1024  # 50GB max
app.config['UPLOAD_FOLDER'] = DATA_DIR
app.config['CHUNK_FOLDER'] = os.path.join(DATA_DIR, 'chunk_uploads')
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'm4v', 'mp3', 'wav', 'mpeg', 'webm', 'mxf', 'wmv', 'flv'}
app.config['SECRET_KEY'] = 'whisper-transcriber-secret-key-2024'
app.config['CHUNK_SIZE'] = 10 * 1024 * 1024  # 10MB per chunk
app.config['MAX_FILE_SIZE'] = 50 * 1024 * 1024 * 1024  # 50GB
app.config['PROCESS_TIMEOUT'] = 7200  # 2 ore timeout pentru procesare

# Dicționar pentru modele încărcate
loaded_models = {}
model_lock = threading.Lock()

# Modele de traducere
translation_models = {}
translation_lock = threading.Lock()

# Dicționar pentru sesiuni de upload
upload_sessions = {}
upload_lock = threading.Lock()

# Managementul task-urilor în background
processing_tasks = {}
tasks_lock = threading.Lock()

# Cache pentru suport hardware
_hardware_caps = {
    'nvenc': None
}
_caps_lock = threading.Lock()

def is_nvenc_available():
    """Verifică dacă h264_nvenc este disponibil în FFmpeg"""
    global _hardware_caps
    with _caps_lock:
        if _hardware_caps['nvenc'] is not None:
            return _hardware_caps['nvenc']

        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
            _hardware_caps['nvenc'] = 'h264_nvenc' in result.stdout
        except:
            _hardware_caps['nvenc'] = False

        return _hardware_caps['nvenc']

# Opțiuni modele disponibile
AVAILABLE_MODELS = {
    'tiny': 'Tiny (Rapid, 39M) - Pentru teste rapide',
    'base': 'Base (Bun, 74M) - Balanță bună viteză/calitate',
    'small': 'Small (Mai bun, 244M) - Recomandat pentru română',
    'medium': 'Medium (Excelent, 769M) - Calitate foarte bună',
    'large': 'Large (Best, 1550M) - Calitate profesională',
    'large-v3': 'Large v3 (Latest, 1550M) - Cel mai recent model'
}

# Model implicit
DEFAULT_MODEL = 'small'

# Limbi suportate de Whisper și pentru traducere
SUPPORTED_LANGUAGES = {
    'auto': 'Detectare automată',
    'ro': 'Română',
    'en': 'Engleză',
    'fr': 'Franceză',
    'de': 'Germană',
    'es': 'Spaniolă',
    'it': 'Italiană',
    'ru': 'Rusă',
    'ja': 'Japoneză',
    'zh': 'Chineză',
    'ar': 'Arabă',
    'bg': 'Bulgară',
    'cs': 'Cehă',
    'da': 'Daneză',
    'el': 'Greacă',
    'fi': 'Finlandeză',
    'he': 'Ebraică',
    'hi': 'Hindi',
    'hu': 'Maghiară',
    'id': 'Indoneziană',
    'ko': 'Coreeană',
    'nl': 'Olandeză',
    'no': 'Norvegiană',
    'pl': 'Poloneză',
    'pt': 'Portugheză',
    'sv': 'Suedeză',
    'sk': 'Slovacă',
    'sl': 'Slovenă',
    'tr': 'Turcă',
    'uk': 'Ucraineană'
}

# Modele de traducere mai bune (NLLB-200 pentru traduceri multilingve de calitate)
TRANSLATION_MODELS_CONFIG = {
    # Model NLLB-200 (No Language Left Behind) - 200 de limbi, calitate bună
    'nllb': {
        'name': 'facebook/nllb-200-distilled-600M',
        'display_name': 'NLLB-200 (Multilingv, 600M)',
        'languages': {
            'en': 'eng_Latn', 'ro': 'ron_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn',
            'es': 'spa_Latn', 'it': 'ita_Latn', 'ru': 'rus_Cyrl', 'zh': 'zho_Hans',
            'ja': 'jpn_Jpan', 'ko': 'kor_Hang', 'ar': 'ara_Arab', 'hi': 'hin_Deva',
            'pt': 'por_Latn', 'nl': 'nld_Latn', 'pl': 'pol_Latn', 'tr': 'tur_Latn',
            'sv': 'swe_Latn', 'da': 'dan_Latn', 'fi': 'fin_Latn', 'no': 'nob_Latn',
            'cs': 'ces_Latn', 'hu': 'hun_Latn', 'bg': 'bul_Cyrl', 'el': 'ell_Grek',
            'uk': 'ukr_Cyrl', 'vi': 'vie_Latn', 'th': 'tha_Thai', 'he': 'heb_Hebr',
            'id': 'ind_Latn', 'ms': 'zsm_Latn', 'fa': 'pes_Arab', 'ur': 'urd_Arab',
            'sw': 'swh_Latn', 'sk': 'slk_Latn', 'sl': 'slv_Latn'
        }
    },
    # Modele MarianMT (specifice perechilor de limbi) - calitate foarte bună pentru perechile specifice
    'marian': {
        'models': {
            'en-ro': 'Helsinki-NLP/opus-mt-en-ro',
            # 'ro-en' nu există ca model separat, folosim NLLB-200 pentru traducerea inversă
            'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
            'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
            'en-de': 'Helsinki-NLP/opus-mt-en-de',
            'de-en': 'Helsinki-NLP/opus-mt-de-en',
            'en-es': 'Helsinki-NLP/opus-mt-en-es',
            'es-en': 'Helsinki-NLP/opus-mt-es-en',
            'en-it': 'Helsinki-NLP/opus-mt-en-it',
            'it-en': 'Helsinki-NLP/opus-mt-it-en',
            'en-ru': 'Helsinki-NLP/opus-mt-en-ru',
            'ru-en': 'Helsinki-NLP/opus-mt-ru-en',
            'en-sk': 'Helsinki-NLP/opus-mt-en-sk',
            'sk-en': 'Helsinki-NLP/opus-mt-sk-en',
            'en-sl': 'Helsinki-NLP/opus-mt-en-sl',
            'sl-en': 'Helsinki-NLP/opus-mt-sl-en',
        }
    }
}

# Limbi pentru traducere cu etichete ușor de înțeles
TRANSLATION_LANGUAGES = {
    'en': 'Engleză',
    'ro': 'Română',
    'fr': 'Franceză',
    'de': 'Germană',
    'es': 'Spaniolă',
    'it': 'Italiană',
    'ru': 'Rusă',
    'zh': 'Chineză',
    'ja': 'Japoneză',
    'ko': 'Coreeană',
    'ar': 'Arabă',
    'hi': 'Hindi',
    'pt': 'Portugheză',
    'nl': 'Olandeză',
    'pl': 'Poloneză',
    'tr': 'Turcă',
    'sv': 'Suedeză',
    'sk': 'Slovacă',
    'sl': 'Slovenă',
    'da': 'Daneză',
    'fi': 'Finlandeză',
    'no': 'Norvegiană',
    'cs': 'Cehă',
    'hu': 'Maghiară',
    'bg': 'Bulgară',
    'el': 'Greacă',
    'uk': 'Ucraineană',
    'vi': 'Vietnameză',
    'th': 'Thai',
    'he': 'Ebraică',
    'id': 'Indoneziană',
    'ms': 'Malaeză',
    'fa': 'Persană',
    'ur': 'Urdu',
    'sw': 'Swahili'
}

# Creează folderele necesare
os.makedirs(app.config['CHUNK_FOLDER'], exist_ok=True)

def load_model(model_name=DEFAULT_MODEL):
    """Încarcă modelul Whisper specificat"""
    global loaded_models
    
    with model_lock:
        if model_name not in loaded_models:
            print(f"Se încarcă modelul Whisper: {model_name}...")
            try:
                start_time = time.time()
                
                # Setăm device-ul automat (CUDA dacă e disponibil)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Folosind device: {device}")
                
                # Încărcăm modelul
                model = whisper.load_model(model_name, device=device)
                load_time = time.time() - start_time
                
                loaded_models[model_name] = {
                    'model': model,
                    'device': device,
                    'load_time': load_time
                }
                
                print(f"✓ Model {model_name} încărcat în {load_time:.1f} secunde pe {device}")
                
                # Curățăm memoria GPU dacă e necesar
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"✗ Eroare la încărcarea modelului {model_name}: {str(e)}")
                # Fallback la CPU dacă CUDA dă eroare
                try:
                    print("Încerc încărcare pe CPU...")
                    model = whisper.load_model(model_name, device="cpu")
                    loaded_models[model_name] = {
                        'model': model,
                        'device': 'cpu',
                        'load_time': time.time() - start_time
                    }
                    print(f"✓ Model {model_name} încărcat pe CPU")
                except Exception as e2:
                    print(f"✗ Eroare critică: {str(e2)}")
                    # Încarcă modelul base ca fallback
                    if model_name != 'base':
                        print(f"Încerc fallback la modelul 'base'...")
                        return load_model('base')
                    else:
                        raise
                        
        return loaded_models[model_name]

def load_translation_model(source_lang, target_lang):
    """Încarcă modelul de traducere pentru o pereche de limbi"""
    global translation_models
    
    model_key = f"{source_lang}-{target_lang}"
    
    with translation_lock:
        if model_key in translation_models:
            return translation_models[model_key]
        
        print(f"Se încarcă modelul de traducere: {source_lang}->{target_lang}...")
        start_time = time.time()
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # MODELE SPECIFICE PENTRU FIECARE PERECHE
            model_map = TRANSLATION_MODELS_CONFIG['marian']['models'].copy()
            # Mapare specială pentru Romanian -> English (ROMANCE-en include ro)
            if 'ro-en' not in model_map:
                model_map['ro-en'] = 'Helsinki-NLP/opus-mt-ROMANCE-en'
            
            if model_key in model_map:
                model_name = model_map[model_key]
                print(f"Încarc modelul Opus-MT: {model_name}")
                
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name).to(device)
                model_type = 'marian'
            else:
                # Fallback la NLLB-200 (mai modern și suportă mai multe limbi)
                model_name = TRANSLATION_MODELS_CONFIG['nllb']['name']
                print(f"Încarc modelul NLLB-200: {model_name}")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                model_type = 'nllb'
            
            load_time = time.time() - start_time
            
            translation_models[model_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': device,
                'load_time': load_time,
                'source': source_lang,
                'target': target_lang,
                'model_name': model_name,
                'model_type': model_type
            }
            
            print(f"✓ Model traducere {model_key} încărcat în {load_time:.1f} secunde pe {device}")
            return translation_models[model_key]
            
        except Exception as e:
            print(f"✗ Eroare la încărcarea modelului de traducere: {str(e)}")
            
            # Fallback: încercă să încarce pe CPU
            try:
                print("Încerc încărcare pe CPU...")
                device = "cpu"
                
                if model_key in model_map:
                    model_name = model_map[model_key]
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name).to(device)
                    model_type = 'marian'
                else:
                    model_name = TRANSLATION_MODELS_CONFIG['nllb']['name']
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
                    model_type = 'nllb'
                
                translation_models[model_key] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': device,
                    'load_time': time.time() - start_time,
                    'source': source_lang,
                    'target': target_lang,
                    'model_name': model_name,
                    'model_type': model_type
                }
                
                print(f"✓ Model traducere {model_key} încărcat pe CPU")
                return translation_models[model_key]
                
            except Exception as e2:
                print(f"✗ Eroare critică la încărcarea modelului: {str(e2)}")
                return None

def translate_segment_batch(segments, source_lang, target_lang, batch_size=5):
    """Traduce un batch de segmente păstrând timecode-ul"""
    if not segments or source_lang == target_lang:
        return segments
    
    try:
        # Încarcă modelul de traducere
        model_data = load_translation_model(source_lang, target_lang)
        
        if not model_data:
            print(f"✗ Nu există model de traducere pentru {source_lang}->{target_lang}")
            return segments
        
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        device = model_data['device']
        model_type = model_data['model_type']
        
        translated_segments = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            batch_texts = [seg['text'] for seg in batch]
            
            try:
                if model_type == 'marian':
                    # MarianMT/Opus-MT - direct translation
                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
                    
                elif model_type == 'nllb':
                    # NLLB-200
                    src_code = TRANSLATION_MODELS_CONFIG['nllb']['languages'].get(source_lang, f"{source_lang}_Latn")
                    tgt_code = TRANSLATION_MODELS_CONFIG['nllb']['languages'].get(target_lang, f"{target_lang}_Latn")

                    if hasattr(tokenizer, 'src_lang'):
                        tokenizer.src_lang = src_code

                    forced_bos_token_id = None
                    try:
                        if hasattr(tokenizer, 'get_lang_id'):
                            forced_bos_token_id = tokenizer.get_lang_id(tgt_code)
                        elif hasattr(tokenizer, 'lang_code_to_id') and tgt_code in tokenizer.lang_code_to_id:
                            forced_bos_token_id = tokenizer.lang_code_to_id[tgt_code]
                    except:
                        pass

                    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    
                    gen_kwargs = {"max_length": 512, "num_beams": 4, "early_stopping": True}
                    if forced_bos_token_id is not None:
                        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

                    translated = model.generate(**inputs, **gen_kwargs)
                    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
                
                else:
                    # Fallback pentru alte modele
                    translated_texts = batch_texts
                
                # Creează segmentele traduse cu timecode-uri originale
                for j, seg in enumerate(batch):
                    if j < len(translated_texts):
                        translated_seg = seg.copy()
                        translated_seg['text'] = translated_texts[j].strip()
                        translated_segments.append(translated_seg)
                    else:
                        # Fallback: păstrează textul original
                        translated_segments.append(seg)
                        
            except Exception as e:
                print(f"Eroare la traducerea batch-ului {i}: {str(e)}")
                # În caz de eroare, păstrează segmentele originale
                translated_segments.extend(batch)
        
        return translated_segments
        
    except Exception as e:
        print(f"✗ Eroare la traducere: {str(e)}")
        return segments

def translate_multilingual_segments(segments, target_lang, process_id=None):
    """
    Traduce segmente care pot fi în mai multe limbi sursă.
    Detectează automat limba fiecărui segment și folosește modelul potrivit.
    """
    if not segments:
        return segments

    print(f"🌍 Traducere multilingvă către {target_lang}...")
    print(f"  Segmente totale: {len(segments)}")

    translated_segments = []
    language_groups = {}

    # Grupăm segmentele după limba sursă
    for seg in segments:
        # Folosește detected_language dacă este disponibil (setat în process_large_file)
        source_lang = seg.get('detected_language', seg.get('language', 'en'))
        if source_lang not in language_groups:
            language_groups[source_lang] = []
        language_groups[source_lang].append(seg)

    print(f"  Limbi detectate în segmente: {list(language_groups.keys())}")

    # Traducem fiecare grup în parte
    for source_lang, group_segments in language_groups.items():
        if source_lang == target_lang:
            # Nu traducem dacă e aceeași limbă
            print(f"  ⏭️  Păstrez {len(group_segments)} segmente în {source_lang} (aceeași limbă)")
            for seg in group_segments:
                translated_seg = seg.copy()
                translated_seg['original'] = False
                translated_seg['target_language'] = target_lang
                translated_seg['source_language'] = source_lang
                translated_segments.append(translated_seg)
        else:
            # Traducem din source_lang în target_lang
            print(f"  🔄 Traduc {len(group_segments)} segmente din {source_lang} în {target_lang}...")

            # Pregătim segmentele pentru traducere
            whisper_segments = []
            for seg in group_segments:
                whisper_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                })

            try:
                translated = translate_segments(whisper_segments, source_lang, target_lang)

                for i, seg in enumerate(translated):
                    translated_seg = group_segments[i].copy()
                    translated_seg['text'] = seg['text']
                    translated_seg['original'] = False
                    translated_seg['target_language'] = target_lang
                    translated_seg['source_language'] = source_lang
                    translated_segments.append(translated_seg)

            except Exception as e:
                print(f"  ❌ Eroare la traducere din {source_lang}: {str(e)}")
                # Fallback: păstrăm originalul
                for seg in group_segments:
                    translated_seg = seg.copy()
                    translated_seg['original'] = False
                    translated_seg['target_language'] = target_lang
                    translated_seg['source_language'] = source_lang
                    translated_segments.append(translated_seg)

    # Sortăm după timp
    translated_segments.sort(key=lambda x: x['start'])

    return translated_segments

def translate_segments(segments, source_lang, target_lang):
    """Traduce toate segmentele păstrând timecode-ul și structura"""
    if not segments or source_lang == target_lang:
        return segments
    
    print(f"Încep traducerea din {source_lang} în {target_lang}...")
    print(f"Număr segmente: {len(segments)}")
    start_time = time.time()
    
    try:
        # Împarte segmentele în grupuri de lungimi similare pentru o traducere mai bună
        translated_segments = []
        
        # Grupează segmentele scurte pentru traducere mai eficientă
        short_segments = []
        long_segments = []
        
        for seg in segments:
            text_length = len(seg['text'])
            if text_length < 50:  # Segmente scurte
                short_segments.append(seg)
            else:  # Segmente lungi
                long_segments.append(seg)
        
        # Traduce segmentele scurte în batch-uri
        if short_segments:
            print(f"Traduc {len(short_segments)} segmente scurte...")
            translated_short = translate_segment_batch(short_segments, source_lang, target_lang, batch_size=10)
            translated_segments.extend(translated_short)
        
        # Traduce segmentele lungi individual pentru mai multă precizie
        if long_segments:
            print(f"Traduc {len(long_segments)} segmente lungi...")
            for seg in long_segments:
                try:
                    # Traduce fiecare segment lung individual
                    batch_result = translate_segment_batch([seg], source_lang, target_lang, batch_size=1)
                    if batch_result:
                        translated_segments.append(batch_result[0])
                    else:
                        translated_segments.append(seg)
                except:
                    translated_segments.append(seg)
        
        # Asigură-te că ordinea este păstrată
        translated_segments.sort(key=lambda x: x['start'])
        
        translation_time = time.time() - start_time
        print(f"✓ Traducere completă în {translation_time:.1f} secunde")
        
        return translated_segments
        
    except Exception as e:
        print(f"✗ Eroare la traducere: {str(e)}")
        # În caz de eroare, returnează segmentele originale
        return segments

def translate_text(text, source_lang, target_lang):
    """Traduce text folosind modelul corespunzător"""
    if not text or not text.strip() or source_lang == target_lang:
        return text
    
    text = text.strip()
    
    try:
        # Încarcă modelul de traducere
        model_data = load_translation_model(source_lang, target_lang)
        
        if not model_data:
            return text
        
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        device = model_data['device']
        model_type = model_data['model_type']
        
        if model_type == 'marian':
            # MarianMT
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            
        elif model_type == 'nllb':
            # NLLB-200
            src_code = TRANSLATION_MODELS_CONFIG['nllb']['languages'].get(source_lang, f"{source_lang}_Latn")
            tgt_code = TRANSLATION_MODELS_CONFIG['nllb']['languages'].get(target_lang, f"{target_lang}_Latn")
            
            # Setează limba sursă
            if hasattr(tokenizer, 'src_lang'):
                tokenizer.src_lang = src_code
            
            # Obține ID-ul limbii țintă
            forced_bos_token_id = None
            try:
                if hasattr(tokenizer, 'get_lang_id'):
                    forced_bos_token_id = tokenizer.get_lang_id(tgt_code)
                elif hasattr(tokenizer, 'lang_code_to_id') and tgt_code in tokenizer.lang_code_to_id:
                    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_code]
            except:
                pass
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            if forced_bos_token_id is not None:
                translated = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            else:
                translated = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        else:
            result = text
        
        return result.strip()
        
    except Exception as e:
        print(f"Eroare la traducere text: {str(e)}")
        return text

def get_model_info(model_name):
    """Returnează informații despre model"""
    model_sizes = {
        'tiny': '39 MB',
        'base': '74 MB', 
        'small': '244 MB',
        'medium': '769 MB',
        'large': '1.5 GB',
        'large-v3': '1.5 GB'
    }
    
    model_descriptions = {
        'tiny': 'Cel mai rapid, potrivit pentru teste',
        'base': 'Bun echilibru între viteză și calitate',
        'small': 'Recomandat pentru limba română',
        'medium': 'Calitate foarte bună, mai lent',
        'large': 'Calitate profesională, necesită multă memorie',
        'large-v3': 'Cel mai recent model, suportă mai multe limbi'
    }
    
    return {
        'size': model_sizes.get(model_name, 'N/A'),
        'description': model_descriptions.get(model_name, ''),
        'name': model_name
    }

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_process_dir(process_id):
    """Returnează directorul dedicat pentru un proces de transcriere"""
    if not process_id:
        return None
    # Ne asigurăm că ID-ul este sigur pentru sistemul de fișiere
    safe_id = "".join([c for c in str(process_id) if c.isalnum() or c == '-'])
    if not safe_id:
        return None
    return os.path.join(app.config['UPLOAD_FOLDER'], f'process_{safe_id}')

def update_task_status(process_id, status, progress=0, message='', result=None):
    """Actualizează statusul unui task pe disc și în memorie"""
    process_dir = get_process_dir(process_id)
    if not process_dir:
        return

    os.makedirs(process_dir, exist_ok=True)
    filepath = os.path.join(process_dir, 'status.json')

    data = {
        'process_id': process_id,
        'status': status,
        'progress': progress,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'last_heartbeat': time.time(),
        'result': result
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with tasks_lock:
        processing_tasks[process_id] = data

def get_task_status(process_id):
    """Obține statusul curent al unui task"""
    with tasks_lock:
        if process_id in processing_tasks:
            return processing_tasks[process_id]

    process_dir = get_process_dir(process_id)
    if not process_dir:
        return None

    filepath = os.path.join(process_dir, 'status.json')
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                with tasks_lock:
                    processing_tasks[process_id] = data
                return data
        except:
            pass
    return None

def run_ffmpeg_with_progress(cmd, process_id, task_name, total_duration=None):
    """Rulează o comandă ffmpeg și raportează progresul"""
    print(f"Running ffmpeg with progress: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL, # Redirecționăm stdout către DEVNULL pentru a evita deadlock-ul pe pipe
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    def parse_stderr():
        current_progress = 0
        for line in process.stderr:
            if "time=" in line:
                try:
                    time_str = line.split("time=")[1].split()[0]
                    h, m, s = time_str.split(':')
                    elapsed_seconds = int(h) * 3600 + int(m) * 60 + float(s)

                    if total_duration and total_duration > 0:
                        progress = min(99, int((elapsed_seconds / total_duration) * 100))
                        if progress > current_progress:
                            current_progress = progress
                            update_task_status(process_id, 'processing', progress, f"{task_name}: {progress}%")
                except:
                    pass

    stderr_thread = threading.Thread(target=parse_stderr)
    stderr_thread.start()
    process.wait()
    stderr_thread.join()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

def get_video_duration(video_path):
    """Obține durata video folosind ffprobe"""
    try:
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return None

def convert_to_wav(input_path, process_id=None):
    """Converteste orice fișier audio/video în WAV pentru procesare"""
    temp_wav = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{uuid.uuid4()}.wav')
    
    try:
        # Mai întâi verifică dacă fișierul are audio
        check_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            input_path
        ]
        
        try:
            result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
            has_audio = result.stdout.strip() == 'audio'
        except:
            has_audio = False
        
        if not has_audio:
            print("Fișierul video nu are audio. Încerc procesare directă...")
            return input_path
        
        duration = get_video_duration(input_path)

        # Folosim subprocess direct pentru a evita problemele cu ffmpeg-python
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vn',                     # Ignoră video
            '-acodec', 'pcm_s16le',    # Codec audio
            '-ac', '1',                # Mono
            '-ar', '16000',            # Sample rate 16kHz
            '-y',                      # Overwrite output
            temp_wav
        ]
        
        if process_id:
            run_ffmpeg_with_progress(cmd, process_id, "Extragere audio", duration)
        else:
            subprocess.run(cmd, check=True, capture_output=True)
        
        # Verifică dacă fișierul WAV a fost creat
        if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
            # Încercare alternativă - folosește doar extrageri de audio
            alt_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-map', '0:a',         # Folosește doar audio streams
                '-c:a', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-loglevel', 'error',
                '-y',
                temp_wav
            ]
            
            print(f"Trying alternative ffmpeg command: {' '.join(alt_cmd)}")
            
            result = subprocess.run(
                alt_cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Alternative ffmpeg output: {result.stderr[:200] if result.stderr else 'No output'}")
            
            if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                # Ultima încercare - folosește aac decoding dacă e necesar
                final_cmd = [
                    'ffmpeg',
                    '-i', input_path,
                    '-c:a', 'pcm_s16le',
                    '-strict', '-2',    # Permite experimental codecs
                    '-ac', '1',
                    '-ar', '16000',
                    '-loglevel', 'error',
                    '-y',
                    temp_wav
                ]
                
                print(f"Trying final ffmpeg command: {' '.join(final_cmd)}")
                
                result = subprocess.run(
                    final_cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                    print("Fișierul WAV rezultat este gol, folosesc fișierul original")
                    return input_path
        
        print(f"✓ Audio convertit cu succes: {os.path.getsize(temp_wav)} bytes")
        return temp_wav
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Eroare ffmpeg (exit code {e.returncode}): {e.stderr[:500] if e.stderr else str(e)}")
        print("Folosesc fișierul original pentru transcriere...")
        return input_path
    except subprocess.TimeoutExpired:
        print("✗ Timeout la conversia audio")
        print("Folosesc fișierul original pentru transcriere...")
        return input_path
    except Exception as e:
        print(f"✗ Eroare generală la conversia audio: {str(e)}")
        print("Folosesc fișierul original pentru transcriere...")
        return input_path

def extract_video_preview(video_path, preview_dir):
    """Extrage cadre pentru preview video"""
    try:
        # Creează un frame din mijlocul video-ului
        output_path = os.path.join(preview_dir, 'preview.jpg')
        
        # Obține durata video folosind ffprobe
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                     '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        
        # Extrage frame la 25% din durată (evită începutul și sfârșitul)
        preview_time = duration * 0.25 if duration > 2 else 0
        
        extract_cmd = [
            'ffmpeg',
            '-ss', str(preview_time),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',  # Calitate bună
            '-loglevel', 'error',
            '-y',
            output_path
        ]
        
        subprocess.run(extract_cmd, capture_output=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return None
            
    except Exception as e:
        print(f"Eroare la extragerea preview: {e}")
        return None

def extract_video_for_preview(video_path, output_dir):
    """Extrage o versiune redusă a video-ului pentru preview (pentru formate non-MP4)"""
    try:
        output_path = os.path.join(output_dir, 'preview_video.mp4')
        
        # Obține informații despre video folosind ffprobe
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                     'stream=width,height,duration,codec_type', 
                     '-of', 'json', video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        video_stream = next((s for s in probe_data.get('streams', []) 
                           if s.get('codec_type') == 'video'), None)
        
        if not video_stream:
            return None
        
        # Dimensiuni reduse
        width = int(video_stream.get('width', 1280))
        height = int(video_stream.get('height', 720))
        
        max_width = 720
        if width > max_width:
            height = int(height * (max_width / width))
            width = max_width
        
        # Extrage primele 30 de secunde pentru preview
        duration = float(video_stream.get('duration', 30))
        preview_duration = min(duration, 30)
        
        # Creează video redus
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-t', str(preview_duration),
            '-vf', f'scale={width}:{height}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '28',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-loglevel', 'error',
            '-y',
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            return None
            
    except Exception as e:
        print(f"Eroare la extragerea video pentru preview: {e}")
        return None

def convert_to_mp4_for_playback(video_path, output_dir, process_id=None):
    """Convertește orice format video la MP4 pentru playback în browser"""
    try:
        output_path = os.path.join(output_dir, 'playback.mp4')
        duration = get_video_duration(video_path)
        
        # Verifică dacă NVENC este disponibil pentru accelerare hardware
        use_nvenc = is_nvenc_available()

        if use_nvenc:
            print("Folosesc accelerare hardware NVENC pentru preview...")
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-i', video_path,
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',          # Cel mai rapid preset NVENC
                '-tune', 'ull',           # Ultra-low latency
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            try:
                if process_id:
                    run_ffmpeg_with_progress(cmd, process_id, "Pregătire video (MP4)", duration)
                else:
                    subprocess.run(cmd, capture_output=True, check=True)

                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    return output_path
                print("NVENC a eșuat sau a produs un fișier gol. Încerc fallback software...")
            except Exception as e:
                print(f"Eroare la NVENC: {e}. Încerc fallback software...")

        # Fallback sau software encoding implicit
        print("Folosesc encoding software pentru preview...")
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',   # Mai rapid pentru preview software
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        if process_id:
            run_ffmpeg_with_progress(cmd, process_id, "Pregătire video (MP4)", duration)
        else:
            subprocess.run(cmd, capture_output=True, check=True)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            # Încercare alternativă
            alt_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-c:v', 'copy',  # Copy video stream dacă e posibil
                '-c:a', 'aac',
                '-movflags', '+faststart',
                '-loglevel', 'error',
                '-y',
                output_path
            ]
            
            try:
                subprocess.run(alt_cmd, capture_output=True, check=True)
            except:
                return None
            
            return output_path if os.path.exists(output_path) else None
            
    except Exception as e:
        print(f"Eroare la conversia la MP4: {e}")
        return None

def format_timestamp(seconds):
    """Formatează timpul în format SRT (HH:MM:SS,mmm)"""
    if seconds is None:
        return "00:00:00,000"
    
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    seconds_int = int(td.total_seconds() % 60)
    milliseconds = int((td.total_seconds() - int(td.total_seconds())) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

def write_srt(segments, output_path):
    """Scrie segmentele în format SRT"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, start=1):
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        return True
    except Exception as e:
        print(f"Eroare la scrierea SRT: {str(e)}")
        return False

def split_text_by_duration(text, duration, max_chars, min_segment_duration=1.0):
    """Împarte textul în bucăți pe baza duratei și numărului de caractere"""
    words = text.split()
    if not words:
        return [text]
    
    # Calculează durata maximă recomandată pe baza vitezei de vorbire (3 cuvinte/secundă)
    words_per_second = 3
    max_words_for_duration = int(duration * words_per_second)
    
    # Limitează și după caractere
    max_words_for_chars = max_chars // 6  # Presupunem 6 caractere/cuvânt în medie
    
    # Alege limita mai strictă
    max_words = min(max_words_for_duration, max_words_for_chars, 20)
    
    chunks = []
    current_chunk = []
    current_chars = 0
    
    for word in words:
        word_length = len(word)
        
        # Dacă adăugarea acestui cuvânt ar depăși limitele, salvează chunk-ul curent
        if (current_chars + word_length + 1 > max_chars or 
            len(current_chunk) >= max_words):
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_chars = word_length
        else:
            current_chunk.append(word)
            current_chars += word_length + 1  # +1 pentru spațiu
    
    # Adaugă ultimul chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Asigură-te că nu avem chunk-uri prea scurte (combinează-le dacă e necesar)
    final_chunks = []
    i = 0
    while i < len(chunks):
        if i < len(chunks) - 1 and len(chunks[i]) < (max_chars // 3):
            # Combinează cu următorul chunk dacă e prea scurt
            combined = f"{chunks[i]} {chunks[i+1]}"
            if len(combined) <= max_chars:
                final_chunks.append(combined)
                i += 2
            else:
                final_chunks.append(chunks[i])
                i += 1
        else:
            final_chunks.append(chunks[i])
            i += 1
    
    return final_chunks

def adjust_segmentation_algorithm(segments, min_duration=1.0, max_duration=5.0, max_chars=80):
    """Ajustează segmentarea pentru a fi mai potrivită pentru subtitrări"""
    adjusted_segments = []
    
    for segment in segments:
        text = segment['text'].strip()
        start = segment['start']
        end = segment['end']
        duration = end - start
        
        # Dacă segmentul e prea scurt, îl combinăm cu următorul (dacă există)
        if duration < min_duration and adjusted_segments:
            last_segment = adjusted_segments[-1]
            last_segment['end'] = end
            
            # Combină textul fără a duplica spații
            combined_text = f"{last_segment['text']} {text}".strip()
            # Elimină spații multiple
            combined_text = ' '.join(combined_text.split())
            last_segment['text'] = combined_text
        # Dacă segmentul e prea lung sau textul e prea lung, îl împărțim
        elif duration > max_duration or len(text) > max_chars:
            # Împarte textul în bucăți rezonabile
            text_segments = split_text_by_duration(text, duration, max_chars, min_duration)
            
            if len(text_segments) > 1:
                # Distribuie timpul uniform între segmentele noi
                segment_duration = duration / len(text_segments)
                for i, text_segment in enumerate(text_segments):
                    seg_start = start + (i * segment_duration)
                    seg_end = start + ((i + 1) * segment_duration)
                    adjusted_segments.append({
                        'start': seg_start,
                        'end': seg_end,
                        'text': text_segment.strip()
                    })
            else:
                adjusted_segments.append(segment)
        else:
            adjusted_segments.append(segment)
    
    return adjusted_segments

# ============================================================================
# FUNCȚII PENTRU UPLOAD SEGMENTAT
# ============================================================================

def init_upload_session(file_name, file_size, total_chunks):
    """Initializează o sesiune de upload"""
    session_id = str(uuid.uuid4())
    chunk_dir = os.path.join(app.config['CHUNK_FOLDER'], session_id)
    os.makedirs(chunk_dir, exist_ok=True)
    
    upload_session = {
        'id': session_id,
        'file_name': file_name,
        'file_size': file_size,
        'total_chunks': total_chunks,
        'received_chunks': [],
        'chunk_dir': chunk_dir,
        'start_time': time.time(),
        'status': 'uploading',
        'progress': 0
    }
    
    with upload_lock:
        upload_sessions[session_id] = upload_session
    
    return upload_session

def update_upload_progress(session_id, chunk_number):
    """Actualizează progresul upload-ului"""
    with upload_lock:
        if session_id in upload_sessions:
            session = upload_sessions[session_id]
            session['received_chunks'].append(chunk_number)
            session['progress'] = len(session['received_chunks']) / session['total_chunks'] * 100
            return session['progress']
    return 0

def save_chunk(session_id, chunk_number, chunk_data):
    """Salvează un chunk de date"""
    with upload_lock:
        if session_id not in upload_sessions:
            return False
        
        session = upload_sessions[session_id]
        chunk_path = os.path.join(session['chunk_dir'], f'chunk_{chunk_number:06d}')
        
        try:
            with open(chunk_path, 'wb') as f:
                f.write(chunk_data)
            
            # Verifică dacă toate chunk-urile au fost primite
            received_count = len(session['received_chunks'])
            if received_count >= session['total_chunks']:
                session['status'] = 'complete'
                session['end_time'] = time.time()
            
            return True
        except Exception as e:
            print(f"Eroare la salvarea chunk-ului {chunk_number}: {str(e)}")
            return False

def combine_chunks(session_id):
    """Combină toate chunk-urile într-un fișier complet"""
    with upload_lock:
        if session_id not in upload_sessions:
            return None
        
        session = upload_sessions[session_id]
        session['status'] = 'combining'
        
        try:
            # Creează fișierul final
            final_path = os.path.join(session['chunk_dir'], 'combined_file')
            
            with open(final_path, 'wb') as outfile:
                # Sortează chunk-urile numeric
                chunk_files = sorted([
                    f for f in os.listdir(session['chunk_dir']) 
                    if f.startswith('chunk_')
                ], key=lambda x: int(x.split('_')[1]))
                
                for chunk_file in chunk_files:
                    chunk_path = os.path.join(session['chunk_dir'], chunk_file)
                    with open(chunk_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
                    # Șterge chunk-ul după combinare pentru a economisi spațiu
                    os.remove(chunk_path)
            
            session['combined_path'] = final_path
            session['status'] = 'ready'
            session['progress'] = 100
            
            return final_path
            
        except Exception as e:
            print(f"Eroare la combinarea chunk-urilor: {str(e)}")
            session['status'] = 'error'
            session['error'] = str(e)
            return None

def cleanup_upload_session(session_id):
    """Curăță resursele unei sesiuni de upload"""
    with upload_lock:
        if session_id in upload_sessions:
            session = upload_sessions[session_id]
            try:
                if 'chunk_dir' in session and os.path.exists(session['chunk_dir']):
                    shutil.rmtree(session['chunk_dir'])
            except:
                pass
            
            # Șterge sesiunea după 1 oră
            del upload_sessions[session_id]

def process_large_file(file_path, model_name, language, translation_target,
                      should_adjust_segmentation, process_id, extract_audio_only=False):
    """Procesează un fișier folosind tehnici optimizate pentru feedback granular"""
    print(f"Procesez fișierul: {file_path}")

    try:
        # Încarcă modelul
        model_data = load_model(model_name)
        model = model_data['model']
        device = model_data['device']

        # Verifică dacă este fișier video
        is_video = any(file_path.lower().endswith(ext) for ext in
                      ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mxf', '.wmv', '.flv'])

        is_mp4 = file_path.lower().endswith('.mp4')

        # Verifică dacă există audio
        if is_video:
            check_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a',
                         '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', file_path]
            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
                if 'audio' not in result.stdout.strip().split('\n'):
                    print(f"Atenție: Nu s-a detectat stream audio în {file_path}")
                    raise ValueError("Fișierul nu conține niciun stream audio.")
            except subprocess.CalledProcessError as e:
                print(f"Atenție: ffprobe a eșuat la verificarea audio: {str(e)}")

        print("Folosesc procesare segmentată pentru feedback granular...")

        # Creează un director temporar pentru chunk-urile audio în interiorul directorului procesului
        process_dir = get_process_dir(process_id)
        audio_chunks_dir = os.path.join(process_dir, 'audio_chunks')
        os.makedirs(audio_chunks_dir, exist_ok=True)

        # Extrage audio complet o singură dată ca WAV (pentru acuratețe maximă la seeking/chunking)
        full_audio_path = os.path.join(process_dir, 'full_audio.wav')
        print(f"Extrag audio complet (WAV): {full_audio_path}")

        duration = get_video_duration(file_path)

        try:
            extract_cmd = [
                'ffmpeg',
                '-i', file_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                full_audio_path
            ]

            update_task_status(process_id, 'processing', 10, 'Extragere audio...')
            run_ffmpeg_with_progress(extract_cmd, process_id, "Extragere audio", duration)

            if extract_audio_only:
                # Convertim WAV la MP3 pentru utilizator (mai compact)
                mp3_path = os.path.join(process_dir, 'extracted_audio.mp3')
                update_task_status(process_id, 'processing', 90, 'Finalizare conversie MP3...')

                conv_cmd = [
                    'ffmpeg', '-i', full_audio_path,
                    '-acodec', 'libmp3lame', '-q:a', '2', '-y',
                    mp3_path
                ]
                subprocess.run(conv_cmd, check=True, capture_output=True)

                return {
                    'success': True,
                    'audio_only': True,
                    'audio_filename': 'extracted_audio.mp3',
                    'process_id': process_id,
                    'message': 'Audio extras cu succes'
                }

            # Obține durata folosind ffprobe pe fișierul audio
            probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', full_audio_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())

            # Chunks de 10 minute
            chunk_duration = 600
            total_chunks = math.ceil(duration / chunk_duration)

            print(f"Durata totală: {duration:.1f}s, Chunks: {total_chunks}")

            all_segments = []
            detected_language = language
            language_per_chunk = []  # Stocăm limba pentru fiecare chunk

            # Procesează fiecare chunk din fișierul audio extras
            for chunk_idx in range(total_chunks):
                # Verifică dacă task-ul a fost anulat
                task = get_task_status(process_id)
                if task and task.get('status') == 'cancelled':
                    print(f"Task {process_id} anulat în timpul procesării chunks.")
                    return None

                start_chunk = chunk_idx * chunk_duration
                length_chunk = min(chunk_duration, duration - start_chunk)

                # Evită chunk-uri insignifiante
                if length_chunk < 0.1:
                    continue

                progress_val = 15 + int((chunk_idx / total_chunks) * 60)
                msg = f"Transcriere: chunk {chunk_idx + 1}/{total_chunks}"
                update_task_status(process_id, 'processing', progress_val, msg)

                print(f"Procesez chunk {chunk_idx + 1}/{total_chunks} ({start_chunk:.1f}s - {start_chunk + length_chunk:.1f}s)")

                # Extrage audio chunk ca WAV
                audio_chunk_path = os.path.join(audio_chunks_dir, f'chunk_{chunk_idx:03d}.wav')

                cmd = [
                    'ffmpeg',
                    '-ss', str(start_chunk),    # Seeking înainte de -i (input seeking) pentru acuratețe maximă
                    '-i', full_audio_path,
                    '-t', str(length_chunk),
                    '-acodec', 'pcm_s16le',
                    '-y',
                    audio_chunk_path
                ]

                subprocess.run(cmd, check=True, capture_output=True)

                # Transcrie chunk-ul cu validare
                chunk_result = None
                if os.path.exists(audio_chunk_path) and os.path.getsize(audio_chunk_path) > 100:
                    try:
                        # Verifică durata chunk-ului
                        chunk_dur = get_video_duration(audio_chunk_path)
                        if chunk_dur and chunk_dur > 0.1:
                            # 🔴 MODIFICARE IMPORTANTĂ: NU mai setăm limba la nivel global
                            # Lăsăm Whisper să detecteze limba pentru FIECARE chunk
                            transcribe_kwargs = {
                                'task': 'transcribe',
                                'fp16': (device == "cuda")
                            }
                            # NU setăm language aici - lăsăm detectarea automată pentru fiecare chunk

                            chunk_result = model.transcribe(audio_chunk_path, **transcribe_kwargs)

                            # Înregistrăm limba detectată pentru acest chunk
                            chunk_lang = chunk_result.get('language', 'unknown')
                            language_per_chunk.append({
                                'chunk': chunk_idx + 1,
                                'start_time': start_chunk,
                                'end_time': start_chunk + length_chunk,
                                'language': chunk_lang,
                                'segments_count': len(chunk_result.get('segments', []))
                            })

                            print(f"  ✓ Chunk {chunk_idx + 1}: Limbă detectată = {chunk_lang}, segmente = {len(chunk_result.get('segments', []))}")
                        else:
                            print(f"Chunk {chunk_idx} prea scurt: {chunk_dur}s")
                    except Exception as e:
                        print(f"Eroare la verificarea/transcrierea chunk {chunk_idx}: {str(e)}")

                if not chunk_result:
                    # Dacă chunk-ul e invalid sau Whisper a eșuat, trecem peste el
                    if os.path.exists(audio_chunk_path):
                        os.remove(audio_chunk_path)
                    continue

                chunk_segments = chunk_result.get('segments', [])

                if not chunk_segments:
                    continue

                # Ajustează timpii segmentelor și previne suprapunerea între chunk-uri
                chunk_end_time = start_chunk + length_chunk
                for seg in chunk_segments:
                    actual_start = seg['start'] + start_chunk
                    actual_end = seg['end'] + start_chunk

                    # Ignorăm segmentele care încep după sfârșitul teoretic al acestui chunk
                    if actual_start >= chunk_end_time:
                        continue

                    # Ajustăm timpii și limităm sfârșitul la granița chunk-ului
                    seg['start'] = actual_start
                    seg['end'] = min(actual_end, chunk_end_time)

                    # Adăugăm informația despre limba chunk-ului în fiecare segment
                    seg['detected_language'] = chunk_result.get('language', 'unknown')
                    all_segments.append(seg)

                # Curăță chunk-ul audio
                os.remove(audio_chunk_path)

            # La final, salvăm informațiile despre limbile detectate
            language_report_path = os.path.join(process_dir, 'language_report.json')
            with open(language_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunks': language_per_chunk,
                    'total_chunks': total_chunks,
                    'languages_detected': list(set([item['language'] for item in language_per_chunk]))
                }, f, ensure_ascii=False, indent=2)

            # Procesează segmentele combinate
            segments = sorted(all_segments, key=lambda x: x['start'])

            if should_adjust_segmentation:
                segments = adjust_segmentation_algorithm(segments)

            # Asigurăm o curățenie finală a timpiilor pentru a evita suprapunerile între chunk-uri
            for i in range(len(segments) - 1):
                if segments[i]['end'] > segments[i+1]['start']:
                    # Dacă există suprapunere, tăiem sfârșitul segmentului curent la începutul următorului
                    segments[i]['end'] = segments[i+1]['start']

            # Determinăm limbile predominante pentru raportare
            lang_counter = Counter([item['language'] for item in language_per_chunk])
            primary_language = lang_counter.most_common(1)[0][0] if lang_counter else 'unknown'
            secondary_languages = [lang for lang, count in lang_counter.most_common()[1:3]]

            print(f"📊 Raport limbă pe chunk-uri:")
            for item in language_per_chunk:
                print(f"  Chunk {item['chunk']}: {item['language']} ({item['start_time']:.0f}s - {item['end_time']:.0f}s)")

            return {
                'result': {'text': " ".join([s['text'] for s in segments]),
                           'language': primary_language,
                           'languages_detected': dict(lang_counter),
                           'secondary_languages': secondary_languages},
                'segments': segments,
                'transcribe_time': 0
            }

        except Exception as e:
            print(f"Eroare la procesarea în chunks: {str(e)}")
            # Fallback la procesare normală
            return process_normal_file(file_path, model, device, language,
                                     translation_target, should_adjust_segmentation,
                                     process_id, is_video, is_mp4)
        
    except Exception as e:
        print(f"Eroare la procesarea fișierului: {str(e)}")
        raise

def process_normal_file(file_path, model, device, language, translation_target,
                       should_adjust_segmentation, process_id, is_video, is_mp4, extract_audio_only=False):
    """Procesează un fișier folosind metoda normală"""
    audio_path = file_path
    
    # Încearcă să extragă audio dacă este video
    if is_video:
        print("Încerc să extrag audio din fișier video...")
        try:
            if extract_audio_only:
                audio_path = os.path.join(os.path.dirname(file_path), "extracted_audio.mp3")
                ffmpeg_cmd = [
                    'ffmpeg', '-y', '-i', file_path,
                    '-vn', '-acodec', 'libmp3lame', '-q:a', '2',
                    audio_path
                ]
                duration = get_video_duration(file_path)
                run_ffmpeg_with_progress(ffmpeg_cmd, process_id, "Extragere audio", duration)
            else:
                audio_path = convert_to_wav(file_path, process_id)
        except Exception as e:
            print(f"Eroare la extragerea audio: {e}")
            # Folosește fișierul original
            print("Folosesc fișierul original pentru transcriere...")
    
    # Transcriere
    print(f"Încep transcrierea pe {device}...")
    start_time = time.time()
    
    transcribe_kwargs = {
        'task': 'transcribe',
        'fp16': (device == "cuda")
    }
    
    if language != 'auto':
        transcribe_kwargs['language'] = language
    
    try:
        print(f"Transcriere fișier: {audio_path}")
        # Validare audio înainte de transcriere
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
            raise Exception("Fișier audio invalid sau prea mic")

        audio_dur = get_video_duration(audio_path)
        if not audio_dur or audio_dur < 0.1:
            raise Exception(f"Durată audio invalidă: {audio_dur}")

        result = model.transcribe(audio_path, **transcribe_kwargs)
    except Exception as e:
        print(f"Eroare la transcriere: {str(e)}")
        # Încearcă să transcrie direct fișierul original fără parametri speciali
        try:
            print("Încerc transcriere directă fără parametri speciali...")
            result = model.transcribe(file_path)
        except Exception as e2:
            raise Exception(f"Transcriere eșuată: {str(e2)}")
    
    transcribe_time = time.time() - start_time
    print(f"✓ Transcriere completă în {transcribe_time:.1f} secunde")
    
    # Curăță fișierul audio temporar dacă a fost creat
    if audio_path != file_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except:
            pass
    
    # Procesează segmentele
    segments = result.get('segments', [])
    
    if should_adjust_segmentation:
        settings = {
            'min_duration': 1.0,
            'max_duration': 5.0,
            'max_chars': 80
        }
        
        segments = adjust_segmentation_algorithm(
            segments,
            min_duration=settings['min_duration'],
            max_duration=settings['max_duration'],
            max_chars=settings['max_chars']
        )
    
    return {
        'result': result,
        'segments': segments,
        'transcribe_time': transcribe_time
    }

# ============================================================================
# RUTE FLASK
# ============================================================================

@app.route('/')
def index():
    """Pagina principală cu selecția modelului"""
    # Inițializează sesiunea dacă nu există
    if 'selected_model' not in session:
        session['selected_model'] = DEFAULT_MODEL
    if 'selected_language' not in session:
        session['selected_language'] = 'auto'
    if 'segmentation_settings' not in session:
        session['segmentation_settings'] = {
            'min_duration': 1.0,
            'max_duration': 5.0,
            'max_chars': 80,
            'adjust_segmentation': True
        }
    if 'translation_target' not in session:
        session['translation_target'] = None
    if 'multiple_translations' not in session:
        session['multiple_translations'] = {}
    
    models_info = {name: get_model_info(name) for name in AVAILABLE_MODELS.keys()}
    
    return render_template('index.html', 
                         models=AVAILABLE_MODELS,
                         models_info=models_info,
                         languages=SUPPORTED_LANGUAGES,
                         translation_languages=TRANSLATION_LANGUAGES,
                         selected_model=session['selected_model'],
                         selected_language=session['selected_language'],
                         segmentation_settings=session['segmentation_settings'],
                         translation_target=session['translation_target'],
                         default_model=DEFAULT_MODEL)

# ============================================================================
# RUTE PENTRU UPLOAD SEGMENTAT
# ============================================================================

@app.route('/api/chunk_upload/init', methods=['POST'])
def chunk_upload_init():
    """Initializează o sesiune de upload segmentat"""
    try:
        data = request.get_json()
        file_name = data.get('fileName')
        file_size = int(data.get('fileSize'))
        total_chunks = int(data.get('totalChunks'))
        
        if file_size > app.config['MAX_FILE_SIZE']:
            return jsonify({
                'error': f'Fișierul este prea mare. Maxim {app.config["MAX_FILE_SIZE"] / (1024**3):.1f}GB.'
            }), 400
        
        if not allowed_file(file_name):
            return jsonify({
                'error': 'Format fișier neacceptat.'
            }), 400
        
        # Initializează sesiunea
        session_info = init_upload_session(file_name, file_size, total_chunks)
        
        return jsonify({
            'success': True,
            'sessionId': session_info['id'],
            'chunkSize': app.config['CHUNK_SIZE'],
            'message': 'Sesiune de upload inițializată'
        })
        
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/api/chunk_upload/upload', methods=['POST'])
def chunk_upload():
    """Primește un chunk de date"""
    try:
        chunk_number = int(request.form.get('chunkNumber'))
        total_chunks = int(request.form.get('totalChunks'))
        session_id = request.form.get('sessionId')
        chunk = request.files.get('chunk')
        
        if not chunk:
            return jsonify({'error': 'Nu s-a primit chunk-ul'}), 400
        
        # Salvează chunk-ul
        chunk_data = chunk.read()
        if not save_chunk(session_id, chunk_number, chunk_data):
            return jsonify({'error': 'Eroare la salvarea chunk-ului'}), 500
        
        # Actualizează progresul
        progress = update_upload_progress(session_id, chunk_number)
        
        # Dacă este ultimul chunk, începe combinarea
        if chunk_number == total_chunks - 1:
            combined_path = combine_chunks(session_id)
            if not combined_path:
                return jsonify({'error': 'Eroare la combinarea chunk-urilor'}), 500
        
        return jsonify({
            'success': True,
            'chunkNumber': chunk_number,
            'progress': progress,
            'sessionId': session_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/api/chunk_upload/status/<session_id>', methods=['GET'])
def chunk_upload_status(session_id):
    """Verifică statusul upload-ului"""
    try:
        with upload_lock:
            if session_id not in upload_sessions:
                return jsonify({'error': 'Sesiunea nu există'}), 404
            
            session = upload_sessions[session_id]
            
            return jsonify({
                'success': True,
                'status': session['status'],
                'progress': session['progress'],
                'fileName': session['file_name'],
                'fileSize': session['file_size'],
                'receivedChunks': len(session.get('received_chunks', [])),
                'totalChunks': session['total_chunks']
            })
            
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

def background_processing_task(original_path, model_name, language, translation_target,
                             should_adjust_segmentation, process_id, extract_audio_only, original_filename):
    """Task de procesare care rulează în background"""
    try:
        update_task_status(process_id, 'processing', 5, 'Inițializare procesare...')

        # Procesează fișierul
        process_result = process_large_file(
            original_path, model_name, language, translation_target,
            should_adjust_segmentation, process_id, extract_audio_only
        )

        if process_result is None:
            # Verifică dacă a fost anulat
            task = get_task_status(process_id)
            if task and task.get('status') == 'cancelled':
                print(f"Task {process_id} a fost anulat.")
                return
            raise ValueError("Procesarea a returnat un rezultat nul.")

        if extract_audio_only:
            update_task_status(process_id, 'completed', 100, 'Audio extras cu succes!', process_result)
            return

        result = process_result.get('result', {})
        segments = process_result.get('segments', [])
        detected_language = result.get('language', language)
        secondary_languages = result.get('secondary_languages', [])
        languages_detected = result.get('languages_detected', {})

        # Salvează raportul de limbi
        process_dir = get_process_dir(process_id)
        language_report_path = os.path.join(process_dir, 'language_report.json')
        if os.path.exists(language_report_path):
            # Deja salvat, nu facem nimic
            pass
        else:
            # Salvăm informațiile
            with open(language_report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'primary_language': detected_language,
                    'secondary_languages': secondary_languages,
                    'languages_detected': languages_detected,
                    'total_segments': len(segments)
                }, f, ensure_ascii=False, indent=2)

        # Creează segmentele originale - PĂSTRĂM INFORMAȚIA DESPRE LIMBA FIECĂRUI SEGMENT
        original_segments = []
        for i, segment in enumerate(segments):
            # Extragem limba segmentului (salvată în procesare)
            segment_lang = segment.get('detected_language', detected_language)

            original_segments.append({
                'id': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'start_formatted': format_timestamp(segment['start']),
                'end_formatted': format_timestamp(segment['end']),
                'original': True,
                'language': segment_lang  # 🟢 Adăugăm limba segmentului
            })

        # Salvează segmentele pe disc pentru persistenta
        with open(os.path.join(process_dir, 'original_segments.json'), 'w', encoding='utf-8') as f:
            json.dump({'segments': original_segments}, f, ensure_ascii=False)

        # Traducere - folosește noua funcție multilingvă
        translated_segments = []
        translation_time = 0
        translation_used = None

        if translation_target and translation_target != detected_language:
            update_task_status(process_id, 'processing', 90, f'Traducere în {translation_target}...')
            translation_start = time.time()
            try:
                # Folosește traducerea multilingvă care ține cont de limba fiecărui segment
                translated = translate_multilingual_segments(segments, translation_target, process_id)
                translation_time = time.time() - translation_start

                # Creăm segmentele traduse
                for i, segment in enumerate(translated):
                    translated_segments.append({
                        'id': i + 1,
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'start_formatted': format_timestamp(segment['start']),
                        'end_formatted': format_timestamp(segment['end']),
                        'original': False,
                        'target_language': translation_target,
                        'source_language': segment.get('source_language', detected_language)
                    })

                # Salvăm pe disc
                with open(os.path.join(process_dir, f'translated_segments_{translation_target}.json'), 'w', encoding='utf-8') as f:
                    json.dump({'segments': translated_segments}, f, ensure_ascii=False)

                translation_used = translation_target
                print(f"✓ Traducere multilingvă completă în {translation_time:.1f} secunde")

            except Exception as e:
                print(f"✗ Eroare la traducere: {str(e)}")

        # Creează fișier SRT
        srt_filename = f"transcription_{process_id}.srt"
        srt_path = os.path.join(process_dir, srt_filename)
        write_srt(segments, srt_path)

        # Preview video
        video_preview_url = None
        image_preview_url = None
        is_video = any(original_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mxf', '.m4v', '.webm', '.flv', '.wmv'])

        if is_video:
            update_task_status(process_id, 'processing', 95, 'Pregătire preview...')
            try:
                # Extrage imagine preview (JPG)
                preview_path = extract_video_preview(original_path, process_dir)
                if preview_path:
                    preview_filename = f"preview_{process_id}.jpg"
                    shutil.copy2(preview_path, os.path.join(app.config['UPLOAD_FOLDER'], preview_filename))
                    image_preview_url = f'/preview_image/{preview_filename}'

                # Pregătește video pentru playback (MP4)
                playback_path = convert_to_mp4_for_playback(original_path, process_dir, process_id)
                if playback_path:
                    video_filename = f"video_playback_{process_id}.mp4"
                    shutil.copy2(playback_path, os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
                    video_preview_url = f'/video_file/{video_filename}'
            except Exception as preview_err:
                print(f"Eroare la generarea preview-ului: {str(preview_err)}")

        final_result = {
            'success': True,
            'filename': srt_filename,
            'full_text': result.get('text', ''),
            'language_used': detected_language,
            'translation_used': translation_used,
            'is_translated': bool(translation_used),
            'process_id': process_id,
            'video_preview_url': video_preview_url,
            'image_preview_url': image_preview_url,
            'is_video': is_video,
            'is_mp4': original_path.lower().endswith('.mp4'),
            'original_format': original_path.rsplit('.', 1)[-1].lower() if '.' in original_path else 'unknown',
            'model_used': model_name,
            'processing_time': 'Finalizat',
            'translation_time': f"{translation_time:.1f}s" if translation_time else None
        }

        update_task_status(process_id, 'completed', 100, 'Procesare finalizată!', final_result)

    except Exception as e:
        print(f"Eroare în background_task: {traceback.format_exc()}")
        update_task_status(process_id, 'error', message=str(e))
    finally:
        # Cleanup fișier original combinat
        if os.path.exists(original_path):
            try: os.remove(original_path)
            except: pass

@app.route('/api/chunk_upload/process/<session_id>', methods=['POST'])
def chunk_upload_process(session_id):
    """Inițiază procesarea în background a fișierului încărcat"""
    try:
        with upload_lock:
            session_info = upload_sessions.get(session_id)
            if not session_info or session_info['status'] != 'ready':
                return jsonify({'error': 'Fișierul nu este gata pentru procesare'}), 400
        
        data = request.get_json()
        process_id = str(uuid.uuid4())[:8]
        process_dir = get_process_dir(process_id)
        os.makedirs(process_dir, exist_ok=True)
        
        # Pregătește calea fișierului
        original_filename = secure_filename(session_info['file_name'])
        original_path = os.path.join(process_dir, original_filename)
        shutil.copy2(session_info['combined_path'], original_path)

        # Lansează task-ul în background
        thread = threading.Thread(target=background_processing_task, args=(
            original_path,
            data.get('model', DEFAULT_MODEL),
            data.get('language', 'auto'),
            data.get('translation_target'),
            data.get('adjust_segmentation', True),
            process_id,
            data.get('extract_audio_only', False),
            session_info['file_name']
        ))
        thread.start()

        return jsonify({'success': True, 'process_id': process_id})
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"✗ Eroare la procesare: {error_details}")

        # Curăță fișierele temporare
        if 'process_dir' in locals() and os.path.exists(process_dir):
            try:
                shutil.rmtree(process_dir)
            except:
                pass

        return jsonify({'error': f'Eroare la procesare: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/api/task_status/<process_id>')
def task_status(process_id):
    """Returnează statusul unui task de procesare"""
    status = get_task_status(process_id)
    if not status:
        return jsonify({'error': 'Task-ul nu a fost găsit'}), 404
    return jsonify(status)

@app.route('/api/cancel_task/<process_id>', methods=['POST'])
def cancel_task(process_id):
    """Anulează un task de procesare în curs"""
    try:
        status = get_task_status(process_id)
        if not status:
            return jsonify({'error': 'Task-ul nu a fost găsit'}), 404

        if status['status'] in ['processing', 'queued']:
            update_task_status(process_id, 'cancelled', message='Task anulat de utilizator.')
            return jsonify({'success': True, 'message': 'Task anulat'})
        else:
            return jsonify({'success': False, 'message': f'Task-ul nu poate fi anulat în starea actuală: {status["status"]}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_edits', methods=['POST'])
def save_edits():
    """Salvează modificările făcute în editorul de subtitrări"""
    try:
        data = request.get_json()
        process_id = data.get('process_id')
        segments = data.get('segments')
        is_translated = data.get('is_translated', False)
        target_lang = data.get('target_lang')

        process_dir = get_process_dir(process_id)
        if not process_dir or not os.path.exists(process_dir):
            return jsonify({'error': 'Procesul nu a fost găsit'}), 404

        # Salvează JSON-ul actualizat
        if is_translated and target_lang:
            filename = f"translated_segments_{target_lang}.json"
            srt_filename = f"transcription_{process_id}_{target_lang}.srt"
        else:
            filename = "original_segments.json"
            srt_filename = f"transcription_{process_id}.srt"

        filepath = os.path.join(process_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'segments': segments}, f, ensure_ascii=False, indent=2)

        # Regenerează fișierul SRT
        srt_path = os.path.join(process_dir, srt_filename)
        srt_segments = []
        for seg in segments:
            srt_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text']
            })
        write_srt(srt_segments, srt_path)

        return jsonify({
            'success': True,
            'message': 'Modificările au fost salvate',
            'srt_filename': srt_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_existing_translations')
def get_existing_translations():
    """Obține toate traducerile disponibile pentru procesul curent"""
    try:
        process_id = session.get('process_id')
        if not process_id:
            return jsonify({'success': False, 'message': 'Nicio sesiune activă'})

        process_dir = get_process_dir(process_id)
        if not os.path.exists(process_dir):
            return jsonify({'success': False, 'message': 'Directorul procesului a fost șters'})

        # Caută fișiere translated_segments_*.json
        translations = []
        for file in os.listdir(process_dir):
            if file.startswith('translated_segments_') and file.endswith('.json'):
                lang_code = file.replace('translated_segments_', '').replace('.json', '')

                # Încarcă segmentele pentru a număra
                try:
                    with open(os.path.join(process_dir, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        count = len(data.get('segments', []))
                except:
                    count = 0

                translations.append({
                    'target_language': lang_code,
                    'target_name': TRANSLATION_LANGUAGES.get(lang_code, lang_code),
                    'segment_count': count
                })

        # Obține info despre original
        orig_count = 0
        try:
            with open(os.path.join(process_dir, 'original_segments.json'), 'r', encoding='utf-8') as f:
                data = json.load(f)
                orig_count = len(data.get('segments', []))
        except:
            pass

        return jsonify({
            'success': True,
            'detected_language': session.get('detected_language', 'unknown'),
            'original_segments_count': orig_count,
            'translations': translations,
            'total_translations': len(translations)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chunk_upload/cleanup/<session_id>', methods=['DELETE'])
def chunk_upload_cleanup(session_id):
    """Curăță resursele unei sesiuni de upload"""
    try:
        cleanup_upload_session(session_id)
        return jsonify({'success': True, 'message': 'Sesiune curățată'})
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

# ============================================================================
# RUTE EXISTENTE (menținute pentru compatibilitate)
# ============================================================================

@app.route('/set_model', methods=['POST'])
def set_model():
    """Setează modelul selectat în sesiune"""
    try:
        data = request.get_json()
        model_name = data.get('model', DEFAULT_MODEL)
        
        if model_name in AVAILABLE_MODELS:
            session['selected_model'] = model_name
            
            def load_in_background(name):
                try:
                    load_model(name)
                except Exception as e:
                    print(f"Eroare la încărcarea în background a modelului {name}: {str(e)}")
            
            thread = threading.Thread(target=load_in_background, args=(model_name,))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'model': model_name,
                'message': f'Model setat la: {model_name}'
            })
        else:
            return jsonify({'error': 'Model invalid'}), 400
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/set_language', methods=['POST'])
def set_language():
    """Setează limba selectată în sesiune"""
    try:
        data = request.get_json()
        language = data.get('language', 'auto')
        
        if language in SUPPORTED_LANGUAGES:
            session['selected_language'] = language
            return jsonify({
                'success': True,
                'language': language,
                'message': f'Limba setată la: {SUPPORTED_LANGUAGES[language]}'
            })
        else:
            return jsonify({'error': 'Limbă invalidă'}), 400
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/set_translation_target', methods=['POST'])
def set_translation_target():
    """Setează limba țintă pentru traducere"""
    try:
        data = request.get_json()
        target_language = data.get('target_language', None)
        
        if target_language is None or target_language == '':
            session['translation_target'] = None
            return jsonify({
                'success': True,
                'message': 'Traducere dezactivată'
            })
        elif target_language in TRANSLATION_LANGUAGES:
            session['translation_target'] = target_language
            current_lang = session.get('selected_language', 'auto')
            
            def load_translation_background(src, tgt):
                try:
                    if src != 'auto':
                        load_translation_model(src, tgt)
                except Exception as e:
                    print(f"Eroare la încărcarea modelului de traducere: {str(e)}")
            
            thread = threading.Thread(target=load_translation_background, args=(current_lang, target_language))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                'success': True,
                'target_language': target_language,
                'message': f'Traducere setată la: {TRANSLATION_LANGUAGES[target_language]}'
            })
        else:
            return jsonify({'error': 'Limbă de traducere invalidă'}), 400
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/set_segmentation', methods=['POST'])
def set_segmentation():
    """Setează setările de segmentare"""
    try:
        data = request.get_json()
        
        session['segmentation_settings'] = {
            'min_duration': float(data.get('min_duration', 1.0)),
            'max_duration': float(data.get('max_duration', 5.0)),
            'max_chars': int(data.get('max_chars', 80)),
            'adjust_segmentation': bool(data.get('adjust_segmentation', True))
        }
        
        return jsonify({
            'success': True,
            'settings': session['segmentation_settings'],
            'message': 'Setări de segmentare actualizate'
        })
    except Exception as e:
        return jsonify({'error': f'Eroare: {str(e)}'}), 500

@app.route('/get_models')
def get_models():
    """Returnează lista modelelor disponibile"""
    try:
        selected_model = session.get('selected_model', DEFAULT_MODEL)
        models_list = []
        
        for name, desc in AVAILABLE_MODELS.items():
            info = get_model_info(name)
            models_list.append({
                'id': name,
                'name': name.capitalize(),
                'description': desc,
                'size': info['size'],
                'selected': selected_model == name
            })
        
        return jsonify({'models': models_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_languages')
def get_languages():
    """Returnează lista limbilor disponibile"""
    try:
        selected_language = session.get('selected_language', 'auto')
        languages_list = []
        
        for code, name in SUPPORTED_LANGUAGES.items():
            languages_list.append({
                'code': code,
                'name': name,
                'selected': selected_language == code
            })
        
        return jsonify({'languages': languages_list})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_translation_languages')
def get_translation_languages():
    """Returnează lista limbilor pentru traducere"""
    try:
        selected_target = session.get('translation_target', None)
        languages_list = []
        
        for code, name in TRANSLATION_LANGUAGES.items():
            languages_list.append({
                'code': code,
                'name': name,
                'selected': selected_target == code
            })
        
        return jsonify({
            'translation_languages': languages_list,
            'current_target': selected_target
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_status')
def model_status():
    """Verifică statusul modelelor încărcate"""
    try:
        status = {}
        for model_name in AVAILABLE_MODELS.keys():
            if model_name in loaded_models:
                status[model_name] = {
                    'loaded': True,
                    'device': loaded_models[model_name]['device'],
                    'load_time': f"{loaded_models[model_name]['load_time']:.1f}s"
                }
            else:
                status[model_name] = {'loaded': False}
        
        translation_status = {}
        for model_key in translation_models.keys():
            translation_status[model_key] = {
                'loaded': True,
                'device': translation_models[model_key]['device'],
                'source': translation_models[model_key]['source'],
                'target': translation_models[model_key]['target']
            }
        
        system_info = {
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': os.cpu_count(),
            'total_models_loaded': len(loaded_models),
            'translation_models_loaded': len(translation_models)
        }
        
        return jsonify({
            'status': status, 
            'translation_status': translation_status,
            'system': system_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint pentru upload simplu (compatibilitate)"""
    if 'file' not in request.files:
        return jsonify({'error': 'Niciun fișier selectat'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Niciun fișier selectat'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format fișier neacceptat'}), 400
    
    # Folosește procesarea normală pentru fișiere mici
    model_name = request.form.get('model', session.get('selected_model', DEFAULT_MODEL))
    language = request.form.get('language', session.get('selected_language', 'auto'))
    translation_target = request.form.get('translation_target', session.get('translation_target', None))
    should_adjust_segmentation = request.form.get('adjust_segmentation', 'true').lower() == 'true'
    
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL
    
    filename = secure_filename(file.filename)
    process_id = str(uuid.uuid4())[:8]
    process_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'process_{process_id}')
    os.makedirs(process_dir, exist_ok=True)
    
    original_path = os.path.join(process_dir, filename)
    
    try:
        file.save(original_path)
        
        # Verifică dimensiunea fișierului
        file_size = os.path.getsize(original_path)
        if file_size > 500 * 1024 * 1024:  # >500MB
            return jsonify({
                'error': 'Fișierul este prea mare pentru upload simplu. Folosește upload segmentat.',
                'use_chunked_upload': True,
                'max_simple_size': '500MB'
            }), 400
        
        # Procesare normală
        is_video = any(original_path.lower().endswith(ext) for ext in
                      ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mxf', '.wmv', '.flv'])
        is_mp4 = original_path.lower().endswith('.mp4')
        
        model_data = load_model(model_name)
        model = model_data['model']
        device = model_data['device']
        
        process_result = process_normal_file(
            original_path, model, device, language, translation_target,
            should_adjust_segmentation, process_id, is_video, is_mp4
        )

        result = process_result['result']
        segments = process_result['segments']
        transcribe_time = process_result['transcribe_time']

        detected_language = result.get('language', 'unknown')

        # Creează segmentele originale
        original_segments = []
        for i, segment in enumerate(segments):
            original_segments.append({
                'id': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'start_formatted': format_timestamp(segment['start']),
                'end_formatted': format_timestamp(segment['end']),
                'duration': segment['end'] - segment['start'],
                'char_count': len(segment['text'].strip()),
                'original': True
            })

        # Traducere
        translated_segments = []
        translation_time = 0
        translation_used = None

        if translation_target and translation_target != detected_language:
            print(f"Încep traducerea din {detected_language} în {translation_target}...")
            translation_start = time.time()

            try:
                translated = translate_segments(segments, detected_language, translation_target)
                translation_time = time.time() - translation_start

                for i, segment in enumerate(translated):
                    translated_segments.append({
                        'id': i + 1,
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip(),
                        'start_formatted': format_timestamp(segment['start']),
                        'end_formatted': format_timestamp(segment['end']),
                        'duration': segment['end'] - segment['start'],
                        'char_count': len(segment['text'].strip()),
                        'original': False,
                        'source_language': detected_language,
                        'target_language': translation_target
                    })

                translation_used = translation_target
                print(f"✓ Traducere completă în {translation_time:.1f} secunde")

            except Exception as e:
                print(f"✗ Eroare la traducere: {str(e)}")
                translated_segments = []

        # Determină segmentele finale
        final_segments = translated_segments if translated_segments else original_segments
        is_translated = bool(translated_segments)

        # Salvează în sesiune
        session['original_segments'] = original_segments
        session['detected_language'] = detected_language
        session['process_id'] = process_id

        if is_translated:
            multiple_translations = session.get('multiple_translations', {})
            multiple_translations[translation_target] = translated_segments
            session['multiple_translations'] = multiple_translations

        # Creează fișier SRT
        base_name = os.path.splitext(filename)[0]
        suffix = f"_{translation_used}" if is_translated else f"_{detected_language}"
        srt_filename = f"{base_name}_{model_name}{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        srt_path = os.path.join(process_dir, srt_filename)

        srt_segments = []
        for seg in final_segments:
            srt_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text']
            })

        if not write_srt(srt_segments, srt_path):
            raise Exception("Eroare la generarea fișierului SRT")

        # Calculează statistici
        full_text = result.get('text', '')
        word_count = len(full_text.split())
        total_duration = final_segments[-1]['end'] if final_segments else 0

        # Verifică dacă este video pentru preview
        video_preview_url = None
        image_preview_url = None

        if is_video:
            try:
                # Extrage preview
                video_preview_path = extract_video_preview(original_path, process_dir)
                if video_preview_path and os.path.exists(video_preview_path):
                    preview_filename = f"preview_{process_id}.jpg"
                    preview_dest = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
                    shutil.copy2(video_preview_path, preview_dest)
                    image_preview_url = f'/preview_image/{preview_filename}'

                # Creează video pentru playback dacă nu este MP4
                if not is_mp4:
                    playback_path = convert_to_mp4_for_playback(original_path, process_dir)
                    if playback_path and os.path.exists(playback_path):
                        video_filename = f"video_playback_{process_id}.mp4"
                        video_dest = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                        shutil.copy2(playback_path, video_dest)
                        video_preview_url = f'/video_file/{video_filename}'
                else:
                    video_filename = f"video_original_{process_id}.mp4"
                    video_dest = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                    shutil.copy2(original_path, video_dest)
                    video_preview_url = f'/video_file/{video_filename}'

            except Exception as e:
                print(f"Eroare la extragerea preview: {str(e)}")
        
        return jsonify({
            'success': True,
            'filename': srt_filename,
            'segments': final_segments,
            'original_segments': original_segments,
            'translated_segments': translated_segments if translated_segments else [],
            'full_text': full_text,
            'model_used': model_name,
            'device_used': device,
            'language_used': detected_language,
            'translation_used': translation_used,
            'processing_time': f"{transcribe_time:.1f}s",
            'translation_time': f"{translation_time:.1f}s" if translation_time else None,
            'word_count': word_count,
            'segment_count': len(final_segments),
            'total_duration': total_duration,
            'process_id': process_id,
            'image_preview_url': image_preview_url,
            'video_preview_url': video_preview_url,
            'is_video': is_video,
            'is_mp4': is_mp4,
            'original_format': os.path.splitext(filename)[1][1:] if '.' in filename else 'unknown',
            'is_translated': is_translated,
            'translation_target': translation_target,
            'translation_available': bool(translated_segments),
            'session_stored': True,
            'upload_session_id': None
        })
        
    except Exception as e:
        import traceback
        print(f"Eroare la upload simplu: {traceback.format_exc()}")
        return jsonify({'error': f'Eroare la procesare: {str(e)}'}), 500

@app.route('/download/<process_id>/<filename>')
def download_file(process_id, filename):
    """Descarcă fișierul SRT generat"""
    try:
        process_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'process_{process_id}')
        srt_path = os.path.join(process_dir, secure_filename(filename))
        
        if not os.path.exists(srt_path):
            return jsonify({'error': 'Fișierul nu există'}), 404
        
        return send_file(
            srt_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/plain'
        )
    except Exception as e:
        return jsonify({'error': f'Eroare la descărcare: {str(e)}'}), 500

@app.route('/preview_image/<filename>')
def preview_image(filename):
    """Returnează imaginea de preview"""
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Imaginea nu există'}), 404
        
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_file/<filename>')
def video_file(filename):
    """Returnează fișierul video pentru preview"""
    try:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video-ul nu există'}), 404
        
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/language_report/<process_id>')
def language_report(process_id):
    """Returnează raportul cu limbile detectate pe chunk-uri"""
    try:
        process_dir = get_process_dir(process_id)
        report_path = os.path.join(process_dir, 'language_report.json')

        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
            return jsonify(report)
        else:
            return jsonify({'error': 'Raportul nu a fost găsit'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_video/<process_id>')
def check_video(process_id):
    """Verifică dacă există un fișier video pentru preview"""
    try:
        filename = f"video_playback_{process_id}.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(filepath):
            duration = get_video_duration(filepath)
            return jsonify({
                'success': True,
                'video_url': f'/video_file/{filename}',
                'duration': duration
            })
        return jsonify({'success': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/segments_json/<process_id>/<filename>')
def segments_json(process_id, filename):
    """Returnează fișierul JSON cu segmentele"""
    try:
        process_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'process_{process_id}')
        json_path = os.path.join(process_dir, secure_filename(filename))
        
        if not os.path.exists(json_path):
            return jsonify({'error': 'Fișierul JSON nu există'}), 404
        
        return send_file(
            json_path,
            mimetype='application/json',
            as_attachment=False
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate_segments', methods=['POST'])
def api_translate_segments():
    """Traduce segmentele existente într-o altă limbă"""
    try:
        data = request.get_json()
        segments = data.get('segments', [])
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'ro')
        
        if not segments:
            return jsonify({'error': 'Nu există segmente pentru traducere'}), 400
        
        print(f"Traduc {len(segments)} segmente din {source_lang} în {target_lang}...")
        
        whisper_segments = []
        for seg in segments:
            whisper_segments.append({
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', '')
            })
        
        translated_segments = translate_segments(whisper_segments, source_lang, target_lang)
        
        formatted_segments = []
        for i, segment in enumerate(translated_segments):
            formatted_segments.append({
                'id': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'start_formatted': format_timestamp(segment['start']),
                'end_formatted': format_timestamp(segment['end']),
                'duration': segment['end'] - segment['start'],
                'char_count': len(segment['text']),
                'original': False,
                'source_language': source_lang,
                'target_language': target_lang
            })
        
        return jsonify({
            'success': True,
            'segments': formatted_segments,
            'source_language': source_lang,
            'target_language': target_lang,
            'segment_count': len(formatted_segments),
            'translation_quality': 'high'
        })
        
    except Exception as e:
        import traceback
        print(f"Eroare API traducere: {traceback.format_exc()}")
        return jsonify({'error': f'Eroare la traducere: {str(e)}'}), 500

@app.route('/translate_existing', methods=['POST'])
def translate_existing():
    """Traduce segmentele existente (din sesiune) într-o nouă limbă"""
    try:
        data = request.get_json()
        target_lang = data.get('target_lang')
        
        if not target_lang or target_lang not in TRANSLATION_LANGUAGES:
            return jsonify({'error': 'Limbă țintă invalidă'}), 400
        
        original_segments = session.get('original_segments', [])
        detected_language = session.get('detected_language', 'en')
        
        if not original_segments:
            return jsonify({'error': 'Nu există segmente în sesiune. Încarcă un fișier mai întâi.'}), 400
        
        whisper_segments = []
        for seg in original_segments:
            whisper_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text']
            })
        
        translated_segments = translate_segments(whisper_segments, detected_language, target_lang)
        
        formatted_segments = []
        for i, segment in enumerate(translated_segments):
            formatted_segments.append({
                'id': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'start_formatted': format_timestamp(segment['start']),
                'end_formatted': format_timestamp(segment['end']),
                'duration': segment['end'] - segment['start'],
                'char_count': len(segment['text']),
                'original': False,
                'source_language': detected_language,
                'target_language': target_lang
            })
        
        multiple_translations = session.get('multiple_translations', {})
        multiple_translations[target_lang] = formatted_segments
        session['multiple_translations'] = multiple_translations
        
        process_id = session.get('process_id', str(uuid.uuid4())[:8])
        process_dir = os.path.join(app.config['UPLOAD_FOLDER'], f'process_{process_id}')
        os.makedirs(process_dir, exist_ok=True)
        
        base_name = f"translation_{detected_language}_{target_lang}"
        srt_filename = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
        srt_path = os.path.join(process_dir, srt_filename)
        
        srt_segments = []
        for seg in formatted_segments:
            srt_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text']
            })
        
        if not write_srt(srt_segments, srt_path):
            raise Exception("Eroare la generarea fișierului SRT tradus")
        
        return jsonify({
            'success': True,
            'segments': formatted_segments,
            'source_language': detected_language,
            'target_language': target_lang,
            'segment_count': len(formatted_segments),
            'translation_quality': 'high',
            'srt_filename': srt_filename,
            'process_id': process_id,
            'multiple_translations_count': len(multiple_translations)
        })
        
    except Exception as e:
        import traceback
        print(f"Eroare traducere existentă: {traceback.format_exc()}")
        return jsonify({'error': f'Eroare la traducere: {str(e)}'}), 500


@app.route('/get_translation_capabilities')
def get_translation_capabilities():
    """Returnează capacitățile de traducere disponibile"""
    try:
        current_lang = session.get('selected_language', 'auto')
        
        available_targets = []
        
        for target_code, target_name in TRANSLATION_LANGUAGES.items():
            if target_code != current_lang:
                marian_key = f"{current_lang}-{target_code}"
                reverse_marian_key = f"{target_code}-{current_lang}"
                
                model_type = 'nllb'
                if marian_key in TRANSLATION_MODELS_CONFIG['marian']['models']:
                    model_type = 'marian'
                elif reverse_marian_key in TRANSLATION_MODELS_CONFIG['marian']['models']:
                    model_type = 'marian'
                
                available_targets.append({
                    'code': target_code,
                    'name': target_name,
                    'model_type': model_type,
                    'quality': 'high' if model_type == 'marian' else 'good'
                })
        
        return jsonify({
            'current_language': current_lang,
            'available_targets': available_targets,
            'total_languages': len(TRANSLATION_LANGUAGES),
            'high_quality_models': list(TRANSLATION_MODELS_CONFIG['marian']['models'].keys())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_preview', methods=['POST'])
def video_preview():
    """Extrage și returnează preview video"""
    if 'file' not in request.files:
        return jsonify({'error': 'Niciun fișier selectat'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Niciun fișier selectat'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format fișier neacceptat'}), 400
    
    preview_id = str(uuid.uuid4())[:8]
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'preview_{preview_id}_{secure_filename(file.filename)}')
    
    try:
        file.save(temp_path)
        
        is_video = any(temp_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.mxf', '.wmv', '.flv'])
        
        if not is_video:
            return jsonify({
                'success': True,
                'is_video': False,
                'message': 'Fișier audio - nu este disponibil preview video'
            })
        
        # Obține informații despre video
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                     'stream=width,height,duration,codec_type', 
                     '-of', 'json', temp_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(result.stdout)
        
        video_stream = next((s for s in probe_data.get('streams', []) 
                           if s.get('codec_type') == 'video'), None)
        
        if not video_stream:
            return jsonify({'error': 'Nu s-a găsit stream video'}), 400
        
        duration = float(video_stream.get('duration', 0))
        preview_filename = f'video_preview_{preview_id}.jpg'
        preview_path = os.path.join(app.config['UPLOAD_FOLDER'], preview_filename)
        
        preview_time = duration * 0.25 if duration > 2 else duration / 2
        
        extract_cmd = [
            'ffmpeg',
            '-ss', str(preview_time),
            '-i', temp_path,
            '-vframes', '1',
            '-q:v', '2',
            '-loglevel', 'error',
            '-y',
            preview_path
        ]
        
        subprocess.run(extract_cmd, capture_output=True, check=True)
        
        width = int(video_stream.get('width', 640))
        height = int(video_stream.get('height', 480))
        
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'is_video': True,
            'preview_url': f'/preview_image/{preview_filename}',
            'width': width,
            'height': height,
            'duration': duration,
            'preview_id': preview_id
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        import traceback
        print(f"Eroare preview video: {traceback.format_exc()}")
        return jsonify({'error': f'Eroare la extragerea preview: {str(e)}'}), 500

@app.route('/preview_transcription', methods=['POST'])
def preview_transcription():
    """Previzualizare rapidă a transcrierii"""
    if 'file' not in request.files:
        return jsonify({'error': 'Niciun fișier selectat'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Niciun fișier selectat'}), 400
    
    preview_model_name = 'tiny'
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preview_' + secure_filename(file.filename))
    
    try:
        file.save(temp_path)
        
        model_data = load_model(preview_model_name)
        model = model_data['model']
        
        try:
            result = model.transcribe(
                temp_path, 
                task='transcribe',
                language=None,
                fp16=(model_data['device'] == "cuda")
            )
        except Exception as e:
            print(f"Eroare la transcrierea preview: {str(e)}")
            # Încearcă direct transcriere fără parametri speciali
            result = model.transcribe(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        preview_segments = []
        for i, segment in enumerate(result.get('segments', [])[:5]):
            preview_segments.append({
                'id': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'start_formatted': format_timestamp(segment['start']),
                'end_formatted': format_timestamp(segment['end'])
            })
        
        return jsonify({
            'success': True,
            'preview': preview_segments,
            'has_more': len(result.get('segments', [])) > 5,
            'model_used': preview_model_name,
            'total_segments': len(result.get('segments', [])),
            'detected_language': result.get('language', 'unknown')
        })
        
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return jsonify({'error': f'Eroare la previzualizare: {str(e)}'}), 500

@app.route('/system_info')
def system_info():
    """Returnează informații despre sistem"""
    try:
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'cpu_count': os.cpu_count(),
            'total_memory': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'available_memory': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
            'python_version': os.sys.version.split()[0],
            'torch_version': torch.__version__,
            'whisper_version': whisper.__version__,
            'models_loaded': list(loaded_models.keys()),
            'translation_models_loaded': list(translation_models.keys()),
            'default_model': DEFAULT_MODEL,
            'max_file_size': f"{app.config['MAX_FILE_SIZE'] / (1024**3):.1f} GB",
            'chunk_size': f"{app.config['CHUNK_SIZE'] / (1024**2):.1f} MB",
            'process_timeout': f"{app.config['PROCESS_TIMEOUT']} secunde"
        }
        
        if torch.cuda.is_available():
            try:
                info['gpu_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
                info['cuda_capability'] = torch.cuda.get_device_capability(0)
            except:
                info['gpu_name'] = 'CUDA Device'
                info['gpu_memory'] = 'N/A'
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup')
def cleanup():
    """Curăță modelele încărcate și memoria"""
    try:
        with model_lock:
            loaded_models.clear()
            translation_models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
        return jsonify({
            'success': True,
            'message': 'Memorie curățată',
            'models_loaded': len(loaded_models),
            'translation_models_loaded': len(translation_models)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# FUNCȚIE PENTRU CURĂȚAREA AUTOMATĂ A SESIUNILOR VECHI
# ============================================================================

def cleanup_old_sessions():
    """Curăță sesiunile vechi de upload"""
    while True:
        time.sleep(3600)  # Așteaptă 1 oră
        try:
            with upload_lock:
                current_time = time.time()
                sessions_to_delete = []
                
                for session_id, session in list(upload_sessions.items()):
                    # Șterge sesiunile mai vechi de 24 de ore
                    if current_time - session.get('start_time', 0) > 86400:
                        sessions_to_delete.append(session_id)
                
                for session_id in sessions_to_delete:
                    cleanup_upload_session(session_id)
                    print(f"Curățat sesiunea veche: {session_id}")
                    
        except Exception as e:
            print(f"Eroare la curățarea sesiunilor: {str(e)}")

# Pornire thread pentru curățare automată
cleanup_thread = threading.Thread(target=cleanup_old_sessions)
cleanup_thread.daemon = True
cleanup_thread.start()

# Funcție pentru încărcarea modelului implicit la pornire
def load_default_model_on_startup():
    """Încarcă modelul implicit la pornirea aplicației"""
    try:
        print(f"\n⏳ Se încarcă modelul implicit '{DEFAULT_MODEL}'...")
        start_time = time.time()
        load_model(DEFAULT_MODEL)
        load_time = time.time() - start_time
        print(f"✓ Modelul implicit '{DEFAULT_MODEL}' încărcat în {load_time:.1f} secunde")
    except Exception as e:
        print(f"✗ Eroare la încărcarea modelului implicit: {str(e)}")
        try:
            print("Încerc încărcarea modelului 'tiny' ca fallback...")
            load_model('tiny')
        except:
            print("✗ Nu s-a putut încărca niciun model!")

if __name__ == '__main__':
    # Verifică ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print("✓ FFmpeg este instalat și funcțional!")
        print(f"  Versiune: {result.stdout.split('version')[1].split()[0] if 'version' in result.stdout else 'N/A'}")
    except:
        print("⚠ ATENȚIE: FFmpeg nu este instalat sau nu este în PATH!")
    
    # Verifică CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA este disponibil: {torch.cuda.get_device_name(0)}")
        print(f"  Memorie GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("ℹ CUDA nu este disponibil, se va folosi CPU")
    
    # Informații sistem
    print(f"✓ Upload maxim: {app.config['MAX_FILE_SIZE'] / (1024**3):.1f} GB")
    print(f"✓ Dimensiune chunk: {app.config['CHUNK_SIZE'] / (1024**2):.1f} MB")
    print(f"✓ Timeout procesare: {app.config['PROCESS_TIMEOUT']} secunde")
    
    # Încarcă modelul implicit
    load_default_model_on_startup()
    
    # Pornește aplicația
    print("\n" + "="*70)
    print("🎬 Aplicația de Transcriere Audio/Video cu Upload Segmentat")
    print("="*70)
    print(f"\n📊 Modele disponibile: {', '.join(AVAILABLE_MODELS.keys())}")
    print(f"🌍 Limbi suportate: {len(SUPPORTED_LANGUAGES)} limbi")
    print(f"📁 Fișiere mari: Suport până la {app.config['MAX_FILE_SIZE'] / (1024**3):.1f} GB")
    print(f"🔀 Upload segmentat: Chunks de {app.config['CHUNK_SIZE'] / (1024**2):.1f} MB")
    print(f"🌐 Port: 5000")
    print("\n👉 Accesează http://localhost:5000 în browser")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)