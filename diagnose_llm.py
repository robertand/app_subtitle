import torch
import time
import sys
import os

# Simulăm structura din app.py pentru a găsi modelul
TRANSLATION_MODELS_CONFIG = {
    'gemma': {'name': 'google/translategemma-12b-it'},
    'mistral': {'name': 'mistralai/Mistral-7B-Instruct-v0.3'}
}

def diagnose(engine='gemma'):
    print(f"\n=== Diagnostic Accelerare LLM ({engine}) ===")

    # 1. Verificare CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Disponibil: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Totală: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ ATENȚIE: CUDA nu este detectat. LLM-ul va rula pe CPU (foarte încet).")

    # 2. Verificare Biblioteci Optimizare
    try:
        import bitsandbytes
        print("bitsandbytes (4bit): Instalat ✅")
    except ImportError:
        print("bitsandbytes (4bit): Lipsă ❌ (Recomandat: pip install bitsandbytes)")

    try:
        import accelerate
        print("accelerate (device_map): Instalat ✅")
    except ImportError:
        print("accelerate (device_map): Lipsă ❌ (Recomandat: pip install accelerate)")

    # 3. Test Încărcare Model (doar dacă userul dorește, fiindcă consumă net/timp)
    print("\nÎncerc să încarc modelul pentru test de viteză...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = TRANSLATION_MODELS_CONFIG[engine]['name']
    device = "cuda" if cuda_available else "cpu"

    try:
        start_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {
            "torch_dtype": torch.float16 if cuda_available else torch.float32,
            "device_map": "auto" if cuda_available else None,
        }

        if cuda_available:
            try:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            except:
                pass

        print(f"Se descarcă/încarcă {model_name}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            if "quantization_config" in model_kwargs:
                print(f"⚠️ Încărcarea 4bit a eșuat ({e}). Reîncerc FP16...")
                del model_kwargs["quantization_config"]
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                raise e

        if not cuda_available:
            model = model.to(device)

        print(f"✓ Model încărcat în {time.time() - start_load:.1f}s")
        print(f"Modelul rulează pe: {model.device}")

        # 4. Benchmark Infernță
        print("\nRulez benchmark inferență...")
        prompt = "Translate to Romanian: The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_inf = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50)

        duration = time.time() - start_inf
        tokens = len(output[0])
        tps = tokens / duration

        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Rezultat: {result}")
        print(f"Viteză: {tps:.2f} tokens/secundă")
        print(f"Timp total: {duration:.2f}s")

        if tps < 5 and cuda_available:
            print("\n💡 Sugestie: Viteza pare mică pentru GPU. Verifică dacă nu cumva modelul a făcut spill în RAM/Swap.")
        elif not cuda_available:
            print("\n💡 Sugestie: Instalează driverele NVIDIA și 'bitsandbytes' pentru a muta procesarea pe placa video.")

    except Exception as e:
        print(f"\n❌ Eroare în timpul diagnosticului: {str(e)}")

if __name__ == "__main__":
    eng = sys.argv[1] if len(sys.argv) > 1 else 'gemma'
    diagnose(eng)
