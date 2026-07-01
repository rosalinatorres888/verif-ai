"""
training/generate_synthetic_es.py
Generates synthetic Spanish misinformation training data using Claude API.

- 4 classes × 5 domains × configurable examples per cell
- Latin American Spanish register, context-rich (40-80 words per claim)
- Author (native Spanish speaker) reviews before merging into train.csv
- All synthetic examples tagged source=synthetic_es for tracking

Usage:
    # Step 1: Generate preview batch (50 examples) for quality review
    python training/generate_synthetic_es.py --preview

    # Step 2: After approving preview, generate full dataset
    python training/generate_synthetic_es.py --generate --per-cell 25

    # Step 3: Merge approved synthetic data into train.csv
    python training/generate_synthetic_es.py --merge

CRITICAL: Do not merge without reviewing the preview output first.
Author must verify Spanish quality and label accuracy before --merge.
"""
import sys
import os
import json
import time
import argparse
import random
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import anthropic

DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "synthetic_es.csv"
TRAIN_PATH  = DATA_DIR / "train.csv"

SEED = 42
random.seed(SEED)

# ─── Domains and classes ──────────────────────────────────────────

DOMAINS = ["salud", "política", "economía mundial", "ciencia", "redes sociales"]

DOMAIN_CONTEXT = {
    "salud": "salud pública, vacunas, medicamentos, enfermedades, tratamientos médicos, pandemia",
    "política": "gobiernos latinoamericanos, elecciones, corrupción, derechos humanos, líderes políticos",
    "economía mundial": "inflación, crisis económica, FMI, comercio internacional, desempleo, criptomonedas",
    "ciencia": "cambio climático, energía renovable, tecnología, inteligencia artificial, medio ambiente",
    "redes sociales": "viral en WhatsApp y Twitter, teorías conspirativas digitales, desinformación en línea",
}

CLASSES = ["true", "false", "misleading", "unverifiable"]

CLASS_INSTRUCTIONS = {
    "true": """Genera una afirmación VERDADERA y verificable.
Debe ser un hecho real, comprobable con fuentes confiables.
La afirmación debe ser específica, con datos o contexto concreto.
No debe ser obvia ni trivial — debe ser el tipo de afirmación que alguien podría dudar.""",

    "false": """Genera una afirmación FALSA — desinformación creíble pero incorrecta.
Debe sonar plausible (no absurda), del tipo que circula en WhatsApp o redes sociales.
Puede contener un elemento real mezclado con información falsa para hacerla más creíble.
No uses nombres de personas reales identificables de forma negativa.""",

    "misleading": """Genera una afirmación ENGAÑOSA — técnicamente parcial o fuera de contexto.
Debe contener un elemento verdadero pero presentado de forma que lleva a conclusiones incorrectas.
Ejemplos: estadísticas reales pero sin contexto, comparaciones injustas, omisión de información clave.
El lector promedio creería que la afirmación implica algo que no es correcto.""",

    "unverifiable": """Genera una afirmación NO VERIFICABLE — que no puede confirmarse ni refutarse con evidencia pública.
Puede ser una especulación sobre intenciones gubernamentales, planes secretos, o estadísticas no publicadas.
Debe sonar como algo que 'alguien dijo' o 'se rumora que' — sin fuente verificable.
No debe ser obviamente falsa ni obviamente verdadera.""",
}

SYSTEM_PROMPT = """Eres un experto en verificación de hechos y análisis de desinformación en América Latina.
Tu tarea es generar ejemplos de entrenamiento para un sistema de detección de desinformación bilingüe.

REGLAS ESTRICTAS:
- Escribe en español latinoamericano natural y fluido (no español de España)
- Cada afirmación debe tener entre 40 y 80 palabras — con suficiente contexto para que el modelo aprenda
- Varía el registro: algunas formales, otras como mensajes de WhatsApp, otras como titulares
- NO uses nombres de personas reales de forma negativa o difamatoria
- NO generes contenido que promueva violencia o discriminación
- Las afirmaciones deben ser plausibles y representativas de lo que circula en redes sociales latinoamericanas
- Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin bloques de código markdown

VARIACIÓN REGIONAL — MUY IMPORTANTE:
- América Latina no es homogénea. Evita palabras con significado radicalmente distinto según el país.
- EVITA estas palabras que son neutras en algunos países pero ofensivas en otros:
  marica, huevón, boludo, coger, concha, verga, pendejo, güey, cabrón, culero
- Usa vocabulario pan-latinoamericano que sea neutro en toda la región:
  amigo/a, chico/a, oye, mira, fíjate, resulta que, te cuento, según dicen, dicen que
- Si usas jerga local, que sea claramente de un contexto específico (ej: 'che' para Argentina)
  y nunca en un contexto que pueda resultar ofensivo fuera de esa región"""


def generate_batch(
    client: anthropic.Anthropic,
    label: str,
    domain: str,
    n: int = 10,
) -> list[dict]:
    """Generate n synthetic examples for a given label and domain."""

    user_prompt = f"""Genera exactamente {n} afirmaciones de entrenamiento con estas características:

ETIQUETA: {label.upper()}
DOMINIO: {domain} ({DOMAIN_CONTEXT[domain]})

INSTRUCCIONES PARA ESTA ETIQUETA:
{CLASS_INSTRUCTIONS[label]}

Responde con este JSON exacto (array de {n} objetos):
[
  {{
    "text": "La afirmación completa aquí, entre 40 y 80 palabras...",
    "label": "{label}",
    "language": "es",
    "domain": "{domain}"
  }}
]

Genera {n} afirmaciones variadas — distintos ángulos, distintos registros, distintas estructuras.
No repitas la misma idea con distintas palabras."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            examples = json.loads(raw)
            # Validate structure
            valid = []
            for ex in examples:
                if all(k in ex for k in ["text", "label", "language"]):
                    ex["source"] = "synthetic_es"
                    ex["label"] = label  # enforce correct label
                    ex["language"] = "es"
                    if len(ex["text"].split()) >= 15:  # minimum length check
                        valid.append(ex)
            return valid
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return []


def run_preview(client: anthropic.Anthropic):
    """Generate a small preview batch for quality review — 2 per class per domain = 40 examples."""
    print("\n" + "="*60)
    print("PREVIEW MODE — generating 2 examples per class × 5 domains")
    print("Review these before running --generate")
    print("="*60)

    preview_rows = []
    for label in CLASSES:
        for domain in DOMAINS:
            print(f"\n  [{label.upper()}] {domain}...")
            batch = generate_batch(client, label, domain, n=2)
            for ex in batch:
                preview_rows.append(ex)
                print(f"    → {ex['text'][:100]}...")
            time.sleep(0.5)

    preview_path = DATA_DIR / "synthetic_es_preview.csv"
    df = pd.DataFrame(preview_rows)
    df.to_csv(preview_path, index=False)

    print(f"\n✅ Preview saved to {preview_path}")
    print(f"   {len(preview_rows)} examples generated")
    print(f"\n{'='*60}")
    print("REVIEW CHECKLIST (native speaker verification):")
    print("  □ Spanish register is natural Latin American (not Castilian)")
    print("  □ Labels are correctly assigned for each example")
    print("  □ Claims are plausible and diverse in structure")
    print("  □ No harmful, defamatory, or discriminatory content")
    print("  □ Length is appropriate (40-80 words each)")
    print(f"{'='*60}")
    print(f"\nIf quality looks good, run:")
    print(f"  python training/generate_synthetic_es.py --generate --per-cell 25")


def run_generate(client: anthropic.Anthropic, per_cell: int = 25):
    """Generate full synthetic dataset: per_cell examples × 4 classes × 5 domains."""
    total_target = per_cell * len(CLASSES) * len(DOMAINS)
    print("\n" + "="*60)
    print(f"FULL GENERATION — {per_cell} per cell × {len(CLASSES)} classes × {len(DOMAINS)} domains")
    print(f"Target: ~{total_target} examples")
    print("="*60)

    all_rows = []
    for label in CLASSES:
        for domain in DOMAINS:
            print(f"\n  [{label.upper()}] {domain} (n={per_cell})...")
            batch = generate_batch(client, label, domain, n=per_cell)
            all_rows.extend(batch)
            print(f"    ✓ {len(batch)} examples generated")
            time.sleep(1)  # rate limit courtesy

    df = pd.DataFrame(all_rows)
    df = df[["text", "label", "language", "source"]].copy()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Full dataset saved to {OUTPUT_PATH}")
    print(f"   Total examples: {len(df)}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nNow review {OUTPUT_PATH} before running --merge")
    print(f"  python training/generate_synthetic_es.py --merge")


def run_merge():
    """Merge synthetic_es.csv into train.csv after author review."""
    if not OUTPUT_PATH.exists():
        print(f"ERROR: {OUTPUT_PATH} not found. Run --generate first.")
        return

    synthetic_df = pd.read_csv(OUTPUT_PATH)
    train_df     = pd.read_csv(TRAIN_PATH)

    print(f"\nCurrent train.csv: {len(train_df)} rows")
    print(f"Synthetic data:    {len(synthetic_df)} rows")
    print(f"\nSynthetic label distribution:")
    print(synthetic_df["label"].value_counts().to_string())
    print(f"\nSynthetic language distribution:")
    print(synthetic_df["language"].value_counts().to_string())

    confirm = input("\nHave you reviewed the synthetic data and approve merging? (yes/no): ")
    if confirm.strip().lower() != "yes":
        print("Merge cancelled. Review synthetic_es.csv first.")
        return

    # Backup current train.csv
    backup_path = DATA_DIR / "train_backup.csv"
    train_df.to_csv(backup_path, index=False)
    print(f"\nBackup saved to {backup_path}")

    # Merge and shuffle
    cols = ["text", "label", "language", "source"]
    merged = pd.concat([
        train_df[cols],
        synthetic_df[cols]
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

    merged.to_csv(TRAIN_PATH, index=False)

    print(f"\n✅ Merged successfully!")
    print(f"   New train.csv: {len(merged)} rows")
    print(f"\nNew label distribution:")
    print(merged["label"].value_counts().to_string())
    print(f"\nNew language distribution:")
    print(merged["language"].value_counts().to_string())
    print(f"\nNext step: rebuild tokenizer and retrain")
    print(f"  python training/build_tokenizer.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview",  action="store_true", help="Generate 40-example preview for review")
    parser.add_argument("--generate", action="store_true", help="Generate full synthetic dataset")
    parser.add_argument("--merge",    action="store_true", help="Merge synthetic data into train.csv")
    parser.add_argument("--per-cell", type=int, default=25,
                        help="Examples per class×domain cell (default 25 = 500 total)")
    args = parser.parse_args()

    if not any([args.preview, args.generate, args.merge]):
        parser.print_help()
        sys.exit(0)

    if args.preview or args.generate:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or "YOUR_KEY" in api_key:
            print("ERROR: ANTHROPIC_API_KEY not set in .env")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)

    if args.preview:
        run_preview(client)
    elif args.generate:
        run_generate(client, per_cell=args.per_cell)

    if args.merge:
        run_merge()
