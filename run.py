"""
run.py
======
PAN 2026 - Voight-Kampff AI Detection
Script de inferencia compatible con la plataforma TIRA.

Uso:
    python run.py -i $inputDataset -o $outputDir [--model_path ./models]
"""

import os
import json
import argparse
import warnings
import torch
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
from data_loader import preprocess_text
from train import StyleAIClassifier

def load_custom_model(model_path: str, device: torch.device):
    """
    Carga el modelo V2 (StyleAIClassifierV2) sincronizado con el entrenamiento,
    forzando la precisión Float32 para evitar errores de mismatch de tipos.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os

    print(f"Iniciando carga de configuración y arquitectura desde {model_path}...")
    
    # 1. Cargamos la configuración base (esto no lee los archivos de pesos pesados)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    
    # 2. Definición de la arquitectura exacta V2
    class StyleAIClassifierV2(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # Cargamos la estructura vacía basada en la configuración
            # Esto evita errores si el archivo .safetensors original no está o es corrupto
            self.encoder = AutoModel.from_config(cfg)
            self.dropout = torch.nn.Dropout(0.2)
            self.classifier = torch.nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Usamos el token [CLS] (índice 0) igual que en tu entrenamiento en la Universidad de Guayaquil
            return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))

    # 3. Instanciar el modelo vacío y el tokenizador
    model = StyleAIClassifierV2(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 4. Carga manual de tus pesos entrenados (best_model.pt)
    pt_path = os.path.join(model_path, 'best_model.pt')
    if os.path.exists(pt_path):
        print(f"Inyectando pesos desde {pt_path}...")
        # Cargamos el state_dict (weights_only=False por compatibilidad con archivos de Colab/Kaggle)
        checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # --- CAMBIO CRÍTICO PARA EL ERROR DE DTYPE ---
        # Convertimos todo el modelo a Float32. Esto asegura que tanto el encoder 
        # como la capa classifier usen el mismo tipo de dato, eliminando el error de BFloat16.
        model = model.float() 
        # ---------------------------------------------
        
        print("✅ Modelo V2 cargado y convertido a Float32 exitosamente.")
    else:
        raise FileNotFoundError(f"❌ ERROR: No se encontró el archivo de pesos en {pt_path}")
        
    model.to(device).eval()
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia para PAN 2026 - Universidad de Guayaquil")
    
    # Argumentos posicionales requeridos por la plataforma TIRA
    parser.add_argument("input_file", type=str, help="Ruta completa al archivo dataset.jsonl")
    parser.add_argument("output_dir", type=str, help="Carpeta donde se guardará predictions.jsonl")
    parser.add_argument("--model_path", type=str, default="/app/models", help="Ruta al directorio del modelo")
    
    args = parser.parse_args()
    
    # Configuración de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Usando dispositivo: {device}")

    # 1. Cargar modelo con corrección Float32 integrada
    model, tokenizer = load_custom_model(args.model_path, device)
    
    # 2. Cargar parámetros de abstención (Threshold y Margin)
    threshold = 0.5
    margin = 0.0
    thr_path = os.path.join(args.model_path, 'threshold_config.json')
    
    if os.path.exists(thr_path):
        try:
            with open(thr_path, 'r') as f:
                cfg = json.load(f)
                threshold = cfg.get('best_threshold', 0.5)
                margin = cfg.get('best_margin', 0.0)
            print(f"🎯 Configuración de Abstención: Threshold={threshold:.3f}, Margin={margin:.3f}")
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo leer {thr_path}: {e}")
    
    # 3. Preparar directorio de salida
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_predictions = os.path.join(args.output_dir, 'predictions.jsonl')
    total_processed = 0

    print(f"📄 Procesando archivo: {args.input_file}")
    
    # 4. Bucle de inferencia optimizado para lectura directa de archivo
    with open(output_predictions, 'w', encoding='utf-8') as out_file:
        if not os.path.exists(args.input_file):
            print(f"❌ ERROR: El archivo de entrada {args.input_file} no existe.")
            exit(1)

        with open(args.input_file, 'r', encoding='utf-8') as in_file:
            for line in in_file:
                if not line.strip():
                    continue
                    
                item = json.loads(line)
                text_id = item.get('id', f'item_{total_processed}')
                raw_text = item.get('text', '')
                
                # Preprocesamiento (Asegúrate de que preprocess_text esté definido arriba)
                clean_text = preprocess_text(raw_text)
                
                if not clean_text:
                    # Caso de texto vacío: asignamos neutralidad (0.5)
                    out_file.write(json.dumps({"id": text_id, "label": 0.5}) + '\n')
                    total_processed += 1
                    continue
                
                # Tokenización e inferencia
                inputs = tokenizer(
                    clean_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512, # Restaurado a 512 para máxima precisión local
                    padding=False
                )
                
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                with torch.inference_mode():
                    logits = model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)
                    score = probs[0][1].item()
                    
                # Aplicar lógica de abstención basada en margen
                if (threshold - margin) <= score <= (threshold + margin):
                    final_score = 0.5
                else:
                    final_score = float(score)
                    
                # Escritura inmediata al archivo (ahorra memoria RAM)
                out_file.write(json.dumps({"id": text_id, "label": final_score}) + '\n')
                total_processed += 1
                
                if total_processed % 500 == 0:
                    print(f"  ... procesados {total_processed} items.")
                            
    print(f"\n✅ ¡Proceso completado! Total predicciones: {total_processed}")
    print(f"📁 Resultado guardado en: {output_predictions}")
