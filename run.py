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
    Carga el modelo V2 (StyleAIClassifierV2) sincronizado con el entrenamiento.
    """
    from transformers import AutoModel, AutoTokenizer
    print(f"Iniciando carga de modelo V2 desde {model_path}...")
    
    # 1. Definir la arquitectura EXACTA de la V2
    class StyleAIClassifierV2(torch.nn.Module):
        def __init__(self, model_name_or_path):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
            self.dropout = torch.nn.Dropout(0.2)
            self.classifier = torch.nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Usamos el token [CLS] (índice 0) igual que en tu entrenamiento
            return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))

    # 2. Instanciar y cargar pesos
    # Usamos model_path porque ahí debe estar el config.json del modelo base
    model = StyleAIClassifierV2(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    pt_path = os.path.join(model_path, 'best_model.pt')
    if os.path.exists(pt_path):
        checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
        # Cargamos el state_dict directamente
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Pesos V2 cargados exitosamente.")
    else:
        print("❌ ERROR: No se encontró best_model.pt")
        
    model.to(device).eval()
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia para PAN 2026")
    parser.add_argument("-i", "--input", type=str, help="Carpeta de entrada (variable $inputDataset en TIRA)")
    parser.add_argument("-o", "--output", type=str, help="Carpeta para guardar predictions.jsonl ($outputDir)")
    parser.add_argument("--model_path", type=str, default="/app/models", help="Ruta al modelo fine-tuneado")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        # Modo por lotes para TIRA
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Usando dispositivo: {device}")

        # Cargar modelo
        model, tokenizer = load_custom_model(args.model_path, device)
        
        # 1.5 Cargar threshold y margin (PAN metrics)
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
        
        if not os.path.exists(args.output):
            os.makedirs(args.output)

        print("\nBuscando archivos .jsonl en entrada...")
        
        # 2. Leer archivos en la carpeta de entrada (puede haber varios)
        input_files = [f for f in os.listdir(args.input) if f.endswith('.jsonl') or f.endswith('.json')]
        if not input_files:
            print("ERROR: No se encontraron archivos json/jsonl en el directorio de entrada.")
            exit(1)

        output_predictions = os.path.join(args.output, 'predictions.jsonl')
        
        total_processed = 0
        
        # Abriendo el archivo de salida para escribir de a poco (ahorra RAM)
        with open(output_predictions, 'w', encoding='utf-8') as out_file:
            
            for file_name in input_files:
                file_path = os.path.join(args.input, file_name)
                print(f"Procesando: {file_name}")
                
                with open(file_path, 'r', encoding='utf-8') as in_file:
                    for line in in_file:
                        if not line.strip():
                            continue
                            
                        item = json.loads(line)
                        text_id = item.get('id', f'item_{total_processed}')
                        raw_text = item.get('text', '')
                        
                        # Preprocesamiento local del dataset
                        clean_text = preprocess_text(raw_text)
                        
                        if not clean_text:
                            # Texto vacio o nulo
                            out_file.write(json.dumps({"id": text_id, "label": 0.5}) + '\n')
                            total_processed += 1
                            continue
                        
                        # Inferencia
                        inputs = tokenizer(
                            clean_text, 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=192,
                            padding=False # No se necesita padding para bs=1
                        )
                        
                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs['attention_mask'].to(device)
                        
                        with torch.inference_mode():
                            logits = model(input_ids, attention_mask)
                            probs = torch.softmax(logits, dim=1)
                            score = probs[0][1].item() # P(AI) continua
                            
                        # Aplicar abstención estratégica si cae en el margen
                        if (threshold - margin) <= score <= (threshold + margin):
                            final_score = 0.5
                        else:
                            final_score = float(score)
                            
                        # Guardar prediccion TIRA
                        out_file.write(json.dumps({"id": text_id, "label": final_score}) + '\n')
                        total_processed += 1
                        
                        if total_processed % 500 == 0:
                            print(f"  ... procesados {total_processed} items.")
                            
        print(f"\n¡Proceso completado exitosamente! Total predicciones: {total_processed}")
        print(f"Guardado en: {output_predictions}")
        
    else:
        print("Modo prueba única. Ejecuta la lógica local con -i y -o.")
        parser.print_help()
