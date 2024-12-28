import json

input_file = "data3.json"
output_file = "cleaned_data3.json"

def clean_entry(entry):
    """
    Nettoie une entrée JSON en supprimant tout ce qui précède et inclut ':' dans les champs 'instruction' et 'output'.
    """
    if "instruction" in entry and ":" in entry["instruction"]:
        entry["instruction"] = entry["instruction"].split(":", 1)[1].strip()
    if "output" in entry and ":" in entry["output"]:
        entry["output"] = entry["output"].split(":", 1)[1].strip()
    return entry

def clean_json_file(input_file, output_file):
    """
    Charge un fichier JSON, nettoie chaque entrée et sauvegarde
    le résultat dans un nouveau fichier JSON.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned_data = []
    for entry in data:
        cleaned_entry = clean_entry(entry)
        
        if cleaned_entry.get("instruction", "").strip():
            cleaned_data.append(cleaned_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
    
    print(f"Fichier nettoyé enregistré sous : {output_file}")

clean_json_file(input_file, output_file)
