import json

# Chemin du fichier source
input_file = "trainData.json"
# Chemin du fichier de sortie
output_file = "trainData_HuggingFace.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# On s'attend à ce que 'data' soit une liste de dicts avec "instruction", "input", "output"
converted = []
for entry in data:
    question = entry.get("instruction", "")
    answer = entry.get("output", "")
    # Ignorer le champ "input" car ici il est vide, mais si nécessaire, on pourrait l'incorporer
    converted.append({"question": question, "answer": answer})

# Sauvegarde du nouveau format
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print("Conversion terminée. Le nouveau fichier se trouve dans", output_file)
