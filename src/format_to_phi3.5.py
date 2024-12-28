import json

input_file = "trainData.json"
output_file = "trainData_HuggingFace.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

converted = []
for entry in data:
    question = entry.get("instruction", "")
    answer = entry.get("output", "")
    converted.append({"question": question, "answer": answer})

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print("Conversion terminée. Le nouveau fichier se trouve dans", output_file)
