import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

file1 = 'cleaned_data1.json'
file2 = 'cleaned_data2.json'
file3 = 'cleaned_data3.json'

data1 = load_json(file1)
data2 = load_json(file2)
data3 = load_json(file3)

merged_data = data1 + data2 + data3

output_file = 'trainData.json'
save_json(merged_data, output_file)

print(f"Fichier fusionné sauvegardé sous le nom {output_file}")
