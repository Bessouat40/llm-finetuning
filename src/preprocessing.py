import json
import pdfplumber
import os
from dotenv import load_dotenv
import re

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
train_folder = os.getenv("TRAIN_FOLDER")
output_file = "fine_tuning_data.json"

if not os.path.exists(output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

def extract_text_from_pdfs(pdf_paths):
    extracted_data = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_data.append(text)
    return extracted_data

pdf_paths = [os.path.join(train_folder, "3doc.pdf")]

texts = extract_text_from_pdfs(pdf_paths)
if texts:
    print("Exemple de texte brut extrait :\n", texts[0])

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

cleaned_texts = [clean_text(text) for text in texts]

loader = PyPDFLoader(pdf_paths[0])
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50,
)
chunks = text_splitter.split_documents(documents)

print("Exemples de chunks :")
for chunk in chunks[:5]:
    print(chunk.page_content)

llm = Ollama(model="llama3")

prompt = PromptTemplate(
    input_variables=["texte"],
    template="Tu es la personne qui a écrit le texte qui suit. Génère en français une question pertinente et sa réponse basée sur le texte suivant : {texte}"
)

chain = LLMChain(llm=llm, prompt=prompt)

print("\nGénération des questions-réponses :")
for chunk in chunks:
    try:
        text = chunk.page_content
        result = chain.run({"texte": text})
        question, answer = result.split("\n", 1)
        
        entry = {
            "instruction": question.strip(),
            "input": "",
            "output": answer.strip()
        }
        
        with open(output_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"Ajouté : {entry}")
    
    except Exception as e:
        print(f"Erreur lors du traitement du chunk : {e}")
