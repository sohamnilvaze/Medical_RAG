import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from data_chunking import semantic_chunking, extract_sections_chunks,chunk_string_with_overlap,recursive_chunking,context_aware_chunking_nltk, context_aware_chunking_spacy



def prepare_model():
    model_name='sentence-transformers/all-MiniLM-L6-v2'
    embedding_model = SentenceTransformer(model_name)

    return embedding_model



def read_all_txt_files(folder_path:str,chunk_choice:int) -> list[str]:
    contents = []
     
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks = []
                    if chunk_choice == 0:
                        chunks = chunk_string_with_overlap(content)
                    elif chunk_choice == 1:
                        chunks = semantic_chunking(content,0.65)
                    elif chunk_choice == 2:
                        chunks = extract_sections_chunks(content)
                    elif chunk_choice == 3:
                        chunks = recursive_chunking(content)
                    elif chunk_choice == 4:
                        chunks = context_aware_chunking_spacy(content)
                    else:
                        chunks = context_aware_chunking_nltk(content)
                
                for chunk in chunks:
                    contents.append(chunk)

    else:
        print(f"Invalid folder path: {folder_path}")
    
    return contents

def get_embeddings(embedding_model,text):
    embeddings = embedding_model.encode(text,convert_to_numpy=True)
    return embeddings.tolist()

def prepare_vector_database(embedding_model,contents : list[str]):
    df = pd.DataFrame(contents,columns = ["Document content"])
    df["Embeddings"] = df['Document content'].apply(lambda x:get_embeddings(embedding_model,text = x))

    index = faiss.IndexFlatL2(384)
    for i in range(df.shape[0]):
        emb = np.array(df['Embeddings'][i]).astype('float32') 
        emb = emb.reshape(1,-1)
        index.add(emb)
    
    return index, df
    




