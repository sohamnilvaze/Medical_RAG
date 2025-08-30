from data_preprocessing import get_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import re
import cohere
from dotenv import load_dotenv
import os

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

def extract_id_query(query:str):
  match = re.search(r"(?:patient\s*id|id)\s*[:=]?\s*(\d+)", query, re.IGNORECASE)
  if match:
    return match.group(1)

def prepare_content(query:str, index, df,embedding_model,k = 3):
    query_id = extract_id_query(query)
    query_embedding = get_embeddings(embedding_model,query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding,k = k)
    retrieved_content = ""
    tot_cis_score = 0.0
    retrieved_count = 0
    for doc_index, distance in zip(I[0], D[0]):
        if doc_index < 0 or doc_index>=len(df):
            print(f"Invalid index: {doc_index}, dataframe length = {len(df)}")
            continue

        patient_id = df.iloc[doc_index]["Patient ID"]
        if str(patient_id) != str(query_id):
          continue

        document = df.iloc[doc_index]["Document content"]
        retrieved_content = retrieved_content + document
        cosine_score = 1 - distance / 2
        tot_cis_score = tot_cis_score + cosine_score
        retrieved_count = retrieved_count + 1
        if retrieved_count>=k:
          break

    if retrieved_content.strip() == "":
        retrieved_content = "No notes found for this patient ID."
    
    return retrieved_content, tot_cis_score

def answer_with_content(query:str, index, df,embedding_model):

    co = cohere.ClientV2(cohere_api_key)
    retrieved_content, tot_score = prepare_content(query,index, df,embedding_model)
    additional_context = """
      You are an expert medical assistant tasked with answering questions based on the provided clinical context.
      The context contains important sections or headers. Please ensure that you pay close attention to these headers
      and use the most relevant section to answer the query. Your response should be clear, concise, medically accurate,
      and tailored to the specific medical scenario. Always provide a full-sentence answer to the query based on the
      information from the most appropriate section of the context.

      Context: {context}

      Question: {query}

      Answer:
      """
    combined_prompt = additional_context.format(context=retrieved_content, query=query)

    response = co.chat(
        model = 'command-xlarge-nightly',
         messages=[
        {
            "role": "user",
            "content": combined_prompt,
        }
    ]
    )
    print(response.message.content[0].text)
    print(type(response.message.content))
    print(len(response.message.content))
    response1 = response.message.content[0].text

    return response1,tot_score



