from data_preprocessing import get_embeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np



def prepare_llm():
    model_name2 = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_name2)
    model = AutoModelForCausalLM.from_pretrained(
        model_name2,
        device_map="auto",   # puts model on GPU if available
        torch_dtype="auto"
    )

    # Create a pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )

    return generator


def prepare_content(query:str, index, df, embedding_model):
    query_embedding = get_embeddings(embedding_model,query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding,k = 3)
    retrieved_content = ""
    tot_cis_score = 0.0
    for rank, (doc_index,distance) in enumerate(zip(I[0],D[0])):
        if doc_index < df.shape[0]:
            document = df.iloc[doc_index]["Document content"]
            retrieved_content = retrieved_content + " " + document
            cosine_score = 1 - distance / 2
            tot_cis_score = tot_cis_score + cosine_score
    
    return retrieved_content, tot_cis_score

def answer_with_content(query:str, index, df,generator,embedding_model):
    retrieved_content, tot_score = prepare_content(query,index, df,embedding_model)
    prompt = f"""
    You are a helpful assistant. Answer the question ONLY using the provided content. 
    If the answer is not in the content, say "The content does not provide this information.

    Content : {retrieved_content}

    Question : {query}

    Answer:

    """
    response = generator(prompt, do_sample = True, temperature = 0.3, top_p = 0.9)[0]["generated_text"]

    if "Answer:" in response:
        return response.split("Answer:")[-1].strip(), tot_score
    else:
        return response.strip(), tot_score



