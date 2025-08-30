import gradio as gr
from data_preprocessing import read_all_txt_files, prepare_vector_database, prepare_model
from query_handling import answer_with_content, prepare_content, prepare_llm

folder_path = "E:\COEP ACADEMICS\Nitin Sir Colab Work\Demo RAG\Notes"

print("Reading all text files")
all_contents, ids = read_all_txt_files(folder_path, 2)

print("Initiating the models")
embedding_model = prepare_model()

print("Preparing Faiss vector database")
faiss_index, content_df, metadata = prepare_vector_database(embedding_model,all_contents,ids)

def process_query(query:str):
    if not query.strip():
        return "Please enter a query"

    required_content, score = prepare_content(query,faiss_index,content_df,embedding_model)
    print(required_content)
    print(score)
    answer, score = answer_with_content(query,faiss_index,content_df,embedding_model)

    return answer, f"Confidence Score: {score}"


with gr.Blocks() as demo:
    gr.Markdown("Medical RAG system")
    gr.markdown("Ansk query and get answer based on the notes provided.")
    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", placeholder="Type your question here...")
    
    submit_btn = gr.Button("Get Answer")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", lines=5)
    
    with gr.Row():
        score_output = gr.Number(label="Score", lines=1)
    
    submit_btn.click(fn=process_query, inputs=query_input, outputs=[answer_output, score_output])


if __name__ == "__main__":
    demo.launch(debug = True)
