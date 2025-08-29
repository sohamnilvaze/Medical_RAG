import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter
import nltk
nltk.download('punkt_tab')
from data_preprocessing import prepare_model

embedding_model = prepare_model()


def semantic_chunking(text, similarity_threshold):
    """
    Perform semantic chunking on a given text by grouping semantically similar sentences.

    Parameters:
    - text (str): The input text to chunk.
    - model (SentenceTransformer): Preloaded SentenceTransformer model for embeddings.
    - similarity_threshold (float): Threshold to determine if sentences belong in the same chunk.

    Returns:
    - List[str]: A list of chunks, each containing semantically similar sentences.
    """
    # Split the text into sentences
    print("Splitting the sentences")
    # sentence_pattern = r"(.*?[.!?])"
    sentence_pattern = r'([^.!?]*[.!?])'
    sentences = re.findall(sentence_pattern, text)

    # Generate embeddings for sentences
    print("Creating the embeddings")
    embeddings = embedding_model.encode(sentences)
    embeddings_list = embeddings.tolist()

    semantic_chunks = []

    for i in range(len(sentences)):
        if i == 0:
            semantic_chunks.append(sentences[i])
        else:
            # Reshape embeddings to 2D arrays for cosine similarity
            embedding1 = np.array(embeddings_list[i - 1]).reshape(1, -1)
            embedding2 = np.array(embeddings_list[i]).reshape(1, -1)
            similarity = cosine_similarity(embedding1, embedding2)

            if similarity[0][0] >= similarity_threshold:
                # Combine the current sentence with the previous chunk
                semantic_chunks[-1] += " " + sentences[i]
            else:
                # Start a new chunk
                semantic_chunks.append(sentences[i])

    return semantic_chunks

headers = [
    "Patient ID",
    "Admission Date",
    "Discharge Date",
    "Date of Birth",
    "Sex",
    "Service",
    "Chief Complaint",
    "History of Present Illness",
    "Past Medical History",
    "Past Surgical History",
    "Social History",
    "Family History",
    "Allergies",
    "Medications on Admission",
    "Hospital Course",
    "Investigations",
    "Procedures",
    "Discharge Plan"
]


def extract_sections_chunks(text):
    # Sort headers by length (descending) to avoid partial matches
    chunks=[]
    headers = sorted(headers, key=len, reverse=True)

    # Create a regex pattern to detect headers
    pattern = r"(?:" + "|".join(re.escape(header) for header in headers) + r")"

    # Use regex to find all occurrences of headers
    matches = list(re.finditer(pattern, text))

    section_dict = {}

    for i, match in enumerate(matches):
        header = match.group().strip()
        start_idx = match.end()  # Start of the section content

        # Determine end index (either next header or end of text)
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start_idx:end_idx].strip()
        section_dict[header] = content
        chunks.append(content)
    return chunks

def recursive_chunking(text):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100
  )
  docs = text_splitter.create_documents([text])
  required_notes_chunks4=[]
  for doc in docs:
    required_notes_chunks4.append(doc.page_content)

  return required_notes_chunks4

def chunk_string_with_overlap(string):
    chunks = []
    chunk_size = len(string) // 10
    overlap_size = chunk_size // 10
    i = 0
    while i < len(string) - chunk_size + 1:
        chunk = string[i:i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap_size
    # Add the last chunk if there's remaining string
    if i < len(string):
        chunks.append(string[i:])
    return chunks


def context_aware_chunking_spacy(text):
  text_splitter =SpacyTextSplitter(chunk_size=400)
  docs = text_splitter.split_text(text)
  doc_final=[]
  for i, doc in enumerate(docs):
    doc_final.append(doc)

  return doc_final

def context_aware_chunking_nltk(text):
  text_splitter = NLTKTextSplitter(chunk_size=400)
  docs = text_splitter.split_text(text)
  doc_final=[]
  for i, doc in enumerate(docs):
    doc_final.append(doc)

  return doc_final



