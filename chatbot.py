import os
import requests
import numpy as np
import faiss
import torch
import time
from sentence_transformers import SentenceTransformer


# Your OpenRouter API key
OR_API_KEY = open("./OR_key.txt").read().strip()
# Your LLM
model = "openai/gpt-3.5-turbo"
# Retrieve top k most similar documents to the user's query
top_k = 2   
# Keep the most recent conversations between the user and the bot as the bot's memory              
conversation_memory = 5

if torch.backends.mps.is_available():
    device = "mps"  # Metal GPU on Mac
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU 
else:
    device = "cpu"
print(f"Device: {device}")

# A small dataset to illustrate how the bot works
# I have intentionally included two incorrect sentences to show how the bot uses the provided documents through RAG. 
# You can use your specific dataset and prompts based on your need
documents = [
    # Dogs
    "Dogs are known for their loyalty and companionship, making them popular pets worldwide.",
    "A Labrador Retriever is one of the most friendly and intelligent dog breeds.",
    "Dogs can be trained to perform various tasks such as assisting the disabled, hunting, or working as police dogs.",
    "The Dachshund, with its long body and short legs, was originally bred for hunting small animals.",
    "Golden Retrievers are known for their friendly disposition and are commonly used as service dogs.",
    "The French Bulldog is a compact, muscular dog with a playful, affectionate personality, known for its distinct 'bat-like' ears.",
    "The Beagle is a small hound breed known for its keen sense of smell, often used for tracking and detection work.",
    "The Border Collie is considered one of the smartest and most agile dog breeds, often excelling in dog sports and herding tasks.",
    # Incorrect sentence for dogs - Chocolate is toxic to dogs! :)
    "The favorite food of a dog is chocolate, which is safe for them to eat.",
    # Cars
    "Cars are a common mode of transportation that have revolutionized personal mobility.",
    "The electric car market has been growing rapidly due to environmental concerns and advancements in battery technology.",
    "The Ford Mustang is a classic American muscle car known for its powerful engine and sleek design.",
    "Electric cars are powered by electric motors and are considered more environmentally friendly than traditional gasoline-powered cars.",
    "The Tesla Model S is a luxury electric vehicle that has set new standards for performance and technology.",
    "The Chevrolet Corvette is an iconic American sports car, known for its high-performance engine and aggressive styling.",
    "The Toyota Prius is one of the best-selling hybrid cars in the world, known for its fuel efficiency and eco-friendly design.",
    "The Lamborghini Aventador is a high-performance supercar with a powerful V12 engine, known for its speed and exotic design.",
    # Incorrect sentence for cars :)
    "A Ferrari is always slower than a turtle."
]


# 1. Compute embeddings and build FAISS Index

# Building embeddings will take some time on large datasets and when using a CPU
# You can use other embedding models
print("Initializing embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device = device) # 384-dimensional vector space
embedding_path = "document_embeddings.npy"
if not os.path.exists(embedding_path):
    print("Computing embeddings for the corpus...")
    start_time = time.time()
    document_embeddings = embed_model.encode(documents,
                                            batch_size = 32,
                                            show_progress_bar = True,
                                            convert_to_numpy = True,
                                            )
    np.save(embedding_path, document_embeddings)  # Save to disk
    end_time = time.time()
    print("Embeddings saved!")
    print(f"Embedding computing time: {np.round( (end_time - start_time), 2)} seconds")
    
else:
    print("Loading precomputed embeddings...")
    document_embeddings = np.load(embedding_path)

embedding_dim = document_embeddings.shape[1]

# Normalize the embeddings to unit vectors (needed for cosine similarity)
faiss.normalize_L2(document_embeddings)
# Initialize FAISS index with Inner Product (dot product) for cosine similarity
print("Building FAISS index with cosine similarity...")
index = faiss.IndexFlatIP(embedding_dim)
index.add(document_embeddings)
print(f"Indexed {index.ntotal} documents.\n")


# 2. Define Retrieval and Generation functions

def retrieve_documents(query, top_k = top_k):
    """
    Compute the embedding for the query and retrieve the top_k most similar documents.
    """
    query_embedding = embed_model.encode([query], convert_to_numpy = True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]

    return retrieved_docs


def generate_answer(conversation_history):
    """
    Sends the conversation history (which includes the current query with retrieved context)
    to the OpenRouter API and returns the LLM generated answer.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OR_API_KEY}"
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model,
        "messages": conversation_history,
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers= headers, json= payload)
        
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return "Sorry, I couldn't generate an answer at this time."

        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        return answer

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return "Sorry, I couldn't connect to the API at this time."
    except (KeyError, ValueError) as e:
        # if the response JSON structure is not as expected
        print(f"Error parsing response: {e}")
        return "Sorry, there was an error parsing the response."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "Sorry, an unexpected error occurred."

# 3. Chatbot with Conversation Memory

def chatbot():
    # Initialize conversation history with a system prompt.
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers the user's questions using **provided context** and previous **conversation history** to generate accurate and relevant answers."
                " **Pretend you do not have access to the provided context** and treat it as your own knowledge."
            )
        }
    ]
    print(f'{"*"*100}\n\n')
    print("Welcome to your assistant Chatbot! Type 'exit' to quit.")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Retrieve relevant documents from the knowledge base.
        retrieved_docs = retrieve_documents(user_query)
        # Combine the retrieved documents into one context string.
        context = "\n\n".join(retrieved_docs)
        current_query = f"Context:\n{context}\n\nQuestion: {user_query}"
        
        # Append the current user query (with context) to the conversation history
        conversation_history.append({
            "role": "user",
            "content": current_query
        })
        
        # Generate an answer using the full conversation history.
        answer = generate_answer(conversation_history)

        # Append the bot's asnwer to the conversation history
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        # Keep only the recent exchanges as the bot's memory to avoid having too large prompts for LLM
        # The first item is the system prompt. We keep it as it is.
        # Multiplying by 2 because both the user's query and the assistant's response are included in the conversation history. A pair of query and response is counted as a single entry in memory.
        if len(conversation_history[1: ]) > conversation_memory * 2:
            conversation_history = conversation_history[ :1] + conversation_history[-(conversation_memory * 2): ]

        print("\nBot:", answer)


if __name__ == "__main__":
    chatbot()
