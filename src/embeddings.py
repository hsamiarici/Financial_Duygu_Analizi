import numpy as np
import pickle
import requests
import os
from tqdm import tqdm

def download_fasttext_embeddings():
    """Download Turkish FastText embeddings if not exists"""
    url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.tr.300.vec.gz"
    save_path = "assets/fasttext/cc.tr.300.vec.gz"
    
    if not os.path.exists(save_path):
        print("Downloading Turkish FastText embeddings...")
        os.makedirs("assets/fasttext", exist_ok=True)
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    return save_path

def load_fasttext_embeddings(tokenizer, embedding_dim=300, max_words=20000):
    """
    Create embedding matrix from FastText
    """
    embedding_path = download_fasttext_embeddings()
    
    # Load FastText vectors (first 50k words for speed)
    print("Loading FastText embeddings...")
    embeddings_index = {}
    
    with open(embedding_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)  # Skip first line (header)
        for i, line in enumerate(tqdm(f, desc="Processing embeddings")):
            if i >= 50000:  # Limit to 50k words
                break
            values = line.rstrip().split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Prepare embedding matrix
    word_index = tokenizer.word_index
    num_words = min(len(word_index) + 1, max_words)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    
    for word, i in word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            # If word not in FastText, use random initialization
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Coverage: {np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1)) / num_words:.2%}")
    
    return embedding_matrix