import torch
import logging
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

def compute_embeddings(dataloader, doc_col, device, method='cls'):
    '''
    Compute the embeddings of the documents in a dataloader. 
    The embeddings are computed using model 'bert-base-uncased'.
    By default, the embedding of the [CLS] token is used.
    '''
    embeddings_model = 'bert-base-uncased'
    with torch.no_grad():
        model = AutoModel.from_pretrained(embeddings_model).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(embeddings_model)

        embeddings = torch.empty((len(dataloader.sampler), 768), dtype=torch.float, device=device)        
        start = 0
        for batch in tqdm(dataloader):
            input = tokenizer(batch[doc_col], return_tensors="pt", padding="max_length", truncation=True).to(device)
            output = model(**input)
            if method == 'cls':
                batch_embeddings = output.last_hidden_state[:, 0, :] # use [CLS] embedding
            elif method == 'mean':
                batch_embeddings = output.last_hidden_state.mean(dim=1) # use mean of embeddings
            end = start + len(batch[doc_col])
            embeddings[start:end].copy_(batch_embeddings, non_blocking=True)
            start = end
        del model, tokenizer
    return embeddings
