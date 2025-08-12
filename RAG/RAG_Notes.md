### RAG

Two main components of RAG:
1) Indexing: A pipeline for ingesting data from a source and indexing it. This usually happens offline.
2) Retrieval and generation: The actual RAG chain, which takes the user query to run time and retrieves the relevant data from the index, then passes that to the model.


Indexing:
1) Load: First we need to load our data. This is done through document loaders.
2) Split: Text splitters break large Documents into smaller chunks. This is useful both for indexing data and passing it into a model, as large chunks are harder to search over and won't fit in a model's finite context window.
3) Store: We need somewhere to store and index our splits, so that they can be searched over later. This is often done using a vectorStore and Embeddings model.


Retrieval and Generation:
4) Retrieve: Given a user input, relevant splits are retrieved from storage using a Retriever.
5) Generate: A chatModel/LLM produces an answer using a prompt that includes both the question with the retrieved data.


| Vector Store | Type | Persistence | Scale | Use Case |
|--------------|------|-------------|-------|----------|
| InMemoryVectorStore | Local | ❌ No | Small | Development/Testing |
| FAISS | Local | ✅ Files | Medium | Local production |
| Chroma | Local | ✅ SQLite | Medium | Development |
| Pinecone | Cloud | ✅ Managed | Large | Production |
| Weaviate | Cloud/Self | ✅ Managed | Large | Production |
| PGVector | Database | ✅ SQL | Large | Existing PostgreSQL |
| Redis | In-Memory | ✅ Optional | Medium | Fast access |