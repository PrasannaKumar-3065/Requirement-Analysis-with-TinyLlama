from advanced_rag import advanced_rag

while(True):
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    answer, sources = advanced_rag(query)
    print("\nAnswer:", answer)
    print("Sources:", "; ".join(sources))