from main import DynamicAgentRAG

rag = DynamicAgentRAG()

with open("resources/Engine.txt", "r") as file:
    rag.add_to_memory(file.read())


while(True):
    txt_in = input("Write a search query: > ")
    result = rag.query_memory(txt_in)
    print(rag.format_context(result["context"]))
