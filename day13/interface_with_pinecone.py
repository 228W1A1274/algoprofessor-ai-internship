from dotenv import load_dotenv
import os 
from pinecone import Pinecone

load_dotenv()
key =os.getenv("pinecone_key")

client=Pinecone(api_key=key)

index=client.Index(host="my-first-index-owx83gn.svc.aped-4627-b74a.pinecone.io")

records=[
    {"id":"policy1","text":"Leave policy states that you cant take leave for more than 5 days","category":"leave"},
    {"id":"policy2","text":"Casual leave and Sick leave cannot exceed more than 7 days","category":"leave"},

]

# index.upsert_records(
#     "hr",
#     records
# )
result=index.search(
    namespace="hr",
    query={
        "top_k":1,
        "inputs":{
            "text":"Leave policy"
        }
    }

)

print(result["result"]["hits"])
