# This is a version of the main.py file found in ../../../server/main.py without authentication.
# Copy and paste this into the main file at ../../../server/main.py if you choose to use no authentication for your retrieval plugin.
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Body, UploadFile
from fastapi.staticfiles import StaticFiles

from models.api import *
from datastore.factory import get_datastore
from services.file import get_document_from_file

from models.models import DocumentMetadata, Source

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

description = """
This API is a proof of concept for ZURU's Code Compliance AI project. 🚀

The API uses Pinecone to store and retrieve documents, and GPT-4 to answer questions.

## How to upload documents

This endpoint allows uploading a single file (PDF, TXT, DOCX, PPTX, or MD) and storing its text and metadata in the vector database. The file is converted to plain text and split into chunks of around 200 tokens, each with a unique ID. The endpoint returns a list containing the generated id of the inserted file.

1. Open the `upsert-file` tab
2. Click on the `Try it out` button
3. Click on the `Choose File` button and select a file
4. Click on the `Execute` button

## How to retrieve (or query) documents

This endpoint allows querying the vector database using one or more natural language queries and optional metadata filters. The endpoint returns a list of objects that each contain a list of the most relevant document chunks for the given query, along with their text, metadata and similarity scores.

1. Open the `query` tab
2. Click on the `Try it out` button
3. Enter a query in the `query` field (replacing the _string_, leave the quotes)
4. Filter field is optional, if you don't use it, remove the `filter` field from the request body otherwise the request will fail
5. Enter the number of results you want in the `top_k` field, max value is around 40
6. Click on the `Execute` button

## How to ask a question about a document

This endpoint allows you to ask questions about all of the documents stored in the vector database. The endpoint returns the answer to the question, along with the document sources that the answer was extracted from.

1. Open the `answer` tab
2. Click on the `Try it out` button
3. Enter a query for the document retriever in the `query` field (replacing the _string_, leave the quotes)
4. Filter field is optional, if you don't use it, remove the `filter` field from the request body otherwise the request will fail
5. Enter the number of documents you want in the `top_k` field, max value is around 40
3. Enter a question in the `prompt` field (replacing the _string_, leave the quotes)
6. Click on the `Execute` button

"""

app = FastAPI(
    title="ZURU Code Compliance AI API",
    description=description,
    version="0.0.1",
    servers=[{"url": "https://zuru-code-compliance-poc.fly.dev"}],
)
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="static")

# Create a sub-application, in order to access just the query endpoints in the OpenAPI schema, found at http://0.0.0.0:8000/sub/openapi.json when the app is running locally
sub_app = FastAPI(
    title="Retrieval Plugin API",
    description="A retrieval API for querying and filtering documents based on natural language queries and metadata",
    version="1.0.0",
    servers=[{"url": "https://zuru-code-compliance-poc.fly.dev"}],
)
app.mount("/sub", sub_app)


@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            DocumentMetadata.parse_raw(metadata)
            if metadata
            else DocumentMetadata(source=Source.file)
        )
    except:
        metadata_obj = DocumentMetadata(source=Source.file)

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"str({e})")


@app.post(
    "/upsert",
    response_model=UpsertResponse,
)
async def upsert(
    request: UpsertRequest = Body(...),
):
    try:
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.post(
    "/query",
    response_model=QueryResponse,
)
async def query_main(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@sub_app.post(
    "/query",
    response_model=QueryResponse,
    description="Accepts search query objects with query and optional filter. Break down complex questions into sub-questions. Refine results by criteria, e.g. time / source, don't do this often. Split queries if ResponseTooLargeError occurs.",
)
async def query(
    request: QueryRequest = Body(...),
):
    try:
        results = await datastore.query(
            request.queries,
        )
        return QueryResponse(results=results)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")
    
    
@app.post(
    "/gpt4Test",
    response_model=GPT4TestResponse,
)
async def gpt4Test(
    request: GPT4TestRequest = Body(...),
):
    try:
        chat = ChatOpenAI(model_name="gpt-4")
        res = chat([HumanMessage(content=request.prompt)])
        return GPT4TestResponse(result=res.content)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")
    
@app.post(
    "/answer",
    response_model=AnswerResponse,
)
async def answer(
    request: AnswerRequest = Body(...),
):
    try:
        retrievedQueries = await datastore.query(
            request.queries,
        )
        chat = ChatOpenAI(model_name="gpt-4")
        
        finalPrompt = 'Given the following extracts from a collection of building codes: '
        for que in retrievedQueries:
            for doc in que.results:
                finalPrompt += doc.text + '. '
        finalPrompt += request.prompt
        print('Final Prompt: ', finalPrompt)
        res = chat([HumanMessage(content=finalPrompt)])
        return AnswerResponse(result=res.content, sources=retrievedQueries)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")


@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()


def start():
    uvicorn.run("server.main:app", host="0.0.0.0", port=6888, reload=True)
