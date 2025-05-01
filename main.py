from fastapi import FastAPI, HTTPException, Response, Depends
from pydantic import BaseModel, Field, model_validator
from typing import Optional
from RAG import RAG
from auth import verify_basic_auth

app = FastAPI(
    title="RAG API",
    description="API for interacting with a Retrieval-Augmented Generation (RAG) model — add and delete documents.",
    tag="Knowlegde base",
    version="1.0.0"
)
rag = RAG()

class Metadata(BaseModel):
    created_by: Optional[str] = Field(None, alias="created by")
    category: Optional[str]

    @model_validator(mode="after")
    def validate_strings(self):
        if self.created_by is not None and not isinstance(self.created_by, str):
            raise ValueError("'created by' must be a string if provided.")
        if self.category is not None and not isinstance(self.category, str):
            raise ValueError("'category' must be a string if provided.")
        return self

class DataRequest(BaseModel):
    data: str
    metadata: Optional[Metadata] = None

    @model_validator(mode="after")
    def validate_data(self):
        if self.data is None or not isinstance(self.data, str):
            raise ValueError("'data' field is mandatory and must be a string.")
        return self

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_query(self):
        if self.query is None or not isinstance(self.query, str):
            raise ValueError("'query' must be a string.")
        return self

@app.post("/data", tags=["Documents"])
async def index_data(request_body: DataRequest, _: None = Depends(verify_basic_auth)):
    try:
        metadata_dict = request_body.metadata.dict(by_alias=True) if request_body.metadata else {}
        document_id = rag.indexing_pipeline(request_body.data, metadata_dict)
        return {"message": "Document indexed successfully", "document_id": document_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@app.delete("/data/{id}", tags=["Documents"])
async def delete_data(id: str, _: None = Depends(verify_basic_auth)):
    try:
        rag.delete_document(id)
        return Response(status_code=204)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/rag", include_in_schema=False)
async def execute_rag_pipeline(request_body: QueryRequest):
    try:
        response, session_id = rag.rag_pipeline(request_body.query, request_body.session_id)
        return {"response": response, "session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing RAG pipeline: {str(e)}")