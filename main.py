from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Optional
from rag import RAG

app = FastAPI()
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

    @model_validator(mode="after")
    def validate_query(self):
        if self.query is None or not isinstance(self.query, str):
            raise ValueError("'query' must be a string.")
        return self

@app.post("/data")
async def index_data(request_body: DataRequest):
    try:
        metadata_dict = request_body.metadata.dict(by_alias=True) if request_body.metadata else {}
        rag.indexing_pipeline(request_body.data, metadata_dict)
        return {"message": "Document indexed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/rag")
async def execute_rag_pipeline(request_body: QueryRequest):
    """
    Executes the RAG pipeline based on the user query.
    """
    try:
        query = request_body.query
        response = rag.rag_pipeline(query)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing RAG pipeline: {str(e)}")