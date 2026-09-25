"""
Pydantic shemas for the FastAPI layer.
These describe the *public contract* of the API - seperate from the 
internal 'dict' state LangGraph passes betwwen nodes.
"""
from typing import Literal,Optional
from pydantic import BaseModel, Field

class chatMessage(BaseModel):
    role:Literal["user","assistant"]
    content:str

class QueryRequest(BaseModel):
    query: str = Field(...,description="User's VGC question",min_length=1)
    chat_hostory: Optiional[list[chatMessage]] = None

class QueryResponse(BaseModel):
    answer:str
    is_grounded:bool
    status:Literal["answered","declined"]
    retries_used:int
    run_id:str

    @classmethod
    def from_final_state(cls,state:dict) -> "QueryResponse":
        """Maps the raw LangGraph state dict to the API's public schema"""
        is_grounded = state.get("is_grounded",False)
        retries = state.get("_retry_count",0)
        return cls(
            answer=state.get("draft_answer",""),
            is_grounded=is_grounded,
            status="answered" if is_grounded else "declined",   
            retries_used=retries,
            run_id=state.get("run_id","")
        )

class HealthResponse(BaseModel):
    status:Literal["ok","degraded"]