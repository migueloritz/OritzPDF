from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class SummarizationRequest(BaseModel):
    document_id: str
    page_range: Optional[List[int]] = None
    max_length: int = Field(default=150, ge=50, le=500)
    min_length: int = Field(default=50, ge=30, le=200)
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "page_range": [1, 5],
                "max_length": 200
            }
        }


class SummarizationResponse(BaseModel):
    document_id: str
    summary: str
    pages_summarized: List[int]
    confidence_score: Optional[float] = None


class QuestionAnswerRequest(BaseModel):
    document_id: str
    question: str
    context_pages: Optional[List[int]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "question": "What is the main conclusion of the document?"
            }
        }


class QuestionAnswerResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    confidence_score: float
    source_pages: List[int]
    context: Optional[str] = None


class TranslationRequest(BaseModel):
    document_id: str
    target_language: str
    source_language: Optional[str] = None  # Auto-detect if not provided
    page_range: Optional[List[int]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "target_language": "es",
                "page_range": [1, 3]
            }
        }


class TranslationResponse(BaseModel):
    document_id: str
    source_language: str
    target_language: str
    translated_text: str
    pages_translated: List[int]


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    CUSTOM = "custom"


class ExtractedEntity(BaseModel):
    text: str
    entity_type: EntityType
    confidence: float
    page_number: int
    context: Optional[str] = None


class DataExtractionRequest(BaseModel):
    document_id: str
    entity_types: Optional[List[EntityType]] = None
    extract_tables: bool = True
    extract_numbers: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc123",
                "entity_types": ["person", "organization", "money"],
                "extract_tables": True
            }
        }


class DataExtractionResponse(BaseModel):
    document_id: str
    entities: List[ExtractedEntity]
    tables: Optional[List[Dict[str, Any]]] = None
    numerical_data: Optional[Dict[str, Any]] = None


class ComparisonRequest(BaseModel):
    document_id_1: str
    document_id_2: str
    comparison_type: str = Field(default="both", pattern="^(text|visual|both)$")
    detailed_diff: bool = True
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id_1": "doc123",
                "document_id_2": "doc456",
                "comparison_type": "both"
            }
        }


class ComparisonDifference(BaseModel):
    page_number: int
    diff_type: str  # added, removed, modified
    original_text: Optional[str] = None
    modified_text: Optional[str] = None
    visual_difference: Optional[bool] = None


class ComparisonResponse(BaseModel):
    document_id_1: str
    document_id_2: str
    similarity_score: float
    total_differences: int
    differences: List[ComparisonDifference]
    summary: str