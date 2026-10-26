import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QuizTopicInput(BaseModel):
    id: str
    name: str

class QuizCreateRequest(BaseModel):
    title: str
    selected_documents: List[str]
    selected_topics: List[str]
    question_count: int = Field(default=10, ge=1, le=50)
    difficulty: str = "medium" # 'easy', 'medium', 'hard', 'mixed'
    question_types: List[str] # ['mcq', 'short_answer', 'coding']
    duration: int = Field(default=30, ge=5, le=180) # minutes
    scoring_configuration: Optional[Dict[str, Any]] = None

class QuizQuestionResponse(BaseModel):
    id: uuid.UUID
    topic_id: Optional[uuid.UUID]
    question_type: str
    difficulty: str
    question_text: str
    public_options: Optional[Dict[str, Any]]
    marks: float
    order_index: int

class QuizResponse(BaseModel):
    id: uuid.UUID
    title: str
    question_count: int
    duration: int
    created_at: datetime
    questions: Optional[List[QuizQuestionResponse]] = None

class QuizAttemptStartResponse(BaseModel):
    id: uuid.UUID
    quiz_id: uuid.UUID
    status: str
    start_timestamp: datetime
    expiry_timestamp: datetime

class QuizAnswerUpdate(BaseModel):
    student_answer: Optional[str] = None
    selected_options: Optional[Dict[str, Any]] = None
    submitted_code: Optional[str] = None
    programming_language: Optional[str] = None
    review_flag: Optional[bool] = None

class CodeRunRequest(BaseModel):
    code: str
    language: Optional[str] = "python"
    question_id: Optional[str] = None
    stdin: Optional[str] = ""

class CodeRunResponse(BaseModel):
    status: str
    output: Optional[str] = None
    error: Optional[str] = None
    exit_code: Optional[int] = 0
    provider: Optional[str] = None

class CodeSubmitRequest(BaseModel):
    question_id: str
    code: str
    language: Optional[str] = "python"

