# Ensure all SQLAlchemy models are registered in Base.metadata
from models import (
    auth_models,
    document_models,
    planner_models,
    quiz_models,
    summary_models,
    chat_models,
    discussion_models,
    subject_models,
    note_models,
    past_paper_models,
)
