from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
import os
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Load OpenAI API Key
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define response format
response_schemas = [
    ResponseSchema(
        name="praise",
        description="One sentence praising their deep work or productive time."
    ),
    ResponseSchema(
        name="time_leaks",
        description="Identify the biggest waste of time or distraction from the data."
    ),
    ResponseSchema(
        name="action_items",
        description="A list of 3 strict, actionable improvements for tomorrow.",
        type="list"
    )
]

# Output parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Prompt template (FIXED)
prompt = PromptTemplate(
    template="""You are a strict productivity coach.

Analyze this user's daily activity:

Date: {date}
Logs: {logs}

Give structured output:
{format_instructions}
""",
    input_variables=["date", "logs"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)

# Chain
chain = prompt | llm | output_parser

# Request model
class DailyData(BaseModel):
    date: str
    logs: List[Dict[str, Any]]
    user_id: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "LangChain API is running 🚀"}

# AI analysis endpoint
@app.post("/analyze-day")
async def analyze_day(data: DailyData):
    try:
        result = chain.invoke({
            "date": data.date,
            "logs": data.logs
        })
        return result
    except Exception as e:
        return {"error": str(e)}

# Run locally (not used by Render but safe to keep)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
