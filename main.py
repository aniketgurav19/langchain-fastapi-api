from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import uvicorn
import json

app = FastAPI()

# LLM setup
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Prompt
prompt = PromptTemplate(
    template="""You are a strict productivity coach.

Analyze this user's daily activity:

Date: {date}
Logs: {logs}

Return ONLY valid JSON:

{
  "praise": "...",
  "time_leaks": "...",
  "action_items": ["...", "...", "..."]
}
""",
    input_variables=["date", "logs"]
)

chain = prompt | llm

# Request model
class DailyData(BaseModel):
    date: str
    logs: List[Dict[str, Any]]
    user_id: str

# Root check
@app.get("/")
def home():
    return {"message": "LangChain API is running 🚀"}

# AI endpoint
@app.post("/analyze-day")
async def analyze_day(data: DailyData):
    try:
        response = chain.invoke({
            "date": data.date,
            "logs": data.logs
        })

        content = response.content

        # Try to parse JSON
        try:
            return json.loads(content)
        except:
            return {"raw_response": content}

    except Exception as e:
        return {"error": str(e)}

# Local run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
