from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import uvicorn

app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

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

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

prompt = PromptTemplate(
    template="""You are a strict productivity coach. Analyze the following raw JSON log of a user's day:

Date: {date}
Logs: {logs}

{format_instructions}
""",
    input_variables=["date", "logs"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)

chain = prompt | llm | output_parser

class DailyData(BaseModel):
    date: str
    logs: List[Dict[str, Any]]
    user_id: str

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

@app.get("/")
def home():
    return {"message": "LangChain API is running 🚀"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
