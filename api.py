from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from kyc_agent.crew import KycAgentCrew

app = FastAPI()

class KycRequest(BaseModel):
    aadhaar_number: str
    mobile_number: str

def run_kyc_crew(inputs: dict):
    KycAgentCrew().crew().kickoff(inputs=inputs)

@app.post("/initiate-kyc")
async def initiate_kyc(request: KycRequest, background_tasks: BackgroundTasks):
    inputs = {
        "aadhaar_number": request.aadhaar_number,
        "mobile_number": request.mobile_number
    }
    background_tasks.add_task(run_kyc_crew, inputs)
    return {"message": "KYC process initiated successfully in the background."}

@app.get("/")
def read_root():
    return {"message": "KYC Agent API is running."}
