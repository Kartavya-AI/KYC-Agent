import sys
from kyc_agent.crew import KycAgentCrew

def run():
    inputs = {
        'aadhaar_number': '1234-5678-9012',
        'mobile_number': '+919876543210'
    }
    KycAgentCrew().crew().kickoff(inputs=inputs)


def train():
    inputs = {
        'aadhaar_number': '1234-5678-9012',
        'mobile_number': '+919876543210'
    }
    try:
        KycAgentCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
