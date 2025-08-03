from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from .tools.custom_tool import send_otp, verify_otp, initiate_video_call, verify_face

@CrewBase
class KycAgentCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def kyc_initiator(self) -> Agent:
        return Agent(
            config=self.agents_config['kyc_initiator'],
            tools=[send_otp, verify_otp],
            verbose=True,
            llm=ChatOpenAI(temperature=0.7, model_name="gpt-4")
        )

    @agent
    def video_verifier(self) -> Agent:
        return Agent(
            config=self.agents_config['video_verifier'],
            tools=[initiate_video_call, verify_face],
            verbose=True,
            llm=ChatOpenAI(temperature=0.7, model_name="gpt-4")
        )

    @task
    def initiate_kyc_task(self) -> Task:
        return Task(
            config=self.tasks_config['initiate_kyc_task'],
            agent=self.kyc_initiator()
        )

    @task
    def video_verification_task(self) -> Task:
        return Task(
            config=self.tasks_config['video_verification_task'],
            agent=self.video_verifier()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )