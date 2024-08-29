import os
from lyzr_agent import LyzrAgent
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LYZR_API_KEY = os.getenv("LYZR_API_KEY")

# Streamlit page configuration
st.set_page_config(
    page_title="Lyzr Job Assistant",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

st.title("Lyzr Job Assistant")
st.markdown("### Welcome to the Lyzr Job Assistant!")

# Initialize the LyzrAgent
Agent = LyzrAgent(api_key=LYZR_API_KEY, llm_api_key=OPENAI_API_KEY)


@st.cache_resource(show_spinner=True)
def create_agent():
    """Create and return the agent for generating cold emails."""
    env_id = Agent.create_environment(
        name="Job Assistant Environment",
        features=[
            {
                "type": "SHORT_TERM_MEMORY",
                "config": {},
                "priority": 0
            },
            {
                "type": "TOOL_CALLING",
                "config": {"max_tries":3},
                "priority": 0
            }
        ],
        tools=["perplexity_search"]
    )

    # System prompt for guiding the agent's behavior
    prompt = """
   You are an AI assistant trained to assist with job applications. Your task is to handle all aspects of the job application process when provided with a job description (JD). Follow the steps below:
    
    Company Analysis:
        -Identify the company's HQ location, sector, brief description.
        -Provide insights into stock sentiment, employee satisfaction, and management quality.
        -Compare the seniority of the current job profile to the user's profile.
    
    Account Creation:
        -Suggest an email and a randomly generated password for account creation.
    
    SWOT Analysis:
        -Conduct a SWOT analysis by assessing strengths, weaknesses, opportunities, and threats related to the job.
    
    Match Analysis:
        -Compare the job description with the user's CV.
        -Evaluate the fit based on seniority level, industry experience, technical and soft skills.
        -Consider company factors like profitability, ownership stability, size, gross margin, and IT spending.
        -Consider language and cultural factors, including whether the company hires internationals.
        -Provide a consolidated rating on the user's fit for the position and a nuanced conclusion on whether it is worth applying.
        -Ask if the user wants to proceed.
    
    Application Materials Preparation (if user continues):
        -Adjust the CV to match the job description, ensuring it's ATS-friendly but doesn't look AI-generated.
        -Create a concise, ATS-friendly cover letter.
        -Provide 2-3 short LinkedIn messages designed to grab attention.
    
    Job Tracking Outline:
        -Summarize the job application details including company, position, date, follow-up, sentiment, and account creation details (email and password).
        -Present this information in a tabular format suitable for copying into Excel.
    
    Ensure the entire process can be completed efficiently with a single copy-paste of the job description.
    """

    agent_id = Agent.create_agent(
        env_id=env_id['env_id'],
        system_prompt=prompt,
        name="Job Assistant Agent"
    )

    return agent_id


# Maintain the agent session using Streamlit's session state
if "agent_id" not in st.session_state:
    st.session_state.agent_id = create_agent()

# Input area for the product description and target audience
query = st.text_area("Enter Job Description", height=150)

if st.button("Submit"):
    if query.strip():
        with st.spinner("Researching..."):
            response = Agent.send_message(
                agent_id=st.session_state.agent_id['agent_id'],
                user_id="default_user",
                session_id="new_session",
                message=query
            )
            # Display the generated email
            st.markdown(f"\n\n{response['response']}")
    else:
        st.warning("Please provide Job Description...")

# Optional footer or credits
st.markdown("---")
st.markdown("Powered by Lyzr and OpenAI")
