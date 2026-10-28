import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
## Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")


groq_api_key = os.getenv("GROQ_API_KEY")  # Fetch the Groq API Key from .env file

## Set upi the Stramlit apppython -m streamlit run app.py
st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
st.title("Text To Math Problem Solver Uing llama3")


llm=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)


## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()

def safe_wikipedia(query: str) -> str:
    try:
        return wikipedia_wrapper.run(query)
    except Exception:
        return "Wikipedia is unavailable right now. Answer from your own knowledge and mention that it could not be verified."

wikipedia_tool=Tool(
    name="Wikipedia",
    func=safe_wikipedia,

## Initializa the MAth tool

import numexpr

def safe_calculator(expression: str) -> str:
    try:
        cleaned = expression.strip().strip("`").strip('"').strip("'")
        result = numexpr.evaluate(cleaned)
        return str(result.item() if hasattr(result, "item") else result)
    except Exception as e:
        return f"Could not calculate: {e}"

calculator=Tool(
    name="Calculator",
    func=safe_calculator,
    description="A tool for answering math related questions. Only input a mathematical expression, like (5-2)+(7-3)"
)

prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning_tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

## initialize the agents

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math assistant. Use the Calculator tool for arithmetic, the Wikipedia tool for facts, and the Reasoning tool for logic questions. Give a clear step-by-step explanation and the final answer."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

all_tools = [wikipedia_tool, calculator, reasoning_tool]
agent = create_tool_calling_agent(llm, all_tools, agent_prompt)
assistant_agent = AgentExecutor(agent=agent, tools=all_tools, verbose=False, handle_parsing_errors=True)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

## LEts start the interaction
question=st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.invoke({"input": question}, {"callbacks": [st_cb]})["output"]
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response:')
            st.success(response)

    else:
        st.warning("Please enter the question")

# Custom footer styling
footer_html = """
    <style>
    body {
        margin: 0;
        padding: 0;
        min-height: 100%;
        display: flex;
        flex-direction: column;
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(135deg, #6a82fb, #fc5c7d);  /* Smooth purple-pink gradient */
        color: white;
        text-align: center;
        padding: 8px 16px;  /* Reduced padding to decrease height */
        font-size: 14px;  /* Reduced font size */
        font-family: 'Arial', sans-serif;
        border-top: 2px solid #ffffff33;
        box-shadow: 0 -2px 5px rgba(0, 0, 0, 0.3);
        animation: fadeIn 3s ease-out;
    }

    .footer b {
        color: #e0fffc;  /* Soft cyan color */
        font-size: 16px;  /* Reduced font size */
        transition: color 0.3s ease;
    }

    .footer b:hover {
        color: #ffdd00;  /* Bright yellow hover effect */
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>

    <div class="footer">
        Developed by <b>Saifalikhan</b> | © Faculty of IT 
    </div>
"""

# Render the footer in the Streamlit app
st.markdown(footer_html, unsafe_allow_html=True)









