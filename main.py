###load dependencies
from typing import Union, List, Tuple
from langchain.agents import Tool, tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import StrOutputParser
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate

from langchain_core.agents import AgentAction, AgentFinish

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from callbacks import AgentCallbackHandler

import json

###load the environment variables
import os, boilerplate


boilerplate.load_dotenv()




# The tool decoractor is used to define a function as a tool that can be used by an agent. 
# It takes a function as an argument and returns a new function that can be used by an agent. 
# The tool decorator is used to define the name, description, and return type of the tool. 
# In this case, the tool is called "get_text_length" and it takes a string as input and returns an integer. 
# The tool is used to get the length of a text in characters.
# ========
# Note that we can no longer call it like a regular function and need to use the .invoke method to call it passing the input={dictionary} 
# ie. print(get_text_length.invoke(input={"text":"Dog"}))  #note the syntax for call it via invoke

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")

def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation: ",
    llm_prefix: str = "Thought: ",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process"""
    thoughts = ""

    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"

#we do this because the stop token filters out anything from observation onwards...so we nee to build it back in if it exists.
#kind of hacky / confusing in the flow?

    return thoughts

@tool 
def get_text_length(text:str)-> int:
    """Returns the length of the text in characters.
    """
    text = text.strip("'\n").strip('"')

    return len(text)



        


if __name__ == "__main__":
    print("#### React-Langchain")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}
    
    Use the following format in your response:

    Question: the input question you must answer
    Thought: you should always think about what to do
    
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    
    Question: {input}
    Thought: {agent_scratchpad}
    """
    #manually removing the agent_scratchpad from the template....apparently when used as a hub.pull prompt it is managed automatically by the AgentExecutor function. its just an internal memory for the agent.



    #partial plugs in the already known values into the template. eg. already known things like the list of tools and their names
    prompt = PromptTemplate(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
        )


    stop_tokens = [""]#["\nObservation"],

    #llm = ChatAnthropic(temperature=0.0, model='claude-3-5-haiku-latest', stop=stop_tokens, callbacks=[AgentCallbackHandler()])
    llm = ChatOpenAI(temperature=0.0,model="gpt-4o-mini", stop=["\nObservation"], callbacks=[AgentCallbackHandler()])
    #llm = ChatOllama(temperature=0.0, model="llama3.2_32k", stop=stop_tokens, callbacks=[AgentCallbackHandler()])
    #llm = ChatGroq(temperature=0.0, model="llama-3.2-3b-preview", stop=["\nObservation"], callbacks=[AgentCallbackHandler()])


    intermediate_steps = []

    #agent chains are writted in the LangChain Expression Language (LCEL)
    # https://python.langchain.com/docs/concepts/lcel/
    # https://python.langchain.com/docs/how_to/sequence/   <= pipe operator
    # pipe operator takes the output of the left hand side and inputs it into the right hand side
    # each item in the chain is called a 'runnable'. the chain itself is also a runnable => ie can be called with the invoke method
    agent = (
        {
            #the lambda function in this case simply just extracts the input element from the dictionary that gets passed into it.....not sure why we need to do this but maybe its nesseary later
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser() #shows prints inside loop       
        #| StrOutputParser()       #doesnt show prints inside loop
        )

    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        ###### Typing
        #this is a way of typing a variable to say that the variable can only contain either an AgentAction or AgentFinish object.
        #the union allows us to join types....so we might say Union[str,int] if we wanted a magic variable that can hold either a string or an integer.
        #syntax: variablename : type1 | type2 | type3 = Value

        agent_step: Union[AgentAction, AgentFinish]  = agent.invoke(
            {
                "input": "What is the length in characters of the text: DOG?",
                "agent_scratchpad": intermediate_steps #this is the history of all the steps that have been taken so far.  We pass this in as context to the LLM so it can remember what has happened before.
            }
            )

       #output is of type Union[AgentAction, AgentFinish]
        #we need to check if it is an action or a finish


        #isinstance checks to see if object a (agent_step) is an instance of the class b (AgentAction).
        #so we receive agent_step....but we dont know if its an action or a finish or a...and isinstance figures out which

        if isinstance(agent_step, AgentAction):
            #this means that the agent has taken an action and needs to be given more information
            #the action will have a tool name and a tool input which we can use to query our database or other tools
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input
            
            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agent_step, str(observation)))  #appending tuple of action and observation to intermediate steps list 

            print("====================================")
            print(f"{agent_step=}")
            print(f"{tool_name=}")
            print(f"{tool_input=}")  #huh cool! the fstring {name=} adding = operator prints the variable anme and value...saves redundant boilerplate typing!
            print(f"{observation=}")
            print("intermediate_steps=" + json.dumps(intermediate_steps, indent=2, default=str,sort_keys=True))
            
        
        elif isinstance(agent_step, AgentFinish):
            #this means that the agent has finished its task and returned a final answer
            print("====================================")
            print(f"{agent_step=}")
            print("intermediate_steps=" + json.dumps(intermediate_steps, indent=2, default=str))
            print(">>>Agent finished with output:", agent_step.return_values["output"])
            
            #end the loop
            isfinished = True

    

"""things to figure out....why is tool_input 'DOG"  \nObservation: 3' rather than just DOG
    It seems to be related to a bad prompt including additional headings in the input field

    Is the agent expecting some sort of prompt that ive strayed too far from?
    Now its including everything and its thoughts in the input...
    """


    


