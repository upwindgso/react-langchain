###load dependencies
from typing import Union, List
from langchain.agents import Tool, tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import StrOutputParser
from langchain.tools.render import render_text_description
from langchain.prompts import PromptTemplate

from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


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

@tool 
def get_text_length(text:str)-> int:
    """Returns the length of the text in characters.
    """
    text = text.strip("'\n").strip('"')

    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool wtih name {tool_name} not found")

if __name__ == "__main__":
    print("#### React-Langchain")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

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
    Thought:
    """
    #manually removing the agent_scratchpad from the template....apparently when used as a hub.pull prompt it is managed automatically by the AgentExecutor function. its just an internal memory for the agent.



    #partial plugs in the already known values into the template. eg. already known things like the list of tools and their names
    prompt = PromptTemplate(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
        )

    #llm = ChatAnthropic(temperature=0.1, model='claude-3-5-haiku-latest', stop=["\nObservation"])
    llm = ChatOpenAI(temperature=0.1,model="gpt-4o-mini", stop=["\nObservation"])
    #llm = ChatOllama(temperature=0.1, model="llama3.2_32k", stop=["\nObservation"])

    #agent chains are writted in the LangChain Expression Language (LCEL)
    # https://python.langchain.com/docs/concepts/lcel/
    # https://python.langchain.com/docs/how_to/sequence/   <= pipe operator
    # pipe operator takes the output of the left hand side and inputs it into the right hand side
    # each item in the chain is called a 'runnable'. the chain itself is also a runnable => ie can be called with the invoke method
    agent = {"input": lambda x: x["input"] } | prompt | llm | ReActSingleInputOutputParser() #StrOutputParser()
        #the lambda function in this case simply just extracts the input element from the dictionary that gets passed into it.....not sure why we need to do this but maybe its nesseary later



    ###### Typing
    #this is a way of typing a variable to say that the variable can only contain either an AgentAction or AgentFinish object.
    #the union allows us to join types....so we might say Union[str,int] if we wanted a magic variable that can hold either a string or an integer.
    #syntax: variablename : type1 | type2 | type3 = Value

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of 'DOG' in characters?"})

    print(agent_step)

    #isinstance checks to see if object a (agent_step) is an instance of the class b (AgentAction).
    #so we receive agent_step....but we dont know if its an action or a finish or a...and isinstance figures out which

    if isinstance(agent_step, AgentAction):
        #this means that the agent has taken an action and needs to be given more information
        #the action will have a tool name and a tool input which we can use to query our database or other tools
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input
        
        observation = tool_to_use.func(str(tool_input))
        print(f"{tool_name=} {tool_input=}")  #huh cool! the fstring {name=} adding = operator prints the variable anme and value...saves redundant boilerplate typing!
        print(f"{observation=}")

    elif isinstance(agent_step, AgentFinish):
        #this means that the agent has finished its task and returned a final answer
        print("Agent finished with output:", agent_step.output)




    


