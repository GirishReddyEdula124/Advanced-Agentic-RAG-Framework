"""
Agent Manager: Implementation of an Agentic framework for multi-step reasoning.
Uses LangChain's OpenAIFunctionsAgent for robust tool orchestration.
"""

from typing import List, Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

# Local imports
from core.rag.rag_engine import RAGEngine

class AgentManager:
    """
    Orchestrator for Agentic workflows.
    Manages task decomposition, tool selection, and execution loop.
    """

    def __init__(self, model_name: str = "gpt-4-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.rag_engine: Optional[RAGEngine] = None
        self.agent_executor: Optional[AgentExecutor] = None

    def _setup_rag_tool(self, engine: RAGEngine) -> Tool:
        """Creates a tool from the RAG engine."""
        return Tool(
            name="knowledge_retrieval",
            func=engine.query,
            description="Useful for retrieving specialized information from the internal knowledge base."
        )

    def _get_default_tools(self) -> List[Tool]:
        """Defines the set of tools available to the agent."""
        # Note: In a real scenario, this would include Web Search, DB Lookups, etc.
        tools = []
        
        # Add RAG tool if engine is provided
        if self.rag_engine:
            tools.append(self._setup_rag_tool(self.rag_engine))
            
        # Example: Mock search tool
        tools.append(Tool(
            name="general_search",
            func=lambda q: f"Search results for: {q}",
            description="Useful for general web search when specific data is not available in the internal DB."
        ))
        
        return tools

    def initialize_agent(self, rag_engine: Optional[RAGEngine] = None):
        """
        Initializes the ReAct agent with tools and memory.
        """
        self.rag_engine = rag_engine
        tools = self._get_default_tools()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional research agent. Use the provided tools to answer complex queries. "
                       "Decompose the user's request into smaller tasks if necessary. "
                       "Always cite your sources if provided by the tools."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, user_input: str) -> str:
        """
        Executes the agentic reasoning loop.
        """
        if not self.agent_executor:
            self.initialize_agent()
            
        try:
            response = self.agent_executor.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            return f"Agent Execution Error: {str(e)}"

if __name__ == "__main__":
    # Integration Test
    from langchain.schema import Document
    
    # 1. Setup RAG Engine
    rag = RAGEngine()
    docs = [Document(page_content="The Project X was launched in 2024.", metadata={"source": "confidential_internal"})]
    rag.initialize_retriever(docs)
    
    # 2. Setup Agent
    manager = AgentManager()
    manager.initialize_agent(rag_engine=rag)
    
    # 3. Run Complex Task
    task = "When was Project X launched? Also, check if there are any public search results about its success."
    output = manager.run(task)
    print(f"\nFinal Agent Output:\n{output}")
