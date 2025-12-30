"""Agentic workflow implementation using LangGraph.

This module implements a simplified agentic RAG workflow with three main nodes:
Router → Retriever → Generator, plus a Tool node for extensibility demonstration.
The workflow uses LangGraph StateGraph for state management and conditional routing.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..config import WorkflowConfig
from ..exceptions import InferenceError
from ..generation.generator import ResponseGenerator
from ..llm.client import LLMClient


class AgentState(TypedDict):
    """State for the agentic workflow.
    
    This TypedDict defines the state that flows through the LangGraph workflow.
    Each node can read from and write to this state.
    
    Attributes:
        messages: Conversation messages (managed by LangGraph)
        context: Retrieved context from the retrieval pipeline
        route: Routing decision ("retrieve", "tool", "respond")
        tool_result: Result from tool execution
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    route: str
    tool_result: str


class AgenticWorkflow:
    """LangGraph-based agentic RAG workflow with 3 main nodes + tool node.
    
    This class implements a simplified agentic workflow that:
    1. Routes queries based on content analysis
    2. Retrieves relevant documents when needed
    3. Executes tools for specific requests (product comparison demo)
    4. Generates responses using context and conversation history
    
    The workflow is designed to be extensible for future MCP/tools integration.
    
    Attributes:
        config: Workflow configuration
        llm_client: LLM client for routing decisions
        retriever: Retrieval pipeline for document search
        generator: Response generator for final output
        workflow: LangGraph StateGraph instance
        app: Compiled workflow application
    """
    
    def __init__(
        self,
        config: WorkflowConfig,
        llm_client: LLMClient,
        retrieval_pipeline,  # RetrievalPipeline from retrieval module
        response_generator: ResponseGenerator
    ):
        """Initialize the agentic workflow.
        
        Args:
            config: Workflow configuration
            llm_client: Initialized LLM client
            retrieval_pipeline: Initialized retrieval pipeline
            response_generator: Initialized response generator
        """
        self.config = config
        self.llm_client = llm_client
        self.retriever = retrieval_pipeline
        self.generator = response_generator
        
        # Build and compile the workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with Router → Retriever/Tool → Generator.
        
        Returns:
            StateGraph instance with nodes and edges configured
        """
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("tool", self._tool_node)
        workflow.add_node("generator", self._generator_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "retriever": "retriever",
                "tool": "tool",
                "generator": "generator"
            }
        )
        
        # Add edges from retriever and tool to generator
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("tool", "generator")
        
        # Set finish point
        workflow.add_edge("generator", END)
        
        return workflow
    
    def _router_node(self, state: AgentState) -> Dict[str, Any]:
        """Route query to retriever, tool, or direct response.
        
        This node analyzes the user query and decides which path to take:
        - "retrieve": Product-related queries needing database lookup
        - "tool": Requests to compare products (demo functionality)
        - "respond": General conversation or questions answerable without retrieval
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with routing decision
        """
        # Get the latest user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            # No user message found, default to direct response
            return {"route": "respond"}
        
        # Simple keyword-based routing for MVP
        query_lower = user_message.lower()
        
        # Check for tool keywords first (more specific)
        if any(keyword in query_lower for keyword in self.config.tool_keywords):
            return {"route": "tool"}
        
        # Check for product keywords
        if any(keyword in query_lower for keyword in self.config.product_keywords):
            return {"route": "retrieve"}
        
        # Default to direct response for general queries
        return {"route": "respond"}
    
    def _retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents using retrieval pipeline.
        
        This node uses the retrieval pipeline to find relevant documents
        for the user query and adds the formatted context to the state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieved context
        """
        # Get the latest user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"context": ""}
        
        try:
            # Use the retrieval pipeline to get relevant documents
            retrieval_result = self.retriever.retrieve(user_message)
            context = retrieval_result.formatted_context
            
            return {"context": context}
            
        except Exception as e:
            # Log error and continue without context
            # In a production system, you might want more sophisticated error handling
            return {"context": f"[Retrieval failed: {str(e)}]"}
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute tool (product comparison demo).
        
        This is a demonstration node showing how tools could be integrated.
        In a full implementation, this would use MCP or other tool frameworks.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with tool execution result
        """
        # Get the latest user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"tool_result": "No query provided for tool execution"}
        
        # Simple demo tool: product comparison
        # In a real implementation, this would call actual comparison tools
        tool_result = f"""Product Comparison Tool (Demo)
Query: {user_message}

This is a demonstration of tool integration. In a full implementation, 
this would execute actual product comparison logic, potentially using:
- MCP (Model Context Protocol) servers
- External APIs for product data
- Specialized comparison algorithms

For now, this serves as a placeholder showing the workflow structure."""
        
        return {"tool_result": tool_result}
    
    def _generator_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate final response using context.
        
        This node uses the response generator to create the final response,
        incorporating any retrieved context, tool results, and conversation history.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with generated response
        """
        # Get the latest user message
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            response = "I didn't receive a clear question. Could you please rephrase?"
            return {"messages": [AIMessage(content=response)]}
        
        # Prepare context from retrieval or tool execution
        context = state.get("context", "")
        tool_result = state.get("tool_result", "")
        
        # Combine context and tool result if both exist
        combined_context = ""
        if context and tool_result:
            combined_context = f"Retrieved Context:\n{context}\n\nTool Result:\n{tool_result}"
        elif context:
            combined_context = context
        elif tool_result:
            combined_context = tool_result
        
        # Get conversation history (excluding the current user message)
        history = []
        for msg in state["messages"][:-1]:  # Exclude the last message (current query)
            if not isinstance(msg, SystemMessage):  # Exclude system messages
                history.append(msg)
        
        try:
            # Generate response using the response generator
            response = self.generator.generate(
                query=user_message,
                context=combined_context if combined_context else None,
                history=history if history else None
            )
            
            return {"messages": [AIMessage(content=response)]}
            
        except Exception as e:
            # Fallback response on generation failure
            error_response = (
                "I apologize, but I encountered an error while generating a response. "
                "Please try rephrasing your question."
            )
            return {"messages": [AIMessage(content=error_response)]}
    
    def _route_decision(self, state: AgentState) -> Literal["retriever", "tool", "generator"]:
        """Conditional edge: decide next node based on route.
        
        This function is used by LangGraph to determine which node to visit
        next based on the routing decision made by the router node.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name to visit
        """
        route = state.get("route", "respond")
        
        if route == "retrieve":
            return "retriever"
        elif route == "tool":
            return "tool"
        else:
            return "generator"
    
    def run(self, query: str, history: Optional[List[BaseMessage]] = None) -> str:
        """Execute workflow and return response.
        
        Args:
            query: User query string
            history: Optional conversation history
            
        Returns:
            Generated response string
            
        Raises:
            InferenceError: If workflow execution fails
        """
        try:
            # Prepare initial state
            messages = []
            if history:
                messages.extend(history)
            messages.append(HumanMessage(content=query))
            
            initial_state = {
                "messages": messages,
                "context": "",
                "route": "",
                "tool_result": ""
            }
            
            # Execute the workflow
            result = self.app.invoke(initial_state)
            
            # Extract the final response
            final_messages = result.get("messages", [])
            if final_messages:
                # Get the last AI message
                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage):
                        return msg.content
            
            # Fallback if no AI message found
            return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            raise InferenceError(
                f"Workflow execution failed: {str(e)}",
                error_code="WORKFLOW_EXECUTION_ERROR",
                details={"query": query, "error": str(e)}
            ) from e
    
    async def arun(self, query: str, history: Optional[List[BaseMessage]] = None) -> str:
        """Async workflow execution.
        
        Args:
            query: User query string
            history: Optional conversation history
            
        Returns:
            Generated response string
            
        Raises:
            InferenceError: If workflow execution fails
        """
        try:
            # Prepare initial state
            messages = []
            if history:
                messages.extend(history)
            messages.append(HumanMessage(content=query))
            
            initial_state = {
                "messages": messages,
                "context": "",
                "route": "",
                "tool_result": ""
            }
            
            # Execute the workflow asynchronously
            result = await self.app.ainvoke(initial_state)
            
            # Extract the final response
            final_messages = result.get("messages", [])
            if final_messages:
                # Get the last AI message
                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage):
                        return msg.content
            
            # Fallback if no AI message found
            return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            raise InferenceError(
                f"Async workflow execution failed: {str(e)}",
                error_code="ASYNC_WORKFLOW_EXECUTION_ERROR",
                details={"query": query, "error": str(e)}
            ) from e