"""Agentic workflow implementation using LangGraph.

This module implements a simplified agentic RAG workflow with the following flow:
Router → Reformulator → Retriever → Generator, plus a Tool node for extensibility.
The workflow uses LangGraph StateGraph for state management and conditional routing.

All retrieve-type queries (both fresh and follow-up) flow through the reformulator,
which uses the LLM to either resolve references from history (follow-ups) or optimize
the query for better vector search retrieval (standalone queries).
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..config import WorkflowConfig
from ..exceptions import InferenceError
from ..generation.generator import ResponseGenerator
from ..llm.client import LLMClient
from .reformulator import QueryReformulator


# Follow-up detection patterns (used by reformulator node to pick prompt mode)
CONTEXTUAL_REFERENCES = [
    "that", "this", "it", "they", "them", "those", "these",
    "the one", "the same", "which one",
]

FOLLOW_UP_PHRASES = [
    "tell me more", "more about", "what about", "how about",
    "how does it", "how do they", "can you compare",
    "what's the difference", "is it better", "any other",
    "similar to", "like that", "another option",
]


class AgentState(TypedDict):
    """State for the agentic workflow.
    
    Attributes:
        messages: Conversation messages (managed by LangGraph)
        context: Retrieved context from the retrieval pipeline
        route: Routing decision ("retrieve", "tool", "respond")
        tool_result: Result from tool execution
        reformulated_query: Reformulated/optimized query for retrieval
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    route: str
    tool_result: str
    reformulated_query: str


class AgenticWorkflow:
    """LangGraph-based agentic RAG workflow.
    
    Flow: Router → Reformulator → Retriever → Generator (with Tool branch).
    
    The reformulator always runs for retrieve-type queries, using the LLM to either:
    - Resolve pronoun/reference follow-ups using conversation history
    - Optimize standalone queries for better vector search retrieval
    
    Attributes:
        config: Workflow configuration
        llm_client: LLM client for routing decisions
        retriever: Retrieval pipeline for document search
        generator: Response generator for final output
        reformulator: Query reformulator for search optimization
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
        self.reformulator = QueryReformulator(llm_client)
        
        # Build and compile the workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Retrieve path: Router → Reformulator → Retriever → Generator
        Tool path:     Router → Tool → Generator
        Respond path:  Router → Generator
        
        Returns:
            StateGraph instance with nodes and edges configured
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("reformulator", self._reformulator_node)
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
                "reformulator": "reformulator",
                "tool": "tool",
                "generator": "generator"
            }
        )
        
        # Reformulator always flows to retriever
        workflow.add_edge("reformulator", "retriever")
        
        # Retriever and tool both flow to generator
        workflow.add_edge("retriever", "generator")
        workflow.add_edge("tool", "generator")
        
        # Generator is the finish
        workflow.add_edge("generator", END)
        
        return workflow
    
    def _router_node(self, state: AgentState) -> Dict[str, Any]:
        """Route query to retrieve (via reformulator), tool, or direct response.
        
        Routing priority:
        1. Tool keywords → tool node
        2. Product keywords → retrieve (reformulator → retriever)
        3. Default fallback → retrieve
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with routing decision
        """
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"route": "respond"}
        
        query_lower = user_message.lower()
        
        # Check for tool keywords first (more specific)
        if any(keyword in query_lower for keyword in self.config.tool_keywords):
            return {"route": "tool"}
        
        # Everything else goes through retrieve (reformulator will optimize the query)
        return {"route": "retrieve"}
    
    def _is_follow_up_query(self, query: str, history: List[BaseMessage]) -> bool:
        """Detect if query is a follow-up referencing prior product context.
        
        Used by the reformulator node to decide which prompt mode to use:
        - True → follow-up mode (resolve references from history)
        - False → standalone mode (optimize query for vector search)
        
        Args:
            query: The user query to analyze
            history: Recent conversation history
            
        Returns:
            True if the query is a follow-up to product-related conversation
        """
        if not history:
            return False
        
        query_lower = query.lower()
        
        has_contextual_ref = any(
            ref in query_lower for ref in CONTEXTUAL_REFERENCES
        )
        has_follow_up_phrase = any(
            phrase in query_lower for phrase in FOLLOW_UP_PHRASES
        )
        
        if not (has_contextual_ref or has_follow_up_phrase):
            return False
        
        # Check if recent assistant messages contain product-related content
        recent_assistant_messages = []
        for msg in reversed(history):
            if isinstance(msg, AIMessage):
                recent_assistant_messages.append(msg.content.lower())
                if len(recent_assistant_messages) >= 3:
                    break
        
        for msg_content in recent_assistant_messages:
            if any(keyword in msg_content for keyword in self.config.product_keywords):
                return True
        
        return False
    
    def _reformulator_node(self, state: AgentState) -> Dict[str, Any]:
        """Reformulate query for optimal retrieval using the LLM.
        
        Detects whether the query is a follow-up (referencing prior conversation)
        and picks the appropriate reformulation mode:
        - Follow-up: resolves pronouns/references using conversation history
        - Standalone: optimizes the raw query for better vector search
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with reformulated query
        """
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        
        if not user_message:
            return {"reformulated_query": ""}
        
        # Get conversation history (excluding the current user message)
        history = []
        for msg in state["messages"][:-1]:
            if not isinstance(msg, SystemMessage):
                history.append(msg)
        
        # Detect if this is a follow-up to pick the right prompt mode
        is_follow_up = self._is_follow_up_query(user_message, history)
        
        try:
            reformulated = self.reformulator.reformulate(
                query=user_message,
                history=history if history else None,
                is_follow_up=is_follow_up
            )
            return {"reformulated_query": reformulated}
        except Exception:
            return {"reformulated_query": user_message}
    
    def _retriever_node(self, state: AgentState) -> Dict[str, Any]:
        """Retrieve relevant documents using retrieval pipeline.
        
        This node uses the retrieval pipeline to find relevant documents
        for the user query and adds the formatted context to the state.
        Uses the reformulated query if available (from follow-up detection).
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with retrieved context
        """
        # Check if we have a reformulated query (from follow-up detection)
        query_to_use = state.get("reformulated_query", "")
        
        # If no reformulated query, get the latest user message
        if not query_to_use:
            user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if not user_message:
                return {"context": ""}
            
            query_to_use = user_message
        
        try:
            # Use the retrieval pipeline to get relevant documents
            retrieval_result = self.retriever.retrieve(query_to_use)
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
    
    def _route_decision(self, state: AgentState) -> Literal["reformulator", "tool", "generator"]:
        """Conditional edge: decide next node based on route.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name to visit
        """
        route = state.get("route", "respond")
        
        if route == "retrieve":
            return "reformulator"
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
                "tool_result": "",
                "reformulated_query": ""
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
                "tool_result": "",
                "reformulated_query": ""
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