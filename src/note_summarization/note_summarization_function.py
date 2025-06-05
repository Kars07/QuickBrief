import logging
import glob
import os

from dotenv import load_dotenv
load_dotenv() 

from pydantic import Field
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import EmbedderRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class NoteSummarizationFunctionConfig(FunctionBaseConfig, name="note_summarization"):
    """
    AIQ Toolkit function for document summarization.
    """
    tool_names: list[FunctionRef] = Field(default=[], description="List of tool names to use")
    llm_name: LLMRef = Field(description="LLM to use")
    max_history: int = Field(default=100, description="Maximum number of history messages to provide to the agent")
    ingest_glob: str
    description: str
    chunk_size: int = 1024
    auto_summarize: bool = Field(default=True, description="Automatically summarize loaded documents on first message")
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5" # type: ignore
    
    
@register_function(config_type=NoteSummarizationFunctionConfig)
async def note_summarization_function(
    config: NoteSummarizationFunctionConfig, builder: Builder):
    
    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.schema import Document

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    
    file_paths: list[str] = []
    logger.info("Looking for documents in directory from pattern: %s", config.ingest_glob)
    
    # Extract the directory from the glob pattern
    if '*' in config.ingest_glob:
        data_dir = os.path.dirname(config.ingest_glob.split('*')[0])
    else:
        data_dir = os.path.dirname(config.ingest_glob)
    
    logger.info("Using data directory: %s", data_dir)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        logger.warning("Directory does not exist: %s", data_dir)
        docs = [Document(page_content="No documents found - data directory does not exist.", metadata={"source": "fallback"})]
    else:
        # List files directly from the directory
        logger.info("Files in directory %s: %s", data_dir, os.listdir(data_dir))
        # Process files in directory
        file_paths: list[str] = []
        
        # Handle PDF and TXT files directly
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.pdf') or filename.lower().endswith('.txt'):
                file_paths.append(os.path.join(data_dir, filename))
        
        logger.info("Found %d suitable files: %s", len(file_paths), file_paths)
        
        if not file_paths:
            logger.warning("No PDF or TXT files found in %s", data_dir)
            docs = [Document(page_content="No documents found to summarize.", metadata={"source": "fallback"})]
        else:
            docs = []
            for file_path in file_paths:
                try:
                    if file_path.lower().endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                    else:
                        loader = TextLoader(file_path)
                    loaded_docs = await loader.aload()
                    docs.extend(loaded_docs)
                    logger.info("Successfully loaded %d documents from %s", len(loaded_docs), file_path)
                except Exception as e:
                    logger.error("Error loading file %s: %s", file_path, str(e))

    if not docs:
        logger.warning("No documents were successfully loaded")
        docs = [Document(page_content="No documents were successfully loaded.", metadata={"source": "fallback"})]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size)
    documents = text_splitter.split_documents(docs)
    
    if not documents:
        logger.warning("No documents after splitting")
        documents = [Document(page_content="No content available for processing.", metadata={"source": "fallback"})]

    logger.info("Creating vector store with %d document chunks", len(documents))
    vector = await FAISS.afrom_documents(documents, embeddings)

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "text_file_ingest",
        config.description,
    )
    
    from langchain import hub
    from langchain.agents import AgentExecutor
    from langchain.agents import create_react_agent
    from langchain.schema import BaseMessage
    from langchain_core.messages import trim_messages
    from aiq.data_models.api_server import AIQChatRequest
    from aiq.data_models.api_server import AIQChatResponse
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    
    tools = [retriever_tool] + builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    llm_kwargs = {
        "model_name": "meta/llama-4-maverick-17b-128e-instruct",
        "temperature": 0.0,
        "max_tokens": 1024,
        "api_key": os.getenv("NVIDIA_API_KEY") 
    }
    llm_chat = ChatNVIDIA(**llm_kwargs)
    

    # This ensures it has all necessary variables including agent_scratchpad
    prompt = hub.pull("hwchase17/react-chat")
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt, stop_sequence=["\nObservation"])
    
    agent_executor = AgentExecutor(agent=agent,
                                  tools=tools,
                                  max_iterations=50,
                                  handle_parsing_errors=True,
                                  verbose=True,
                                  return_intermediate_steps=True)
    
    # Extract document filenames for auto-summarization
    document_names = [os.path.basename(file_path) for file_path in file_paths] if 'file_paths' in locals() else []
    
    async def _response_fn(input_message: AIQChatRequest) -> AIQChatResponse:
        messages = input_message.messages
        
        # Auto-summarize loaded documents on first message if enabled
        if config.auto_summarize and len(messages) <= 2 and document_names:
            # Check if this is a generic/greeting message
            last_message_content = str(messages[-1].content).lower()
            generic_phrases = ["hello", "hi", "hey", "summarize", "help", "document"]
            
            if any(phrase in last_message_content for phrase in generic_phrases) or len(last_message_content) < 30:
                #Custom first-message response that automatically retrieves document contents
                prompt = f"Please retrieve and summarize the contents of the document(s) in a detailed manner: {', '.join(document_names)}. Provide a detailed summary including key points and organize information by sections."
                
                # Create custom message with instruction to use the tool
                response = await agent_executor.ainvoke({
                    "input": prompt,
                    "chat_history": []
                })
                return AIQChatResponse.from_string(response["output"])
        
        # Standard processing for follow-up messages
        last_message = messages[-1].content
        chat_history = trim_messages(messages=[m.model_dump() for m in messages],
                                    max_tokens=config.max_history,
                                    strategy="last",
                                    token_counter=len,
                                    start_on="human",
                                    include_system=True)
        response = await agent_executor.ainvoke({"input": last_message, "chat_history": chat_history})
        return AIQChatResponse.from_string(response["output"])
    
    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        print("Function exited early!")
    finally:
        print("Cleaning up note_summarization workflow.")