import asyncio
import json
import functions_framework
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    Part
)
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

FALCON_MCP_URL = "https://falcon-mcp-server-262878782932.asia-south1.run.app"
MCP_ENDPOINT = f"{FALCON_MCP_URL}/mcp"
PROJECT_ID = "gemini-enterprise-non-prod"
LOCATION = "asia-south1"

def get_mcp_auth_token():
    """Generates an identity token for service-to-service communication."""
    try:
        creds, _ = google.auth.default()
        auth_req = Request()
        creds.refresh(auth_req)
        token = id_token.fetch_id_token(auth_req, FALCON_MCP_URL)
        return {"Authorization": f"Bearer {token}"}
    except Exception as exc:
        print(f"Token generation note: {exc}")
        return {}

def mcp_to_vertex_tools(mcp_tools):
    """Converts MCP Tool schemas into Vertex AI FunctionDeclarations."""
    func_decls = []
    for t in mcp_tools:
        schema = t.inputSchema if hasattr(t, "inputSchema") and t.inputSchema else {"type": "object", "properties": {}}
        clean_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", [])
        }
        func_decls.append(
            FunctionDeclaration(
                name=t.name,
                description=t.description or f"Tool: {t.name}",
                parameters=clean_schema
            )
        )
    return [Tool(function_declarations=func_decls)] if func_decls else []

async def perform_agent_work(user_prompt: str) -> str:
    """Orchestrates tool discovery, Gemini reasoning, tool execution, and response synthesis."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    headers = get_mcp_auth_token()
    print(f"Connecting to Falcon MCP at {MCP_ENDPOINT}...")

    
    async with streamable_http_client(MCP_ENDPOINT, headers=headers) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("MCP session initialized successfully.")

            
            tools_resp = await session.list_tools()
            vertex_tools = mcp_to_vertex_tools(tools_resp.tools)
            print(f"Discovered {len(tools_resp.tools)} tools from Falcon MCP.")

            system_instruction = (
                "You are an enterprise SecOps assistant with real-time access to CrowdStrike Falcon tools. "
                "Use the available tools to query devices, investigate detections, and inspect security telemetry. "
                "Always present findings clearly with relevant counts, hostnames, and risk levels."
            )

            
            model = GenerativeModel(
                "gemini-1.5-flash-001",
                system_instruction=[system_instruction],
                tools=vertex_tools
            )

            
            chat = model.start_chat()
            response = chat.send_message(user_prompt)

            
            first_candidate = response.candidates[0]
            function_calls = [part.function_call for part in first_candidate.content.parts if part.function_call]

            if function_calls:
                call = function_calls[0]
                tool_name = call.name
                tool_args = dict(call.args)
                print(f"Gemini requested tool: {tool_name} with arguments: {tool_args}")

                tool_result = await session.call_tool(tool_name, arguments=tool_args)
                raw_result = [c.text for c in tool_result.content if hasattr(c, "text")]
                result_str = "\n".join(raw_result) if raw_result else str(tool_result.content)
                print(f"Tool execution returned: {result_str[:200]}...")

                
                final_response = chat.send_message(
                    Part.from_function_response(
                        name=tool_name,
                        response={"content": result_str}
                    )
                )
                return final_response.text

            return response.text

@functions_framework.http
def gemini_falcon_agent(request):
    request_json = request.get_json(silent=True)
    if not request_json or "prompt" not in request_json:
        return (json.dumps({"error": "Please provide a JSON body with a 'prompt' key."}), 400, {'Content-Type': 'application/json'})

    user_prompt = request_json["prompt"]
    print(f"Incoming prompt: {user_prompt}")
    try:
        final_answer = asyncio.run(perform_agent_work(user_prompt))
        return (json.dumps({"response": final_answer}), 200, {'Content-Type': 'application/json'})
    except Exception as exc:
        print(f"Execution error: {exc}")
        return (json.dumps({"error": str(exc)}), 500, {'Content-Type': 'application/json'})
