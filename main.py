import asyncio
import json
import functions_framework
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import id_token
import vertexai
from vertexai.generative_models import GenerativeModel
from mcp.client.sse import sse_client
from mcp import ClientSession

FALCON_MCP_URL = "https://falcon-mcp-server-nbupl3xsea-el.a.run.app"
SSE_URL = f"{FALCON_MCP_URL}/sse"
PROJECT_ID = "gemini-enterprise-non-prod"
LOCATION = "asia-south1"

def get_mcp_auth_token():
    try:
        creds, _ = google.auth.default()
        auth_req = Request()
        creds.refresh(auth_req)
        token = id_token.fetch_id_token(auth_req, FALCON_MCP_URL)
        return {"Authorization": f"Bearer {token}"}
    except Exception as exc:
        print(f"Error fetching token: {exc}")
        raise

async def perform_agent_work(user_prompt: str) -> str:
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    headers = get_mcp_auth_token()
    print(f"Connecting to Falcon MCP at {SSE_URL}...")

    async with sse_client(SSE_URL, headers=headers) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools_resp.tools])

            system_prompt = (
                "You are an enterprise security operations AI assistant with access to CrowdStrike Falcon tools:\n"
                f"{tool_descriptions}\n"
                "Use the available tools to answer queries and recommend security actions."
            )

            model = GenerativeModel("gemini-1.5-flash-001", system_instruction=[system_prompt])
            final_prompt = f"Using available tools, answer this question: '{user_prompt}'"
            response = model.generate_content(final_prompt)
            return response.text

@functions_framework.http
def gemini_falcon_agent(request):
    request_json = request.get_json(silent=True)
    if not request_json or "prompt" not in request_json:
        return (json.dumps({"error": "Please provide a JSON body with a 'prompt' key."}), 400, {'Content-Type': 'application/json'})

    user_prompt = request_json["prompt"]
    try:
        final_answer = asyncio.run(perform_agent_work(user_prompt))
        return (json.dumps({"response": final_answer}), 200, {'Content-Type': 'application/json'})
    except Exception as exc:
        return (json.dumps({"error": str(exc)}), 500, {'Content-Type': 'application/json'})
