import httpx
import asyncio
import uuid


from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Part,
    TextPart,
    MessageSendParams,
    Message,
    Role,
    SendMessageRequest,
    SendMessageResponse,
)


async def get_agent_card(url: str) -> AgentCard | None:
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)

    card: AgentCard | None = await resolver.get_agent_card()

    return card


async def wait_agent_ready(url, timeout=10):
    
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                return True
            else:
                print(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def create_a2a_client(url: str) -> tuple[A2AClient, httpx.AsyncClient]:
    """
    Create a reusable A2A client
    
    Returns:
        tuple: (A2AClient, httpx.AsyncClient) - httpx_client should be closed after use
    """
    card = await get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=300.0)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)
    return client, httpx_client


async def send_message(
    url, message, task_id=None, context_id=None
) -> SendMessageResponse:
    
    card = await get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=300.0)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)

    try:
        message_id = uuid.uuid4().hex
        params = MessageSendParams(
            message=Message(
                role=Role.user,
                parts=[Part(TextPart(text=message))],
                message_id=message_id,
                task_id=task_id,
                context_id=context_id,
            )
        )
        request_id = uuid.uuid4().hex
        req = SendMessageRequest(id=request_id, params=params)
        response = await client.send_message(request=req)
        return response
    finally:
        await httpx_client.aclose()


async def send_message_with_client(
    client: A2AClient, message: str, task_id=None, context_id=None
) -> SendMessageResponse:
    
    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    response = await client.send_message(request=req)
    return response
