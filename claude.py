import os
from dataclasses import dataclass, field
from google import genai
from google.genai import types


# ── Shims so the rest of the codebase keeps working unchanged ─────────────────

@dataclass
class ContentBlock:
    """Mirrors Anthropic's TextBlock / ToolUseBlock interface."""
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class GeminiMessage:
    """Mimics anthropic.types.Message so Chat/CliChat need no changes."""
    content: list
    stop_reason: str  # "end_turn" | "tool_use"


# Alias so `isinstance(msg, Message)` works in tools.py / chat.py
Message = GeminiMessage


# ── Tool format conversion ────────────────────────────────────────────────────

def _anthropic_tool_to_genai(tool: dict) -> types.Tool:
    """Convert Anthropic-style tool dict → google-genai Tool."""
    schema = tool.get("input_schema", {})
    props = schema.get("properties", {})
    required = schema.get("required", [])

    parameters = {
        "type": "OBJECT",
        "properties": {
            k: {
                "type": _json_type(v.get("type", "string")),
                "description": v.get("description", ""),
            }
            for k, v in props.items()
        },
        "required": required,
    }

    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=parameters,
            )
        ]
    )


def _json_type(t: str) -> str:
    return {
        "string":  "STRING",
        "number":  "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array":   "ARRAY",
        "object":  "OBJECT",
    }.get(t, "STRING")


# ── Message history conversion ────────────────────────────────────────────────

def _build_genai_contents(messages: list) -> list[types.Content]:
    """Convert Anthropic-style messages list → google-genai Content list."""
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        content = msg["content"]
        parts = []

        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))

        elif isinstance(content, list):
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

                if btype == "text":
                    text = block.get("text") if isinstance(block, dict) else block.text
                    parts.append(types.Part.from_text(text=text))

                elif btype == "tool_use":
                    name  = block.get("name")  if isinstance(block, dict) else block.name
                    args  = block.get("input") if isinstance(block, dict) else block.input
                    parts.append(types.Part.from_function_call(name=name, args=args))

                elif btype == "tool_result":
                    tid    = block.get("tool_use_id") if isinstance(block, dict) else block.tool_use_id
                    result = block.get("content")     if isinstance(block, dict) else block.content
                    parts.append(types.Part.from_function_response(
                        name=tid,
                        response={"result": result},
                    ))

        if parts:
            contents.append(types.Content(role=role, parts=parts))

    return contents


# ── Drop-in replacement for the original Claude class ────────────────────────

class Claude:
    def __init__(self, model: str):
        api_key = os.getenv("GEMINI_API_KEY", "")
        assert api_key, "Error: GEMINI_API_KEY cannot be empty. Update .env"
        self.client = genai.Client(api_key=api_key)
        self.model_name = "models/gemini-2.5-flash"

    def add_user_message(self, messages: list, message):
        if isinstance(message, GeminiMessage):
            messages.append({"role": "user", "content": message.content})
        elif isinstance(message, list):
            messages.append({"role": "user", "content": message})
        else:
            messages.append({"role": "user", "content": message})

    def add_assistant_message(self, messages: list, message):
        if isinstance(message, GeminiMessage):
            messages.append({"role": "assistant", "content": message.content})
        else:
            messages.append({"role": "assistant", "content": message})

    def text_from_message(self, message: GeminiMessage) -> str:
        return "\n".join(
            block.text for block in message.content if block.type == "text"
        )

    def chat(
        self,
        messages: list,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> GeminiMessage:

        # Convert tools
        genai_tools = [_anthropic_tool_to_genai(t) for t in tools] if tools else None

        # Convert all messages to genai Contents
        contents = _build_genai_contents(messages)

        # Build config
        config_kwargs = {
            "temperature": min(temperature, 2.0),
        }
        if stop_sequences:
            config_kwargs["stop_sequences"] = stop_sequences
        if system:
            config_kwargs["system_instruction"] = system
        if genai_tools:
            config_kwargs["tools"] = genai_tools

        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        # Parse response into GeminiMessage
        content_blocks = []
        stop_reason = "end_turn"
        counter = 0

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.function_call:
                    stop_reason = "tool_use"
                    counter += 1
                    content_blocks.append(ContentBlock(
                        type="tool_use",
                        id=f"tool_{counter}",
                        name=part.function_call.name,
                        input=dict(part.function_call.args),
                    ))
                elif part.text:
                    content_blocks.append(ContentBlock(type="text", text=part.text))

        return GeminiMessage(content=content_blocks, stop_reason=stop_reason)