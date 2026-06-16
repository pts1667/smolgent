# smolgent
Small barebones LLM harness library

## Reasoning preservation

smolgent keeps assistant reasoning fields (`reasoning`, `reasoning_content`,
and `reasoning_details`) in chat session history and forwards them on later
chat-completions requests. For llama.cpp, models/templates that support seeing
prior thinking still need the server started with the appropriate template
configuration, such as:

```powershell
llama-server --chat-template-kwargs '{"preserve_thinking": true}'
```
