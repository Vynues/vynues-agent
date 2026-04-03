# Coding Agent Prompt (OpenAI Agents SDK)
Build a **modular Jupyter notebook (Python)** that implements a **research → analysis → report** multi-agent workflow
using the **OpenAI Agents SDK**. The notebook must be structured so that the **Researcher agent can be swapped out**
later for a different project without changing the rest of the pipeline.
**Requirements**
1. **Notebook format**: Use clear section headings and separate code/markdown cells.
2. **Dependencies**: Include a setup cell installing `openai-agents`, `python-dotenv`, `pydantic`, and `requests`.
3. **Modularity**:
- Create a dedicated function or class to build the `Researcher` agent.
- The pipeline should accept a `research_agent` object so it can be replaced later.
4. **Agents**:
- `Researcher` (uses a web search tool or stub tool).
- `Analyst` (summarizes trends/risks/insights).
- `Writer` (returns structured output: executive summary, markdown report, follow-up questions).
5. **Tooling**:
- Implement a real Tavily search `function_tool` so the Researcher can search online.
- The Tavily API key is provided in the environment (`TAVILY_API_KEY`). Load it via `python-dotenv`.
- Use the Tavily search endpoint `https://api.tavily.com/search` with JSON payload: `api_key`, `query`,
`max_results`.
6. **Pipeline**:
- A manager function `run_pipeline(user_query, researcher_agent)` that orchestrates:
1. Researcher → summary
2. Analyst → insights
3. Writer → final report
7. **Output formatting**:
- Display the executive summary, markdown report, and follow-up questions cleanly.
8. **Example run**:
- Include a sample query and show the output.
**Notes**
- Use the correct import path for the Agents SDK: `from agents import Agent, Runner, function_tool, SQLiteSession` (do
**not** use `openai_agents`).
- Keep the Researcher creation isolated so I can later swap it for a different domain-specific researcher.
- Use `SQLiteSession` to share context between agents and **always pass a session_id**, e.g.,
`SQLiteSession("research_pipeline")`.
- Use Pydantic output models for structured outputs.
- Ensure the Researcher’s instructions explicitly say to use the Tavily search tool to gather sources before
summarizing.
Deliver only the notebook code and markdown cells, ready to run.