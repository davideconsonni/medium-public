import asyncio
import base64
import json
import os
from pathlib import Path

from browser_use import Agent
from browser_use.browser.browser import BrowserConfig, Browser
from browser_use.browser.context import BrowserContextConfig, BrowserContext
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

load_dotenv()

# Create output directories
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
(output_dir / "images").mkdir(exist_ok=True)
(output_dir / "video").mkdir(exist_ok=True)
(output_dir / "json").mkdir(exist_ok=True)

async def step_callback(state, model_output, step_number):
    try:
        # Save screenshot if available
        last_screenshot = getattr(state, "screenshot", None)
        if last_screenshot:
            img_path = output_dir / "images" / f"{step_number}.png"
            img_data = base64.b64decode(str(last_screenshot))
            with open(img_path, "wb") as f:
                f.write(img_data)

        # Save agent output as JSON
        json_path = output_dir / "json" / f"step_{step_number}.json"

        if hasattr(model_output, "model_dump"):
            output_dict = model_output.model_dump()
        else:
            output_dict = model_output

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)

    except Exception:
        pass

# Initialize LLM
llm = ChatVertexAI(
    project=os.getenv("GOOGLE_PROJECT_ID"),
    location=os.getenv("GOOGLE_REGION"),
    model_name=os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash-preview-05-20"),
)

# Browser configuration
browser_config = BrowserConfig(
    headless=False,
    disable_security=True
)

context_config = BrowserContextConfig(
    wait_for_network_idle_page_load_time=3.0,
    browser_window_size={'width': 1280, 'height': 1100},
    locale='en-US',
    highlight_elements=True,
    viewport_expansion=500,
    save_recording_path=str(output_dir / "video"),
)

browser = Browser(config=browser_config)
context = BrowserContext(browser=browser, config=context_config)

task = f"""
Objective: Test GitHub search functionality and repository navigation

1. Navigate to "https://github.com"
2. Verify the page title contains "GitHub: Where the world builds software"
3. Search for the repository "microsoft/vscode" using the search bar
4. On search results page, verify:
   a. The first result is "microsoft/vscode"
   b. It's marked as "Public"
5. Click on the first search result
6. On repository page, verify:
   a. Page title contains "vscode"
   b. "About" section with description exists
   c. Star count is greater than 100,000
7. Click on the "Issues" tab
8. On Issues page:
   a. Verify open issues exist
   b. Use LLM to analyze the first 5 issue titles:
      - Identify if any issue mentions "performance" problems
      - Check if any issue has a "bug" label
9. Return verification results in JSON format:
    - "homepage_title_correct": true/false
    - "search_functional": true/false
    - "repo_public_correct": true/false
    - "star_threshold_passed": true/false
    - "performance_issue_detected": true/false
    - "bug_issue_detected": true/false
    - "issues_page_accessible": true/false
"""

async def main():
    agent = Agent(
        task=task,
        llm=llm,
        browser_context=context,
        register_new_step_callback=step_callback,
        max_actions_per_step=15,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
