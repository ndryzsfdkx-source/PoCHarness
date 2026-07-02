#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    evaluate_python_code,
)
from .secb.sanitizer.tool import SanitizerParserTool
from .tools import PipelineTool, Tool


@dataclass
class PreTool:
    name: str
    inputs: dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, **kwargs):
        import os

        # Check if running in SEC-bench context (Docker container)
        # When running with secb-run, sandbox checks are disabled since everything runs in Docker
        is_secb_run = os.getenv("SMOLAGENTS_SECB_RUN", "").lower() in ("1", "true", "yes")

        if is_secb_run:
            # Disable sandbox checks by allowing all imports
            self.authorized_imports = ["*"]
        elif authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))

        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        state = {}
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
            )[0]  # The second element is boolean is_final_answer
        )
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def forward(self, question):
        user_input = input(f"{question} => Type your answer here:")
        return user_input


class DuckDuckGoSearchTool(Tool):
    """Web search tool that performs searches using the DuckDuckGo search engine.

    Args:
        max_results (`int`, default `10`): Maximum number of search results to return.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.
        **kwargs: Additional keyword arguments for the `DDGS` client.

    Examples:
        ```python
        >>> from smolagents import DuckDuckGoSearchTool
        >>> web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, rate_limit: float | None = 1.0, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `ddgs` to run this tool: for instance run `pip install ddgs`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        self._enforce_rate_limit()
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

    def _enforce_rate_limit(self) -> None:
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class GoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs a google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi"):
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def forward(self, query: str, filter_year: int | None = None) -> str:
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)


class ApiWebSearchTool(Tool):
    """Web search tool that performs API-based searches.
    By default, it uses the Brave Search API.

    This tool implements a rate limiting mechanism to ensure compliance with API usage policies.
    By default, it limits requests to 1 query per second.

    Args:
        endpoint (`str`): API endpoint URL. Defaults to Brave Search API.
        api_key (`str`): API key for authentication.
        api_key_name (`str`): Environment variable name containing the API key. Defaults to "BRAVE_API_KEY".
        headers (`dict`, *optional*): Headers for API requests.
        params (`dict`, *optional*): Parameters for API requests.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.

    Examples:
        ```python
        >>> from smolagents import ApiWebSearchTool
        >>> web_search_tool = ApiWebSearchTool(rate_limit=50.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, URLs, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        api_key_name: str = "",
        headers: dict = None,
        params: dict = None,
        rate_limit: float | None = 1.0,
    ):
        import os

        super().__init__()
        self.endpoint = endpoint or "https://api.search.brave.com/res/v1/web/search"
        self.api_key_name = api_key_name or "BRAVE_API_KEY"
        self.api_key = api_key or os.getenv(self.api_key_name)
        self.headers = headers or {"X-Subscription-Token": self.api_key}
        self.params = params or {"count": 10}
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def forward(self, query: str) -> str:
        import requests

        self._enforce_rate_limit()
        params = {**self.params, "q": query}
        response = requests.get(self.endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()
        results = self.extract_results(data)
        return self.format_markdown(results)

    def extract_results(self, data: dict) -> list:
        results = []
        for result in data.get("web", {}).get("results", []):
            results.append(
                {"title": result["title"], "url": result["url"], "description": result.get("description", "")}
            )
        return results

    def format_markdown(self, results: list) -> str:
        if not results:
            return "No results found."
        return "## Search Results\n\n" + "\n\n".join(
            [
                f"{idx}. [{result['title']}]({result['url']})\n{result['description']}"
                for idx, result in enumerate(results, start=1)
            ]
        )


class WebSearchTool(Tool):
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, engine: str = "duckduckgo"):
        super().__init__()
        self.max_results = max_results
        self.engine = engine

    def forward(self, query: str) -> str:
        results = self.search(query)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        return self.parse_results(results)

    def search(self, query: str) -> list:
        if self.engine == "duckduckgo":
            return self.search_duckduckgo(query)
        elif self.engine == "bing":
            return self.search_bing(query)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def parse_results(self, results: list) -> str:
        return "## Search Results\n\n" + "\n\n".join(
            [f"[{result['title']}]({result['link']})\n{result['description']}" for result in results]
        )

    def search_duckduckgo(self, query: str) -> list:
        import requests

        response = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        parser = self._create_duckduckgo_parser()
        parser.feed(response.text)
        return parser.results

    def _create_duckduckgo_parser(self):
        from html.parser import HTMLParser

        class SimpleResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    # Store current result if all parts are present
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(self.current["description"])
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", [])
                    self.current["description"].append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        return SimpleResultParser()

    def search_bing(self, query: str) -> list:
        import xml.etree.ElementTree as ET

        import requests

        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")
        results = [
            {
                "title": item.findtext("title"),
                "link": item.findtext("link"),
                "description": item.findtext("description"),
            }
            for item in items[: self.max_results]
        ]
        return results


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[:max_length] + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        )

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return self._truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


class WikipediaSearchTool(Tool):
    """
    Search Wikipedia and return the summary or full text of the requested article, along with the page URL.

    Attributes:
        user_agent (`str`): Custom user-agent string to identify the project. This is required as per Wikipedia API policies.
            See: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
        language (`str`, default `"en"`): Language in which to retrieve Wikipedia article.
            See: http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (`Literal["summary", "text"]`, default `"text"`): Type of content to fetch. Can be "summary" for a short summary or "text" for the full article.
        extract_format (`Literal["HTML", "WIKI"]`, default `"WIKI"`): Extraction format of the output. Can be `"WIKI"` or `"HTML"`.

    Example:
        ```python
        >>> from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool
        >>> agent = CodeAgent(
        >>>     tools=[
        >>>            WikipediaSearchTool(
        >>>                user_agent="MyResearchBot (myemail@example.com)",
        >>>                language="en",
        >>>                content_type="summary",  # or "text"
        >>>                extract_format="WIKI",
        >>>            )
        >>>        ],
        >>>     model=InferenceClientModel(),
        >>> )
        >>> agent.run("Python_(programming_language)")
        ```
    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent
        self.language = language
        self.content_type = content_type

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class SpeechToTextTool(PipelineTool):
    default_checkpoint = "openai/whisper-large-v3-turbo"
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, an url, or a tensor.",
        }
    }
    output_type = "string"

    def __new__(cls, *args, **kwargs):
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        cls.pre_processor_class = WhisperProcessor
        cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls)

    def encode(self, audio):
        from .agent_types import AgentAudio

        audio = AgentAudio(audio).to_raw()
        return self.pre_processor(audio, return_tensors="pt")

    def forward(self, inputs):
        return self.model.generate(inputs["input_features"])

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]


class CmdTool(Tool):
    name = "cmd"
    description = "Execute a shell command inside the current environment and return stdout, stderr, and exit code."
    inputs = {
        "command": {
            "type": "string",
            "description": "The shell command to execute (will be run via bash -lc).",
        },
        "base_dir": {
            "type": "string",
            "description": "Optional base directory to run the command in.",
            "nullable": True,
        },
        "timeout": {
            "type": "integer",
            "description": "Optional timeout in seconds (default 120).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, command: str, base_dir: str | None = None, timeout: int | None = None) -> str:
        try:
            cwd = None
            if base_dir:
                p = Path(base_dir)
                if not p.exists() or not p.is_dir():
                    return f"Error: base_dir does not exist or is not a directory: {base_dir}"
                cwd = str(p)
            timeout_sec = 120 if (timeout is None or int(timeout) <= 0 or int(timeout) > 600) else int(timeout)
            completed = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout_sec,
            )
            if completed.returncode == 0:
                return _truncate_output((completed.stdout or "").rstrip("\n"))
            else:
                return _truncate_output((completed.stderr or "").rstrip("\n"))
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout_sec}s"
        except FileNotFoundError:
            # Fallback if bash is not available
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    shell=True,
                    cwd=cwd,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return f"Timed out after {timeout_sec}s"
            if completed.returncode == 0:
                return _truncate_output((completed.stdout or "").rstrip("\n"))
            else:
                return _truncate_output((completed.stderr or "").rstrip("\n"))


def _resolve_work_dir(base_dir: str | None = None) -> str:
    """Resolve the most likely project work directory for SEC-bench runs."""
    import os

    if base_dir:
        p = Path(base_dir)
        if not p.exists() or not p.is_dir():
            raise ValueError(f"work_dir does not exist or is not a directory: {base_dir}")
        return str(p)

    for env_name in ("SMOLAGENTS_WORK_DIR", "PWD"):
        candidate = os.getenv(env_name)
        if candidate and Path(candidate).exists() and Path(candidate).is_dir():
            return candidate

    src_dir = Path("/src")
    if src_dir.exists() and src_dir.is_dir():
        marker_files = ("Makefile", "CMakeLists.txt", "configure", "configure.ac", "Cargo.toml", "setup.py")
        child_dirs = [child for child in src_dir.iterdir() if child.is_dir() and not child.name.startswith(".")]
        for child in child_dirs:
            if any((child / marker).exists() for marker in marker_files):
                return str(child)
        if len(child_dirs) == 1:
            return str(child_dirs[0])
        return str(src_dir)

    return os.getcwd()


def _normalize_timeout(timeout: int | None, default: int = 300, maximum: int = 600) -> int:
    if timeout is None:
        return default
    try:
        timeout_sec = int(timeout)
    except (TypeError, ValueError):
        return default
    if timeout_sec <= 0:
        return default
    return min(timeout_sec, maximum)


def _truncate_output(text: str, max_chars: int = 16000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... output truncated to {max_chars} characters ..."


def _ensure_system_packages(packages: list[str]) -> tuple[bool, str]:
    """Install missing system packages inside the SEC-bench container if needed."""
    import shutil

    missing = [pkg for pkg in packages if shutil.which(pkg) is None]
    if not missing:
        return True, "already installed"

    try:
        subprocess.run(
            ["apt-get", "update"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        install = subprocess.run(
            ["apt-get", "install", "-y", *missing],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out while installing packages: {', '.join(missing)}"

    if install.returncode != 0:
        stderr = (install.stderr or "").strip()
        return False, f"Failed to install packages {missing}: {stderr}"

    return True, f"installed: {', '.join(missing)}"


def _extract_secb_repro_command(work_dir: str) -> str | None:
    """Best-effort extraction of the real repro command behind `secb repro`."""
    import re

    candidate_paths = [
        Path("/usr/local/bin/secb"),
        Path("/app/secb_helper.sh"),
        Path(work_dir) / "secb_helper.sh",
        Path(work_dir).parent / "secb_helper.sh",
    ]

    def _pick_command(body: str) -> str | None:
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("echo ", "cd ", "build ", "patch ", "return ", "exit ")):
                continue
            if "NOTE:" in line or line == "}":
                continue
            return line
        return None

    patterns = [
        r"repro\s*\(\)\s*\{(?P<body>.*?)\n\}",
        r"repro\(\)\s*\{(?P<body>.*?)\n\}",
    ]

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.DOTALL)
            if not match:
                continue
            command = _pick_command(match.group("body"))
            if command:
                return command
    return None


def _expand_runtime_command(command: str, work_dir: str) -> str:
    """Expand SEC-bench helper commands to the actual target command when possible."""
    normalized = command.strip()
    if normalized == "secb repro":
        extracted = _extract_secb_repro_command(work_dir)
        if extracted:
            return extracted
    return normalized


def _command_requires_shell(command: str) -> bool:
    """Return true when direct argv execution would change command semantics."""
    stripped = command.strip()
    return any(token in stripped for token in ("|", ">", "<", "&", ";", "$", "`", "(", ")", "{", "}"))


def _target_command_args(command: str) -> list[str]:
    if _command_requires_shell(command):
        return ["bash", "-lc", f"exec {command}"]
    try:
        args = shlex.split(command)
    except ValueError:
        return ["bash", "-lc", f"exec {command}"]
    if not args:
        return ["bash", "-lc", f"exec {command}"]
    env_assignments: list[str] = []
    while args and _is_env_assignment(args[0]):
        env_assignments.append(args.pop(0))
    if env_assignments:
        if not args:
            return ["bash", "-lc", command]
        import shutil

        env_bin = shutil.which("env") or "/usr/bin/env"
        return [env_bin, *env_assignments, *args]
    if "/" not in args[0]:
        import shutil

        resolved = shutil.which(args[0])
        if resolved:
            args[0] = resolved
        else:
            return ["bash", "-lc", f"exec {command}"]
    return args


def _is_env_assignment(value: str) -> bool:
    import re

    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*", value))


def _is_gdb_run_command(command: str) -> bool:
    stripped = command.strip()
    return stripped == "run" or stripped.startswith("run ")


def _is_gdb_setup_command(command: str) -> bool:
    stripped = command.strip().lower()
    return stripped.startswith(
        (
            "break ",
            "b ",
            "tbreak ",
            "tb ",
            "catch ",
            "watch ",
            "rwatch ",
            "awatch ",
            "set ",
            "handle ",
            "directory ",
            "source ",
            "sharedlibrary",
            "enable ",
            "disable ",
            "condition ",
            "delete ",
            "clear ",
        )
    )


def _is_gdb_breakpoint_command(command: str) -> bool:
    stripped = command.strip().lower()
    return stripped.startswith(("break ", "b ", "tbreak ", "tb ", "catch ", "watch ", "rwatch ", "awatch "))


def _split_gdb_commands(gdb_commands: str | None) -> tuple[list[str], str, list[str]]:
    """Split user commands into setup, a single run command, and post-run inspection."""
    commands = [cmd.strip() for cmd in (gdb_commands or "").split(";") if cmd.strip()]
    setup: list[str] = []
    post_run: list[str] = []
    run_command = "run"
    saw_run = False

    if any(_is_gdb_run_command(cmd) for cmd in commands):
        for cmd in commands:
            if _is_gdb_run_command(cmd):
                if not saw_run:
                    run_command = cmd
                    saw_run = True
                continue
            if saw_run:
                post_run.append(cmd)
            else:
                setup.append(cmd)
        return setup, run_command, post_run

    for cmd in commands:
        if _is_gdb_setup_command(cmd):
            setup.append(cmd)
        else:
            post_run.append(cmd)
    return setup, run_command, post_run


def _gdb_args(batch_commands: list[str], target_command: str) -> list[str]:
    args = ["gdb", "--batch"]
    for item in batch_commands:
        args.extend(["-ex", item])
    args.extend(["--args", *_target_command_args(target_command)])
    return args


def _breakpoint_setup_failed(output: str, setup_commands: list[str]) -> bool:
    if not any(_is_gdb_breakpoint_command(cmd) for cmd in setup_commands):
        return False

    lowered = output.lower()
    has_breakpoint = "breakpoint " in lowered or "catchpoint " in lowered or "watchpoint " in lowered
    has_failure = (
        "function " in lowered
        or "not defined" in lowered
        or "no source file named" in lowered
        or "no symbol table" in lowered
    )
    return has_failure and not has_breakpoint


def _run_shell_command(command: str, cwd: str, timeout: int, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        env=env,
        check=False,
    )


def _strip_existing_sanitizers(flags: str) -> str:
    import re

    patterns = [
        r"-fsanitize=[^\s]+",
        r"-fno-sanitize-recover=[^\s]+",
        r"-fsanitize-recover=[^\s]+",
    ]
    result = flags
    for pattern in patterns:
        result = re.sub(pattern, "", result)
    return " ".join(result.split())


def _summarize_pattern_output(output: str, patterns: list[str], context_lines: int = 2, max_lines: int = 120) -> str:
    if not output.strip():
        return "No output captured."

    lines = output.splitlines()
    keep: list[str] = []
    used_indices: set[int] = set()

    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(pattern in lowered for pattern in patterns):
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            for pos in range(start, end):
                if pos not in used_indices:
                    keep.append(lines[pos])
                    used_indices.add(pos)

    if not keep:
        keep = lines[-max_lines:]
    else:
        keep = keep[:max_lines]

    return "\n".join(keep)


class GDBTool(Tool):
    name = "gdb"
    description = (
        "Run a program under non-interactive gdb and return the crash backtrace, registers, and relevant output. "
        "Useful for debugging `secb repro` crashes inside SEC-bench containers."
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "Command to debug, for example `secb repro`.",
        },
        "work_dir": {
            "type": "string",
            "description": "Optional working directory for the command. If omitted, the SEC-bench source directory is auto-detected.",
            "nullable": True,
        },
        "gdb_commands": {
            "type": "string",
            "description": "Optional extra gdb commands separated by semicolons, for example `frame 0; info locals`.",
            "nullable": True,
        },
        "timeout": {
            "type": "integer",
            "description": "Optional timeout in seconds (default 300, max 600).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        command: str,
        work_dir: str | None = None,
        gdb_commands: str | None = None,
        timeout: int | None = None,
    ) -> str:
        try:
            cwd = _resolve_work_dir(work_dir)
        except ValueError as e:
            return f"Error: {e}"

        timeout_sec = _normalize_timeout(timeout)
        ok, message = _ensure_system_packages(["gdb"])
        if not ok:
            return f"Error: {message}"

        target_command = _expand_runtime_command(command, cwd)
        # ASAN_OPTIONS=detect_leaks=0: LeakSanitizer ptraces the inferior at exit
        # to walk thread stacks, which conflicts with gdb's ptrace and aborts the
        # process before any user breakpoint can fire. Disabling leak detection
        # keeps ASan instrumentation otherwise intact.
        base_setup_commands = [
            "set pagination off",
            "set confirm off",
            "set breakpoint pending on",
            # Silence DWARF symbol-loading complaints (gdb 10.2 still warns on
            # some pre-DWARF-5 shared libs even when it can read them).
            "set complaints 0",
            "set follow-exec-mode new",
            "set environment ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:handle_segv=0",
            "set environment LSAN_OPTIONS=detect_leaks=0",
            "handle SIGABRT stop print nopass",
            "handle SIGSEGV stop print nopass",
            "handle SIGBUS stop print nopass",
            "handle SIGILL stop print nopass",
            "handle SIGFPE stop print nopass",
            # Auto-catch sanitizer abort points so we still get a useful frame
            # even when the agent's own breakpoints don't resolve. Pending if
            # the symbols aren't present in the binary (no error).
            "break __asan_report_error",
            "break __sanitizer_print_stack_trace",
        ]
        user_setup_commands, run_command, user_post_run_commands = _split_gdb_commands(gdb_commands)
        default_post_run_commands = ["bt 30", "info registers"]

        if any(_is_gdb_breakpoint_command(cmd) for cmd in user_setup_commands):
            setup_probe_commands = [*base_setup_commands, *user_setup_commands, "info breakpoints"]
            try:
                setup_probe = subprocess.run(
                    _gdb_args(setup_probe_commands, target_command),
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return f"Timed out after {timeout_sec}s while setting GDB breakpoints"

            setup_output = "\n".join(part for part in [setup_probe.stdout, setup_probe.stderr] if part).strip()
            if _breakpoint_setup_failed(setup_output, user_setup_commands):
                summary = _summarize_pattern_output(
                    setup_output,
                    patterns=[
                        "function ",
                        "not defined",
                        "no source file named",
                        "no symbol table",
                        "breakpoint",
                        "catchpoint",
                        "watchpoint",
                    ],
                    context_lines=2,
                    max_lines=80,
                )
                return _truncate_output(
                    "GDB package status: "
                    f"{message}\nWorking directory: {cwd}\nTarget command: {target_command}\n\n"
                    "GDB breakpoint setup failed before target execution. The target was not run "
                    "because no requested breakpoint resolved or became pending.\n\n"
                    f"{summary}"
                )

        batch_commands = [
            *base_setup_commands,
            *user_setup_commands,
            run_command,
            *default_post_run_commands,
            *user_post_run_commands,
        ]

        gdb_args = _gdb_args(batch_commands, target_command)

        try:
            completed = subprocess.run(
                gdb_args,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout_sec}s"

        combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        summary = _summarize_pattern_output(
            combined,
            patterns=[
                "program received signal",
                "signal sig",
                "#0",
                "#1",
                "backtrace",
                "registers",
                "error:",
                "function ",
                "not defined",
                "no source file named",
                "no stack",
            ],
            context_lines=3,
            max_lines=140,
        )
        return _truncate_output(
            f"GDB package status: {message}\nWorking directory: {cwd}\nTarget command: {target_command}\n\n{summary}"
        )


class ValgrindTool(Tool):
    name = "valgrind"
    description = (
        "Run a command under Valgrind Memcheck and return a concise summary of memory errors and leak findings. "
        "Useful for debugging SEC-bench PoCs such as `secb repro`."
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "Command to run, for example `secb repro`.",
        },
        "work_dir": {
            "type": "string",
            "description": "Optional working directory. If omitted, the SEC-bench source directory is auto-detected.",
            "nullable": True,
        },
        "valgrind_options": {
            "type": "string",
            "description": "Optional Valgrind options. Defaults to a concise Memcheck setup.",
            "nullable": True,
        },
        "timeout": {
            "type": "integer",
            "description": "Optional timeout in seconds (default 300, max 600).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        command: str,
        work_dir: str | None = None,
        valgrind_options: str | None = None,
        timeout: int | None = None,
    ) -> str:
        try:
            cwd = _resolve_work_dir(work_dir)
        except ValueError as e:
            return f"Error: {e}"

        timeout_sec = _normalize_timeout(timeout)
        ok, message = _ensure_system_packages(["valgrind"])
        if not ok:
            return f"Error: {message}"

        target_command = _expand_runtime_command(command, cwd)
        options = valgrind_options or "--tool=memcheck --quiet --leak-check=summary --show-leak-kinds=definite --num-callers=16"
        wrapped_command = f"valgrind {options} bash -lc 'exec {target_command}'"

        try:
            completed = _run_shell_command(wrapped_command, cwd=cwd, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout_sec}s"

        combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part).strip()
        summary = _summarize_pattern_output(
            combined,
            patterns=[
                "error summary",
                "invalid read",
                "invalid write",
                "invalid free",
                "definitely lost",
                "possibly lost",
                "uninitialised",
                "conditional jump",
            ],
            context_lines=3,
            max_lines=160,
        )
        return _truncate_output(
            f"Valgrind package status: {message}\nWorking directory: {cwd}\nTarget command: {target_command}\n\n{summary}"
        )


class UBSanTool(Tool):
    name = "ubsan"
    description = (
        "Rebuild and rerun a target with UndefinedBehaviorSanitizer flags, then return the relevant runtime errors. "
        "Useful for surfacing integer overflow, bounds, null dereference, and related UB beyond the default build."
    )
    inputs = {
        "build_command": {
            "type": "string",
            "description": "Build command to run with UBSan flags. Defaults to `secb build`.",
            "nullable": True,
        },
        "run_command": {
            "type": "string",
            "description": "Command to run after rebuilding. Defaults to `secb repro`.",
            "nullable": True,
        },
        "work_dir": {
            "type": "string",
            "description": "Optional working directory. If omitted, the SEC-bench source directory is auto-detected.",
            "nullable": True,
        },
        "sanitizers": {
            "type": "string",
            "description": "Comma-separated UBSan checks. Defaults to `undefined,bounds,integer,null,shift`.",
            "nullable": True,
        },
        "timeout": {
            "type": "integer",
            "description": "Optional timeout in seconds per command (default 300, max 600).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(
        self,
        build_command: str | None = None,
        run_command: str | None = None,
        work_dir: str | None = None,
        sanitizers: str | None = None,
        timeout: int | None = None,
    ) -> str:
        import os

        try:
            cwd = _resolve_work_dir(work_dir)
        except ValueError as e:
            return f"Error: {e}"

        timeout_sec = _normalize_timeout(timeout)
        build_cmd = (build_command or "secb build").strip()
        run_cmd = (run_command or "secb repro").strip()
        sanitizer_list = (sanitizers or "undefined,bounds,integer,null,shift").strip()

        env = os.environ.copy()
        cflags = _strip_existing_sanitizers(env.get("CFLAGS", ""))
        cxxflags = _strip_existing_sanitizers(env.get("CXXFLAGS", ""))
        ldflags = _strip_existing_sanitizers(env.get("LDFLAGS", ""))

        ubsan_flags = f"-fsanitize={sanitizer_list} -fno-sanitize-recover=all -fno-omit-frame-pointer -g"
        env["CFLAGS"] = " ".join(part for part in [cflags, ubsan_flags] if part).strip()
        env["CXXFLAGS"] = " ".join(part for part in [cxxflags, ubsan_flags] if part).strip()
        env["LDFLAGS"] = " ".join(part for part in [ldflags, f"-fsanitize={sanitizer_list}"] if part).strip()
        env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"

        try:
            build_result = _run_shell_command(build_cmd, cwd=cwd, timeout=timeout_sec, env=env)
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout_sec}s while running build command"

        if build_result.returncode != 0:
            build_output = "\n".join(part for part in [build_result.stdout, build_result.stderr] if part).strip()
            return _truncate_output(
                f"UBSan build failed.\nWorking directory: {cwd}\nBuild command: {build_cmd}\n\n{build_output}"
            )

        expanded_run_cmd = _expand_runtime_command(run_cmd, cwd)
        try:
            run_result = _run_shell_command(expanded_run_cmd, cwd=cwd, timeout=timeout_sec, env=env)
        except subprocess.TimeoutExpired:
            return f"Timed out after {timeout_sec}s while running repro command"

        combined = "\n".join(part for part in [run_result.stdout, run_result.stderr] if part).strip()
        summary = _summarize_pattern_output(
            combined,
            patterns=[
                "runtime error:",
                "summary:",
                "undefinedbehavior",
                "undefined-behavior",
                "ubsan",
                "signed integer overflow",
                "out of bounds",
                "null pointer",
            ],
            context_lines=3,
            max_lines=160,
        )
        return _truncate_output(
            "UBSan rebuild completed.\n"
            f"Working directory: {cwd}\n"
            f"Build command: {build_cmd}\n"
            f"Run command: {expanded_run_cmd}\n"
            f"Sanitizers: {sanitizer_list}\n\n"
            f"{summary}"
        )


TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        PythonInterpreterTool,
        DuckDuckGoSearchTool,
        VisitWebpageTool,
        CmdTool,
        GDBTool,
        ValgrindTool,
        UBSanTool,
    ]
}

TOOL_MAPPING[SanitizerParserTool.name] = SanitizerParserTool

__all__ = [
    "ApiWebSearchTool",
    "PythonInterpreterTool",
    "FinalAnswerTool",
    "UserInputTool",
    "WebSearchTool",
    "DuckDuckGoSearchTool",
    "GoogleSearchTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "SpeechToTextTool",
    "CmdTool",
    "GDBTool",
    "ValgrindTool",
    "UBSanTool",
]
