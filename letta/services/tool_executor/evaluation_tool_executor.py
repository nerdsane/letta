import json
from typing import Any, Dict, List, Optional

from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.agent import AgentState
from letta.schemas.llm_config import LLMConfig
from letta.schemas.sandbox_config import SandboxConfig
from letta.schemas.tool import Tool
from letta.schemas.tool_execution_result import ToolExecutionResult
from letta.schemas.user import User
from letta.services.tool_executor.tool_executor_base import ToolExecutor
from letta.utils import get_friendly_error_msg

logger = get_logger(__name__)


class LettaEvaluationToolExecutor(ToolExecutor):
    """Executor for Letta evaluation tools using LLM-as-judge."""

    @trace_method
    async def execute(
        self,
        function_name: str,
        function_args: dict,
        tool: Tool,
        actor: User,
        agent_state: Optional[AgentState] = None,
        sandbox_config: Optional[SandboxConfig] = None,
        sandbox_env_vars: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        function_map = {
            "assess_output_quality": self.assess_output_quality,
            "check_logical_consistency": self.check_logical_consistency,
            "compare_versions": self.compare_versions,
            "analyze_information_gain": self.analyze_information_gain,
        }

        if function_name not in function_map:
            raise ValueError(f"Unknown evaluation function: {function_name}")

        # Execute the appropriate function
        function_args_copy = function_args.copy()
        try:
            function_response = await function_map[function_name](agent_state, actor, **function_args_copy)
            return ToolExecutionResult(
                status="success",
                func_return=function_response,
                agent_state=agent_state,
            )
        except Exception as e:
            logger.error(f"Evaluation tool {function_name} failed: {str(e)}")
            return ToolExecutionResult(
                status="error",
                func_return=str(e),
                agent_state=agent_state,
                stderr=[get_friendly_error_msg(function_name=function_name, exception_name=type(e).__name__, exception_message=str(e))],
            )

    async def _call_llm_for_evaluation(
        self,
        agent_state: AgentState,
        actor: User,
        system_prompt: str,
        user_message: str,
        model: str = "claude-sonnet-4-5-20250929",
    ) -> str:
        """
        Make a simple LLM call for evaluation purposes.
        Uses Claude Sonnet 4.5 by default for fast, accurate evaluations.
        """
        from letta.llm_api.llm_client import LLMClient
        from letta.schemas.enums import ProviderType

        # Create LLM config for evaluation (using Sonnet 4.5 for speed/accuracy)
        llm_config = LLMConfig(
            model=model,
            model_endpoint_type=ProviderType.anthropic,
            model_endpoint="https://api.anthropic.com/v1",
        )

        # Create Anthropic client
        client = LLMClient.create(
            provider_type=ProviderType.anthropic,
            actor=actor,
        )

        if not client:
            raise ValueError("Failed to create LLM client for evaluation")

        # Prepare request
        request_data = {
            "model": model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        # Make async call
        response = await client.request_async(request_data, llm_config)

        # Extract text from response
        if "content" in response and len(response["content"]) > 0:
            return response["content"][0].get("text", "")
        else:
            raise ValueError("No response content from LLM")

    async def assess_output_quality(
        self,
        agent_state: AgentState,
        actor: User,
        content: str,
        rubric: str,
        content_type: str = "text",
    ) -> str:
        """
        Evaluate quality of content against a custom rubric using LLM-as-judge.
        """
        system_prompt = """You are an expert evaluator assessing content quality against a rubric.

Your task:
1. Read the content and rubric carefully
2. Evaluate the content against each criterion in the rubric
3. Provide a structured evaluation

Return your evaluation as valid JSON with this exact structure:
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<your detailed reasoning>",
    "strengths": ["<strength 1>", "<strength 2>", ...],
    "improvements": ["<improvement 1>", "<improvement 2>", ...],
    "meets_criteria": <true or false>
}"""

        user_message = f"""Content Type: {content_type}

Rubric:
{rubric}

Content to Evaluate:
{content}

Please evaluate the content against the rubric and return your assessment as JSON."""

        response_text = await self._call_llm_for_evaluation(agent_state, actor, system_prompt, user_message)

        # Parse and validate JSON
        try:
            result = json.loads(response_text)
            # Ensure required fields exist
            required_fields = ["score", "reasoning", "strengths", "improvements", "meets_criteria"]
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in evaluation response")
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse evaluation response as JSON: {response_text}")
            # Return a fallback structure
            return json.dumps(
                {
                    "score": 0.5,
                    "reasoning": "Failed to parse evaluation response",
                    "strengths": [],
                    "improvements": ["Evaluation system error"],
                    "meets_criteria": False,
                    "raw_response": response_text,
                },
                indent=2,
            )

    async def check_logical_consistency(
        self,
        agent_state: AgentState,
        actor: User,
        content: str,
        rules: Optional[List[str]] = None,
        format: str = "text",
    ) -> str:
        """
        Check content for logical contradictions.
        """
        system_prompt = """You are an expert in logical analysis and consistency checking.

Your task:
1. Analyze the content for logical contradictions
2. If explicit rules are provided, check the content against those rules
3. Identify any conflicts, contradictions, or inconsistencies

Return your analysis as valid JSON with this exact structure:
{
    "consistent": <true or false>,
    "contradictions": [
        {
            "elements": ["<element 1>", "<element 2>"],
            "description": "<why they contradict>",
            "severity": "<minor or major>"
        }
    ],
    "checks_performed": <integer count of checks>
}"""

        rules_text = ""
        if rules:
            rules_text = f"\n\nExplicit Rules to Check:\n" + "\n".join(f"- {rule}" for rule in rules)

        user_message = f"""Content Format: {format}

{rules_text}

Content to Check:
{content}

Please analyze this content for logical contradictions and return your findings as JSON."""

        response_text = await self._call_llm_for_evaluation(agent_state, actor, system_prompt, user_message)

        # Parse and validate JSON
        try:
            result = json.loads(response_text)
            required_fields = ["consistent", "contradictions", "checks_performed"]
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in consistency check response")
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse consistency check response as JSON: {response_text}")
            return json.dumps(
                {
                    "consistent": True,
                    "contradictions": [],
                    "checks_performed": 0,
                    "error": "Failed to parse response",
                    "raw_response": response_text,
                },
                indent=2,
            )

    async def compare_versions(
        self,
        agent_state: AgentState,
        actor: User,
        current: str,
        previous: str,
        comparison_criteria: str = "quality, novelty, accuracy",
    ) -> str:
        """
        Compare two versions of content to measure improvement.
        """
        system_prompt = """You are an expert at comparing versions of content and assessing improvement.

Your task:
1. Compare the current version against the previous version
2. Identify what changed
3. Evaluate whether the changes are improvements or regressions
4. Focus on the specified comparison criteria

Return your comparison as valid JSON with this exact structure:
{
    "improved": <true or false>,
    "changes": ["<change 1>", "<change 2>", ...],
    "better_aspects": ["<what improved>", ...],
    "worse_aspects": ["<what regressed>", ...],
    "recommendation": "<keep_current or revert or iterate_more>"
}"""

        user_message = f"""Comparison Criteria: {comparison_criteria}

Previous Version:
{previous}

---

Current Version:
{current}

Please compare these versions and return your assessment as JSON."""

        response_text = await self._call_llm_for_evaluation(agent_state, actor, system_prompt, user_message)

        # Parse and validate JSON
        try:
            result = json.loads(response_text)
            required_fields = ["improved", "changes", "better_aspects", "worse_aspects", "recommendation"]
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in version comparison response")
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse version comparison response as JSON: {response_text}")
            return json.dumps(
                {
                    "improved": False,
                    "changes": [],
                    "better_aspects": [],
                    "worse_aspects": ["Evaluation error"],
                    "recommendation": "iterate_more",
                    "error": "Failed to parse response",
                    "raw_response": response_text,
                },
                indent=2,
            )

    async def analyze_information_gain(
        self,
        agent_state: AgentState,
        actor: User,
        after: str,
        before: str,
        metric: str = "novelty",
    ) -> str:
        """
        Measure what new information was added between versions.
        """
        system_prompt = """You are an expert at analyzing information content and measuring knowledge gain.

Your task:
1. Compare the "after" state against the "before" state
2. Identify genuinely new information, facts, or insights (not just reformulations)
3. Assess the significance of the new information based on the specified metric
4. Focus on substantive additions, not superficial changes

Return your analysis as valid JSON with this exact structure:
{
    "information_gain": <float between 0.0 and 1.0>,
    "new_facts": ["<new fact 1>", "<new fact 2>", ...],
    "insights": ["<new insight 1>", "<new insight 2>", ...],
    "significance": "<trivial or minor or moderate or major or breakthrough>"
}"""

        user_message = f"""Metric to Measure: {metric}

Before State:
{before}

---

After State:
{after}

Please analyze the information gain and return your assessment as JSON."""

        response_text = await self._call_llm_for_evaluation(agent_state, actor, system_prompt, user_message)

        # Parse and validate JSON
        try:
            result = json.loads(response_text)
            required_fields = ["information_gain", "new_facts", "insights", "significance"]
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in information gain response")
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse information gain response as JSON: {response_text}")
            return json.dumps(
                {
                    "information_gain": 0.0,
                    "new_facts": [],
                    "insights": [],
                    "significance": "trivial",
                    "error": "Failed to parse response",
                    "raw_response": response_text,
                },
                indent=2,
            )
