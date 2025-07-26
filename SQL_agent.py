from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase

from langflow.base.agents.agent import LCAgentComponent
from langflow.inputs.inputs import HandleInput, MessageTextInput

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langflow.base.models.anthropic_constants import ANTHROPIC_MODELS
from langflow.base.models.google_generative_ai_constants import GOOGLE_GENERATIVE_AI_MODELS
from langflow.base.models.model import LCModelComponent
from langflow.base.models.openai_constants import OPENAI_CHAT_MODEL_NAMES, OPENAI_REASONING_MODEL_NAMES
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs.inputs import BoolInput
from langflow.io import DropdownInput, MessageInput, MultilineInput, SecretStrInput, SliderInput
from langflow.schema.dotdict import dotdict
from langchain_core.tools import StructuredTool

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.agents.events import ExceptionWithMessageError
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS,
    MODEL_PROVIDERS_DICT,
    MODELS_METADATA,
)
from langflow.base.models.model_utils import get_model_name
from langflow.components.helpers.current_date import CurrentDateComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.custom.utils import update_component_build_config
from langflow.field_typing import Tool
from langflow.io import BoolInput, DropdownInput, IntInput, MultilineInput, Output
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message

class SQLAgentComponent(LCAgentComponent):
    display_name = "SQLAgent"
    description = "Construct an SQL agent from an LLM and tools."
    name = "SQLAgent"
    icon = "LangChain"
    inputs = [
        *LCAgentComponent._base_inputs,
        HandleInput(name="llm", display_name="Language Model", input_types=["LanguageModel"], required=True),
        MessageTextInput(name="database_uri", display_name="Database URI", required=True),
        HandleInput(
            name="extra_tools",
            display_name="Extra Tools",
            input_types=["Tool"],
            is_list=True,
            advanced=True,
        ),
    ]

    def build_agent(self) -> AgentExecutor:
        db = SQLDatabase.from_uri(self.database_uri)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        agent_args = self.get_agent_kwargs()
        agent_args["max_iterations"] = agent_args["agent_executor_kwargs"]["max_iterations"]
        del agent_args["agent_executor_kwargs"]["max_iterations"]
        return create_sql_agent(llm=self.llm, toolkit=toolkit, extra_tools=self.extra_tools or [], **agent_args)



class LanguageModelComponent(LCModelComponent):
    display_name = "Language Model"
    description = "Runs a language model given a specified provider."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    priority = 0  # Set priority to 0 to make it appear first

    inputs = [
        DropdownInput(
            name="provider",
            display_name="Model Provider",
            options=["OpenAI", "Anthropic", "Google"],
            value="OpenAI",
            info="Select the model provider",
            real_time_refresh=True,
            options_metadata=[{"icon": "OpenAI"}, {"icon": "Anthropic"}, {"icon": "GoogleGenerativeAI"}],
        ),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            options=OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_CHAT_MODEL_NAMES[0],
            info="Select the model to use",
            real_time_refresh=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Model Provider API key",
            required=False,
            show=True,
            real_time_refresh=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Input",
            info="The input text to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        provider = self.provider
        model_name = self.model_name
        temperature = self.temperature
        stream = self.stream

        if provider == "OpenAI":
            if not self.api_key:
                msg = "OpenAI API key is required when using OpenAI provider"
                raise ValueError(msg)

            if model_name in OPENAI_REASONING_MODEL_NAMES:
                # reasoning models do not support temperature (yet)
                temperature = None

            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                streaming=stream,
                openai_api_key=self.api_key,
            )
        if provider == "Anthropic":
            if not self.api_key:
                msg = "Anthropic API key is required when using Anthropic provider"
                raise ValueError(msg)
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                anthropic_api_key=self.api_key,
            )
        if provider == "Google":
            if not self.api_key:
                msg = "Google API key is required when using Google provider"
                raise ValueError(msg)
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                streaming=stream,
                google_api_key=self.api_key,
            )
        msg = f"Unknown provider: {provider}"
        raise ValueError(msg)

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name == "provider":
            if field_value == "OpenAI":
                build_config["model_name"]["options"] = OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES
                build_config["model_name"]["value"] = OPENAI_CHAT_MODEL_NAMES[0]
                build_config["api_key"]["display_name"] = "OpenAI API Key"
            elif field_value == "Anthropic":
                build_config["model_name"]["options"] = ANTHROPIC_MODELS
                build_config["model_name"]["value"] = ANTHROPIC_MODELS[0]
                build_config["api_key"]["display_name"] = "Anthropic API Key"
            elif field_value == "Google":
                build_config["model_name"]["options"] = GOOGLE_GENERATIVE_AI_MODELS
                build_config["model_name"]["value"] = GOOGLE_GENERATIVE_AI_MODELS[0]
                build_config["api_key"]["display_name"] = "Google API Key"
        elif field_name == "model_name" and field_value.startswith("o1") and self.provider == "OpenAI":
            # Hide system_message for o1 models - currently unsupported
            if "system_message" in build_config:
                build_config["system_message"]["show"] = False
        elif field_name == "model_name" and not field_value.startswith("o1") and "system_message" in build_config:
            build_config["system_message"]["show"] = True
        return build_config
def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=[],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST] + [{"icon": "brain"}],
        ),
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        *LCToolsAgentComponent._base_inputs,
        # removed memory inputs from agent component
        # *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [Output(name="response", display_name="Response", method="message_response")]

    async def message_response(self) -> Message:
        try:
            # Get LLM model and validate
            llm_model, display_name = self.get_llm()
            if llm_model is None:
                msg = "No language model selected. Please choose a model to proceed."
                raise ValueError(msg)
            self.model_name = get_model_name(llm_model, display_name=display_name)

            # Get memory data
            self.chat_history = await self.get_memory_data()
            if isinstance(self.chat_history, Message):
                self.chat_history = [self.chat_history]

            # Add current date tool if enabled
            if self.add_current_date_tool:
                if not isinstance(self.tools, list):  # type: ignore[has-type]
                    self.tools = []
                current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
                if not isinstance(current_date_tool, StructuredTool):
                    msg = "CurrentDateComponent must be converted to a StructuredTool"
                    raise TypeError(msg)
                self.tools.append(current_date_tool)
            # note the tools are not required to run the agent, hence the validation removed.

            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            agent = self.create_agent_runnable()
            return await self.run_agent(agent)

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            logger.error(f"ExceptionWithMessageError occurred: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e!s}")
            raise

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(session_id=self.graph.session_id, order="Ascending", n_messages=self.n_messages)
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    def get_llm(self):
        if not isinstance(self.agent_llm, str):
            return self.agent_llm, None

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                msg = f"Invalid model provider: {self.agent_llm}"
                raise ValueError(msg)

            component_class = provider_info.get("component_class")
            display_name = component_class.display_name
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")

            return self._build_llm_model(component_class, inputs, prefix), display_name

        except Exception as e:
            logger.error(f"Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        model_kwargs = {}
        for input_ in inputs:
            if hasattr(self, f"{prefix}{input_.name}"):
                model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {input_.name: getattr(self, f"{prefix}{input_.name}") for input_ in inputs}

            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        # Iterate over all providers in the MODEL_PROVIDERS_DICT
        # Existing logic for updating build_config
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call the component class's update_build_config method
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]

                # Delete fields from other providers
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                # Add provider-specific fields
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                # Reset input types for agent_llm
                build_config["agent_llm"]["input_types"] = []
            elif field_value == "Custom":
                # Delete all provider fields
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                # Update with custom component
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    options=[*sorted(MODEL_PROVIDERS), "Custom"],
                    value="Custom",
                    real_time_refresh=True,
                    input_types=["LanguageModel"],
                    options_metadata=[MODELS_METADATA[key] for key in sorted(MODELS_METADATA.keys())]
                    + [{"icon": "brain"}],
                )
                build_config.update({"agent_llm": custom_component.to_dict()})
            # Update input types for all fields
            build_config = self.update_input_types(build_config)

            # Validate required keys
            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        if (
            isinstance(self.agent_llm, str)
            and self.agent_llm in MODEL_PROVIDERS_DICT
            and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call each component class's update_build_config method
                    # remove the prefix from the field_name
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name = field_name.replace(prefix, "")
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent", tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools

