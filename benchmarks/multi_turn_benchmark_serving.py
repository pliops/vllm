from typing import Optional, Tuple, List, Dict, Any, Union, NamedTuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import Counter, deque
import os
import json
import random
import time
from datetime import datetime
import logging
import argparse
import asyncio
from http import HTTPStatus
import multiprocessing as mp
import threading
from queue import Queue
import getpass
from statistics import mean
import numpy as np  # type: ignore
import aiohttp  # type: ignore
import pandas as pd  # type: ignore
from prometheus_client.parser import text_string_to_metric_families  # type: ignore
import pynvml  # type: ignore
from transformers import AutoTokenizer  # type: ignore


COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_PURPLE = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_GRAY = "\033[90m"
COLOR_YELLOW = "\033[93m"

BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"

TEXT_BOLD = "\033[1m"
TEXT_ITALIC = "\033[3m"
TEXT_UNDERLINE = "\033[4m"
TEXT_DIM = "\033[2m"

COLOR_RESET = "\033[0m"

TEXT_SEPARATOR = "-" * 100

NUM_TOKENS_FROM_DATASET = 0
TERM_SIGNAL = None


# Conversation ID is a string (e.g: "UzTK34D")
ConvId = str

# A list of dicts (dicts with keys "id" and "messages")
ShareGptConversations = List[Dict[str, Any]]

# A list of dicts (dicts with keys "role" and "content")
MessagesList = List[Dict[str, str]]

# Map conversation ID to conversation messages
ConversationsMap = Dict[ConvId, MessagesList]


class ConversationSampling(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


class Distribution(ABC):
    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        pass


class UniformDistribution(Distribution):
    def __init__(self,
                 min_val: Union[int, float],
                 max_val: Union[int, float],
                 is_integer: bool = True) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.is_integer = is_integer

    def sample(self, size: int = 1) -> np.ndarray:
        if self.is_integer:
            return np.random.randint(int(self.min_val), int(self.max_val + 1), size=size)
        else:
            return np.random.uniform(self.min_val, self.max_val, size=size)

    def __repr__(self) -> str:
        return f"UniformDistribution[{self.min_val}, {self.max_val}]"


class ConstantDistribution(Distribution):
    def __init__(self, value: Union[int, float]) -> None:
        self.value = value
        self.max_val = value

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(shape=size, fill_value=self.value)

    def __repr__(self) -> str:
        return f"Constant[{self.value}]"


class ZipfDistribution(Distribution):
    def __init__(self, alpha: float,
                 max_val: Optional[int] = None) -> None:
        self.alpha = alpha
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.zipf(self.alpha, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"ZipfDistribution[{self.alpha}]"


class PoissonDistribution(Distribution):
    def __init__(self, alpha: float,
                 max_val: Optional[int] = None) -> None:
        self.alpha = alpha
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.poisson(self.alpha, size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)
        return samples

    def __repr__(self) -> str:
        return f"PoissonDistribution[{self.alpha}]"


class LognormalDistribution(Distribution):
    def __init__(self, mean: float, sigma: float,
                 max_val: Optional[int] = None) -> None:
        self.mean = mean
        self.sigma = sigma
        self.max_val = max_val

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.lognormal(mean=self.mean,
                                      sigma=self.sigma,
                                      size=size)
        if self.max_val:
            samples = np.minimum(samples, self.max_val)

        return np.round(samples).astype(int)

    def __repr__(self) -> str:
        return f"LognormalDistribution[{self.mean}, {self.sigma}]"


class GenConvArgs(NamedTuple):
    num_conversations: int
    text_files: List[str]
    input_num_turns: Distribution
    input_common_prefix_num_tokens: Distribution
    input_prefix_num_tokens: Distribution
    input_num_tokens: Distribution
    output_num_tokens: Distribution
    print_stats: bool


class ClientArgs(NamedTuple):
    seed: int
    max_num_requests: int
    skip_first_turn: bool
    max_turns: Optional[int]
    max_active_conversations: int
    prefix_data_file: Optional[str]
    prefix_num_words: int
    verbose: bool
    print_content: bool
    verify_output: bool
    conversation_sampling: ConversationSampling
    lambda_param: float
    sla_ttft: int
    sla_tpot: int


class RequestArgs(NamedTuple):
    chat_url: str
    model: str
    temperature: float
    stream: bool
    limit_min_tokens: Optional[int]
    limit_max_tokens: Optional[int]


class BenchmarkArgs(NamedTuple):
    url: str
    num_clients: int
    early_stop: bool
    metrics_interval_sec: int


class ServerResponse(NamedTuple):
    valid: bool
    ttft_ms: float  # time to first chunk
    tpot_ms: float  # time per output chunk (one or more tokens)
    latency_ms: float
    start_time_ms: float
    first_chunk: str  # first chunk of the content
    content: str     # includes the first_chunk
    num_chunks: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}"


class RequestStats(NamedTuple):
    ttft_ms: float
    tpot_ms: float
    latency_ms: float
    start_time_ms: float
    input_num_turns: int
    input_num_tokens: int
    output_num_tokens: int
    output_num_chunks: int
    output_num_first_chunk_tokens: int
    approx_cached_percent: float
    conversation_id: str
    client_id: int

    def __str__(self) -> str:
        return f"ttft_ms {self.ttft_ms:.2f}, tpot_ms {self.tpot_ms:.2f}, latency_ms {self.latency_ms:.2f}, input_num_tokens {self.input_num_tokens}, " \
               f"output_num_tokens {self.output_num_tokens} ({self.output_num_chunks} chunks, {self.output_num_first_chunk_tokens} tokens in first chunk), " \
               f"approx_cached_percent {self.approx_cached_percent:.2f}%"


class MetricStats:
    def __init__(self) -> None:
        self.min: Optional[float] = None
        self.max: Optional[float] = None
        self.avg: Optional[float] = None
        self.sum = 0.0
        self.count = 0

    def update(self, value: float) -> None:
        if self.min is None:
            self.min = value
        else:
            self.min = min(self.min, value)

        if self.max is None:
            self.max = value
        else:
            self.max = max(self.max, value)

        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f}, min: {self.min:>10.3f}, max: {self.max:>10.3f}"


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.window = np.zeros(window_size)
        self.index = 0
        self.sum = 0.0
        self.count = 0
        self.avg: Optional[float] = None

    def update(self, new_value: float) -> None:
        if self.count < self.window_size:
            # Filling up the window
            self.sum += new_value
            self.window[self.count] = new_value
            self.count += 1
        else:
            # Window is full, start replacing old values
            old_value = self.window[self.index]
            self.sum = self.sum - old_value + new_value
            self.window[self.index] = new_value
            self.index = (self.index + 1) % self.window_size

        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        if self.count == 0:
            return "no data"
        return f"avg: {self.avg:>10.3f} ({self.count} samples)"


class DebugStats:
    def __init__(self, logger: logging.Logger, window_size: int) -> None:
        self.logger = logger
        self.metrics: Dict[str, Union[MovingAverage, MetricStats]] = {
            "moving_avg_ttft_ms": MovingAverage(window_size),
            "moving_avg_tpot_ms": MovingAverage(window_size),
            "ttft_ms": MetricStats(),
            "tpot_ms": MetricStats(),
            "latency_ms": MetricStats(),
            "input_num_turns": MetricStats(),
            "input_num_tokens": MetricStats(),
            "output_num_tokens": MetricStats()
        }

    def update(self, data: RequestStats) -> None:
        self.metrics["ttft_ms"].update(data.ttft_ms)
        self.metrics["moving_avg_ttft_ms"].update(data.ttft_ms),
        self.metrics["tpot_ms"].update(data.tpot_ms)
        self.metrics["moving_avg_tpot_ms"].update(data.tpot_ms),
        self.metrics["latency_ms"].update(data.latency_ms)
        self.metrics["input_num_turns"].update(data.input_num_turns)
        self.metrics["input_num_tokens"].update(data.input_num_tokens)
        self.metrics["output_num_tokens"].update(data.output_num_tokens)

    def print(self) -> None:
        self.logger.info("-" * 50)
        for k, v in self.metrics.items():
            self.logger.info(f"[{k:25}] {v}")
        self.logger.info("-" * 50)


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)


ServerMetricsSample = Dict[str, Any]


class APIPollingThread(threading.Thread):
    def __init__(self, url: str, interval_sec: int):
        super().__init__()
        self.url = url
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self.data_queue: Queue = Queue()

    async def parse_data(self, metrics_data: str) -> ServerMetricsSample:
        filter_metrics = {
            "vllm:gpu_cache_usage_perc",
            "vllm:gpu_prefix_cache_hit_rate",
            "vllm:xdp_prefix_cache_hit_rate",

            "vllm:avg_prompt_throughput_toks_per_s",
            "vllm:avg_generation_throughput_toks_per_s",

            # "vllm:time_to_first_token_seconds_sum",
            # "vllm:time_to_first_token_seconds_bucket",

            # "vllm:time_per_output_token_seconds_sum",
            # "vllm:time_per_output_token_seconds_bucket",

            # "vllm:e2e_request_latency_seconds_sum",
            # "vllm:e2e_request_latency_seconds_bucket",

            # "vllm:request_prompt_tokens_sum",
            # "vllm:request_prompt_tokens_bucket",

            # "vllm:request_generation_tokens_sum",
            # "vllm:request_generation_tokens_bucket"
        }

        # Parse metrics and prepare ServerMetricsSample
        # data: ServerMetricsSample = []
        data: ServerMetricsSample = {}
        for family in text_string_to_metric_families(metrics_data):
            for sample in family.samples:
                metric_name = sample.name
                value = sample.value
                # labels = sample.labels

                if metric_name in filter_metrics:
                    # labels.pop("model_name", None)
                    # row = {"metric_name": metric_name, "value": value, **labels}
                    # data.append(row)
                    data[metric_name] = value

        return data

    async def fetch_data(self) -> Optional[ServerMetricsSample]:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                if HTTPStatus(response.status) == HTTPStatus.OK:
                    metrics_data = await response.text()
                    parsed_data = await self.parse_data(metrics_data)
                    logger.info(f"{COLOR_BLUE}{parsed_data}{COLOR_RESET}")
                    return parsed_data
                else:
                    return None

    async def polling_task(self) -> None:
        while not self.stop_event.is_set():
            try:
                data = await self.fetch_data()
                self.data_queue.put(data)
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
            if not self.stop_event.is_set():
                await asyncio.sleep(self.interval_sec)
        logger.info(f"{COLOR_BLUE}Server metrics monitor is exiting{COLOR_RESET}")

    def run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.polling_task())

    def stop(self) -> None:
        self.stop_event.set()

    def get_all_data(self) -> List[ServerMetricsSample]:
        all_data = []
        while not self.data_queue.empty():
            new_data: ServerMetricsSample = self.data_queue.get()
            all_data.append(new_data)
        return all_data


# Must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def nanosec_to_millisec(value: float) -> float:
    return value / 1000000.0


def nanosec_to_sec(value: float) -> float:
    return value / 1000000000.0


async def send_request(session: aiohttp.ClientSession,
                       messages: List[Dict[str, str]],
                       chat_url: str,
                       model: str,
                       temperature: float = 0.0,
                       stream: bool = True,
                       min_tokens: Optional[int] = None,
                       max_tokens: Optional[int] = None) -> ServerResponse:

    # List of strings that stop the generation when they are generated.
    # "stop" was disabled because it seems to increase TTFT (first few chunks are empty)
    # end_of_seq = ["</s>", "|</s>", "|end|", "</s1>", "</s2>"]
    # stop_assistant = ["</assistant>", "<|assistant|>", "</assistant|>", "|</assistant|", "</|assistant|>"]
    # stop_user = ["<|user|>", "</user>", "|<|user|", "|</user", "|user|>", "|user|"]
    # stop_special = ["|}", "|)", "|<|reserved_special_token_134|>|", "|<|reserved_special_token_213|>"]

    payload = {
        "model": model,
        "messages": messages,
        "seed": 0,
        "temperature": temperature,
        "top_p": 1.0,  # Sampling - consider all tokens.
        "top_k": -1,  # Sampling - consider all tokens.
        # "presence_penalty": -2.0, # Positive values increase the model's likelihood to talk about new topics.
        # "frequency_penalty": 2.0, # Positive values decrease the model's likelihood to repeat the same line verbatim.
        # "stop": [*end_of_seq, *stop_assistant, *stop_user, *stop_special]
    }

    if stream is True:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": False}

    if min_tokens is not None:
        payload["min_tokens"] = min_tokens

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Content-Type": "application/json"
    }

    # Set timeout based on SLA (TTFT 400ms, TPOT 40ms)
    response_max_tokens = 4096 * 4 + 512  # TODO: hard-coded, should be based on model
    sla_ttft_ms = 400
    sla_tpot_ms = 40
    timeout_sec = (sla_ttft_ms + response_max_tokens * sla_tpot_ms) / 1000.0
    timeout = aiohttp.ClientTimeout(total=timeout_sec)

    valid_response = True
    ttft: Optional[float] = None
    chunk_delay: List[int] = []
    latency: Optional[float] = None
    first_chunk = ""
    generated_text = ""

    start_time: int = time.perf_counter_ns()
    most_recent_timestamp: int = start_time

    async with session.post(url=chat_url,
                            json=payload,
                            headers=headers,
                            timeout=timeout) as response:

        http_status = HTTPStatus(response.status)
        if http_status == HTTPStatus.OK:
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                if chunk == "[DONE]":
                    # End of stream
                    latency = time.perf_counter_ns() - start_time
                elif stream is False:
                    data = json.loads(chunk)
                    message = data["choices"][0]["message"]
                    assert message["role"] == "assistant"
                    generated_text += message["content"]
                else:
                    timestamp: int = time.perf_counter_ns()
                    data = json.loads(chunk)

                    # Delta is the new content/text/data
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        if ttft is None:
                            # First token
                            first_token_time = time.perf_counter_ns()
                            ttft = first_token_time - start_time
                            first_chunk = delta["content"]
                        else:
                            # Decoding phase
                            chunk_delay.append(timestamp - most_recent_timestamp)

                        generated_text += delta["content"]

                    most_recent_timestamp = timestamp
        else:
            valid_response = False
            content = await response.text()
            logger.warning(f"{COLOR_YELLOW}Received HTTP status {http_status.value} ({http_status.phrase}): {content}{COLOR_RESET}")

    if latency is None:
        if valid_response:
            # Streaming is disabled, latency was not set
            latency = time.perf_counter_ns() - start_time
        else:
            latency = -1.0

    if ttft is None:
        # The response was a single chunk
        ttft = latency

    # Each chunk may include more than one token
    tpot: float = mean(chunk_delay) if len(chunk_delay) > 0 else 0.0
    num_chunks: int = len(chunk_delay)

    sr = ServerResponse(valid=valid_response,
                        ttft_ms=nanosec_to_millisec(ttft) if ttft > 0.0 else -1.0,
                        tpot_ms=nanosec_to_millisec(tpot),
                        latency_ms=nanosec_to_millisec(latency),
                        start_time_ms=nanosec_to_millisec(start_time),
                        first_chunk=first_chunk,
                        content=generated_text,
                        num_chunks=num_chunks)
    return sr


def get_short_string(input: str) -> str:
    n = 20
    if len(input) < 400:
        return input

    return f"{input[:n]}...{input[-n:]}"


def get_token_count(tokenizer: AutoTokenizer,
                    text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def get_messages_token_count(tokenizer: AutoTokenizer,
                             messages: List[Dict[str, str]]) -> int:
    token_count = 0
    for m in messages:
        token_count += get_token_count(tokenizer, m["content"])

    return token_count


async def send_turn(session: aiohttp.ClientSession,
                    client_id: int,
                    conv_id: str,
                    conversation_messages: MessagesList,
                    messages_to_use: int,
                    tokenizer: AutoTokenizer,
                    req_args: RequestArgs,
                    verbose: bool,
                    verify_output: bool) -> Optional[RequestStats]:
    assert messages_to_use > 0
    assert messages_to_use <= len(conversation_messages)

    messages = conversation_messages[:messages_to_use]

    # Index of the next message (the role should be "user")
    index = messages_to_use - 1

    # Verify that the message has only two keys, "role" and "content"
    assert len(messages[index].keys()) == 2
    assert "role" in messages[index] and "content" in messages[index]
    assert messages[index]["role"] == "user", \
    f"Failed on conversation ID {conv_id}, invalid message role (should be user)"

    if verbose is True:
        print(f"{COLOR_CYAN}Messages (conversation ID {conv_id}, {len(messages)} turns):{COLOR_RESET}", messages)

    min_tokens = req_args.limit_min_tokens
    max_tokens = req_args.limit_max_tokens

    if len(conversation_messages) > messages_to_use:
        if min_tokens == NUM_TOKENS_FROM_DATASET or \
           max_tokens == NUM_TOKENS_FROM_DATASET:
            # Compute number of tokens in the answer (from the input conversation)
            assistant_answer = conversation_messages[messages_to_use]
            answer_num_tokens = get_token_count(tokenizer, assistant_answer["content"])
            assert assistant_answer["role"] == "assistant"

        if min_tokens == NUM_TOKENS_FROM_DATASET:
            min_tokens = max(1, answer_num_tokens)

        if max_tokens == NUM_TOKENS_FROM_DATASET:
            max_tokens = max(1, answer_num_tokens)

    # Send the current conversation to LLM and get a response
    response: ServerResponse = await send_request(session,
                                                  messages,
                                                  req_args.chat_url,
                                                  req_args.model,
                                                  req_args.temperature,
                                                  req_args.stream,
                                                  min_tokens,
                                                  max_tokens)

    if response.valid is False:
        # Request failed
        return None

    # Compute number of tokens in input / output
    input_num_tokens = get_messages_token_count(tokenizer, messages)

    # Num tokens in the user's last question
    question_num_tokens = get_token_count(tokenizer, messages[index]["content"])

    # Num tokens in the history/context of the question
    assert input_num_tokens >= question_num_tokens
    history_num_tokens = input_num_tokens - question_num_tokens

    # Num tokens in the LLM's answer (first chunk and full answer)
    first_chunk_tokens = get_token_count(tokenizer, response.first_chunk)

    output_content = response.content
    output_num_tokens = get_token_count(tokenizer, output_content)

    # Prefix caching approximated cached percent
    approx_cached_percent = 100.0 * (history_num_tokens / input_num_tokens) if input_num_tokens > 0 else 0.0
    # print(f"output_num_tokens={output_num_tokens}")

    # Compute the correct TTFT and TPOT (based on tokens and not chunks).
    # Required because multiple output tokens may be bundled in a single chunk.
    if output_num_tokens > 1 and output_num_tokens > first_chunk_tokens:
        # More than one token and more than one chunk in the output
        # print(f"response.latency_ms={response.latency_ms}")

        decode_ms = response.latency_ms - response.ttft_ms
        decode_num_tokens = output_num_tokens - first_chunk_tokens
        # print(f"response.decode_ms={decode_ms}")
        # print(f"decode_num_tokens={decode_num_tokens}")
        tpot_ms = decode_ms / decode_num_tokens
        # print(f"tpot_ms={tpot_ms}")
    else:
        # In this case: output_num_tokens == first_chunk_tokens
        # Output was a single chunk (output_num_tokens > 1)
        # or even a single token (output_num_tokens == 1)
        # Assume average tpot is 34ms to estimate ttft
        tpot_ms = 34.0

    if first_chunk_tokens > 1:
        # First chunk had multiple tokens, adjust TTFT for a single token
        delta_ms = (first_chunk_tokens - 1) * tpot_ms
        ttft_ms = max(0.1, response.ttft_ms - delta_ms)
    else:
        # First chunk had only one token
        ttft_ms = response.ttft_ms

    rs = RequestStats(ttft_ms=ttft_ms,
                      tpot_ms=tpot_ms,
                      latency_ms=response.latency_ms,
                      start_time_ms=response.start_time_ms,
                      input_num_turns=len(messages),
                      input_num_tokens=input_num_tokens,
                      output_num_tokens=output_num_tokens,
                      output_num_chunks=response.num_chunks,
                      output_num_first_chunk_tokens=first_chunk_tokens,
                      approx_cached_percent=approx_cached_percent,
                      conversation_id=conv_id,
                      client_id=client_id)

    if verbose is True:
        print(f"\n{COLOR_YELLOW}Response ({output_num_tokens} tokens):{COLOR_RESET}", output_content)
        print(f"{COLOR_YELLOW}Response metrics: {rs}{COLOR_RESET}")
        print("-" * 70)

    # Save the LLM's answer (will be used as part of the context for the next user turn)
    answer_index = messages_to_use
    if len(conversation_messages) > answer_index:
        assert conversation_messages[answer_index]["role"] == "assistant", \
        f"Failed on conversation ID {conv_id}, invalid message role (should be assistant)"

        original_content = conversation_messages[answer_index]["content"]
        if verify_output is True:
            # Compare the new answer to the answer from the input file
            debug_info = f"LLM/dataset answers do not match ({conv_id}):\n'{get_short_string(output_content)}' (len: {len(output_content)}),\n'{get_short_string(original_content)}' (len: {len(original_content)})"

            assert original_content == output_content, debug_info
        # elif abs(len(output_content) - len(original_content)) >= len(original_content) / 50.0:
        #     # Debug info, not a fatal error
        #     logger.warning(f"Large size diff (50% or more) between LLM/dataset answers ({conv_id}): {len(output_content)}, {len(original_content)}")

        # Update the answer
        conversation_messages[answer_index]["content"] = output_content
    else:
        # A user prompt that has no answer, add the answer as a new message
        new_answer = {"role": "assistant", "content": output_content}
        conversation_messages.append(new_answer)

    return rs


async def poisson_sleep(lambda_param: float, verbose: bool = False) -> None:
    # Generate a random time interval from the Poisson distribution
    assert lambda_param > 0
    # interval = np.random.poisson(lambda_param)
    interval = np.random.exponential(1.0 / lambda_param)
    if verbose is True:
        logger.info(f"Sleeping for {interval:.3f} seconds...")
    await asyncio.sleep(interval)


def is_anomaly(result: RequestStats, stream: bool) -> bool:

    if result.ttft_ms < 1.0:
        # TTFT is very small
        return True

    if stream is True and result.tpot_ms < 1.0:
        # TPOT is very small
        return True

    if result.output_num_tokens < 1:
        # Empty answer
        return True

    if result.output_num_tokens == result.output_num_first_chunk_tokens:
        # Answer was a single chunk
        return True

    return False


async def client_main(args: ClientArgs,
                      req_args: RequestArgs,
                      client_id: int,
                      tokenizer: AutoTokenizer,
                      stop_event: mp.Event,  # type: ignore
                      task_queue: mp.Queue,
                      result_queue: mp.Queue,
                      conv_queue: mp.Queue) -> None:

    assert args.max_num_requests > 0, \
    "max_num_requests should be larger than zero"

    logger.info(f"{COLOR_CYAN}Started client {client_id}: max_num_requests={args.max_num_requests}, max_active_conversations={args.max_active_conversations}{COLOR_RESET}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Prefix text (will be part of the first user turn/prompt)
    list_of_words = []
    if args.prefix_data_file:
        with open(args.prefix_data_file, "r") as file:
            data = file.read()
            list_of_words = data.split()  # split text to words

        # Number of words in the file should be much larger than the prefix size (to avoid edge cases)
        min_word_count = 5 * args.prefix_num_words
        assert len(list_of_words) > min_word_count, \
        f"Input file {args.prefix_data_file} has only {len(list_of_words)} words, " \
        f"but {min_word_count} words are required for the prefix text"

    # Active conversations
    active_convs: ConversationsMap = {}
    conv_id_queue: deque = deque(maxlen=args.max_active_conversations)

    # Keep track of how many messages have been used for each conversation
    turns_count: Counter = Counter()
    num_successes = 0
    num_failures = 0

    # Track the timestamp (time.perf_counter()) of the last turn per conversation (only for debug)
    time_of_last_turn: Dict[ConvId, float] = {}

    # Flag that indicates that there are no new tasks (conversations) for the client
    task_queue_empty = False

    async with aiohttp.ClientSession() as session:
        # Print progress

        for i in range(args.max_num_requests):
            result = None

            if stop_event.is_set():  # type: ignore
                logger.info(f"{COLOR_YELLOW}Client {client_id} received a termination signal{COLOR_RESET}")
                break

            while len(active_convs) < args.max_active_conversations and \
                task_queue_empty is False:
                # Get a new conversation from the task queue
                conv_id, messages = task_queue.get()

                if conv_id is TERM_SIGNAL:
                    task_queue_empty = True
                    break

                if args.skip_first_turn:
                    # Skip the first turn (both user and assistant), relevant if warmup was enabled
                    # Default turns_count[conv_id] will be zero if conv_id was never inserted/updated in turns_count
                    turns_count[conv_id] += 2

                if turns_count[conv_id] < len(messages):
                    # Add new conversation
                    active_convs[conv_id] = messages
                    conv_id_queue.append(conv_id)

                    if args.verbose:
                        logger.info(f"{COLOR_GREEN}Client {client_id} will use conversation ID {conv_id} (active conversations {len(active_convs)}){COLOR_RESET}")

                elif args.verbose:
                    # No more messages (conversation finished during the warmup)
                    logger.info(f"{COLOR_YELLOW}Client {client_id} will not use conversation ID {conv_id} (all {len(messages)} messages already sent){COLOR_RESET}")

            if len(active_convs) == 0 or task_queue_empty:
                logger.info(f"{COLOR_YELLOW}Client {client_id} has no more conversations{COLOR_RESET}")
                break

            # Pick an active conversation for the next request
            if args.conversation_sampling == ConversationSampling.ROUND_ROBIN:
                conv_id = conv_id_queue.pop()
            else:
                # ConversationSampling.RANDOM
                active_ids = list(active_convs.keys())
                conv_id = random.choice(active_ids)

            messages = active_convs[conv_id]

            assert isinstance(messages, list)
            assert len(messages) > 0

            # Update the amount of messages to use
            turns_count[conv_id] += 1
            current_turn = turns_count[conv_id]

            assert current_turn < len(messages), \
            f"Turn number {current_turn} is invalid for conversation ID {conv_id} that has only {len(messages)} messages"

            if current_turn == 1 and len(list_of_words) > 0:
                # Use client_id in the start index calculation because the random seed of the clients is equal.
                # We want a different prefix for every conversation (shared prefix may be cached between conversations).
                base_word_offset = 128 * client_id
                max_word_count = len(list_of_words) - (args.prefix_num_words + base_word_offset)
                assert max_word_count > 0
                start = base_word_offset + random.randint(0, max_word_count)
                end = start + args.prefix_num_words

                # Create the prefix text string
                prefix_text = " ".join(list_of_words[start:end])

                # Add the text as prefix/context for the first prompt/turn (messages[0])
                messages[0]["content"] = prefix_text + ". " + messages[0]["content"]

            if args.verbose:
                curr_time_sec: float = time.perf_counter()
                time_since_last_turn: Union[str, float] = "N/A"
                if conv_id in time_of_last_turn:
                    time_since_last_turn = round(curr_time_sec - time_of_last_turn[conv_id], 3)
                logger.info(f"Client {client_id} using conversation ID {conv_id} (turn: {current_turn}, time since last turn [sec]: {time_since_last_turn})")
                time_of_last_turn[conv_id] = curr_time_sec

            success = True
            try:
                result = await send_turn(session,
                                         client_id,
                                         conv_id,
                                         messages,
                                         current_turn,
                                         tokenizer,
                                         req_args,
                                         args.print_content,
                                         args.verify_output)
                if result is not None:
                    result_queue.put(result)
                else:
                    # None means that the request failed,
                    # and should not be added to the statistics.
                    success = False
                    num_failures += 1

                    logger.warning(f"{COLOR_YELLOW}Client {client_id} - Request rejected during conversation ID {conv_id} (turn: {current_turn}){COLOR_RESET}")
                    logger.warning(f"Content length is {len(messages[current_turn - 1]['content'])} characters")

                    # Remove the conversation (should not be used again)
                    active_convs.pop(conv_id)

            except asyncio.exceptions.TimeoutError:
                num_failures += 1
                logger.exception(f"{COLOR_RED}Client {client_id} - Timeout during conversation ID {conv_id} (turn: {current_turn}){COLOR_RESET}")
                break  # Exit gracefully instead of raising an error

            except Exception:
                num_failures += 1
                logger.exception(f"{COLOR_RED}Client {client_id} - Exception during conversation ID {conv_id} (turn: {current_turn}){COLOR_RESET}")
                break  # Exit gracefully instead of raising an error

            if success:
                num_successes += 1

                # Update the turns counter to include the LLM response
                # The LLM response will be used as context for the next user turn
                turns_count[conv_id] += 1

                # Detect outliers/anomalies
                if result is not None:
                    if is_anomaly(result, req_args.stream):
                        logger.warning(f"{COLOR_YELLOW}Client {client_id} - Found anomaly (conversation ID {conv_id}, turn {current_turn}):\n{result}{COLOR_RESET}")

                    if result.ttft_ms > args.sla_ttft or (req_args.stream is True and result.tpot_ms > args.sla_tpot):
                        logger.warning(f"{COLOR_PURPLE}Client {client_id} - Found SLA violation (conversation ID {conv_id}, turn {current_turn}):\n{result}{COLOR_RESET}")

                max_turns = len(messages)
                if args.max_turns is not None:
                    # Limit the number of turns in the conversation
                    max_turns = min(args.max_turns, max_turns)

                if turns_count[conv_id] >= max_turns:
                    # Conversation has no more turns (no longer active)
                    # save the updated conversation (with the LLM server's answer)
                    conv_queue.put((conv_id, active_convs.pop(conv_id)))
                    if args.verbose:
                        logger.info(f"{COLOR_GREEN}Client {client_id} finished conversation ID {conv_id}{COLOR_RESET}")
                else:
                    # Conversation is not finished, insert it at the back of the queue
                    conv_id_queue.appendleft(conv_id)

            # Sleep between requests (if lambda is positive)
            if args.lambda_param > 0:
                await poisson_sleep(args.lambda_param, args.verbose)

    # Send indication that the client is done
    conv_queue.put((TERM_SIGNAL, TERM_SIGNAL))

    logger.info(f"{COLOR_CYAN}Client {client_id} is done ({num_successes=}, {num_failures=}){COLOR_RESET}")


def worker_function(client_id: int,
                    tokenizer: AutoTokenizer,
                    client_args: ClientArgs,
                    req_args: RequestArgs,
                    stop_event: mp.Event,  # type: ignore
                    task_queue: mp.Queue,
                    result_queue: mp.Queue,
                    conv_queue: mp.Queue) -> None:

    asyncio.run(client_main(client_args,
                            req_args,
                            client_id,
                            tokenizer,
                            stop_event,
                            task_queue,
                            result_queue,
                            conv_queue))


def get_client_config(args: argparse.Namespace,
                      input_conv: ConversationsMap) -> Tuple[ClientArgs, RequestArgs]:
    assert args.num_clients > 0, "Number of clients must be a positive number"

    # Max number of requests per client
    requests_per_client = int(args.num_requests / args.num_clients)
    assert requests_per_client > 0

    max_active_conversations = args.max_active_conversations
    if max_active_conversations is None:
        # Each client will have only one active conversation at a time
        max_active_conversations = args.num_clients

    assert max_active_conversations >= args.num_clients, \
    f"Max active conversations {max_active_conversations} must be equal or greater than the number of clients"
    assert max_active_conversations <= len(input_conv), \
    f"Max active conversations {max_active_conversations} must be equal or less than the total number of conversations"

    # Max number of active conversations per client
    max_active_conv_per_client = int(max_active_conversations / args.num_clients)

    # Skip the first user turn (as part of the warmup)
    skip_first_turn = args.warmup

    # Common arguments for all clients
    client_args = ClientArgs(seed=args.seed,
                             max_num_requests=requests_per_client,
                             skip_first_turn=skip_first_turn,
                             max_turns=args.max_turns,
                             max_active_conversations=max_active_conv_per_client,
                             prefix_data_file=args.prefix_data_file,
                             prefix_num_words=args.prefix_num_words,
                             verbose=args.verbose,
                             print_content=args.print_content,
                             verify_output=args.verify_output,
                             conversation_sampling=args.conversation_sampling,
                             lambda_param=args.lambda_param,
                             sla_ttft=args.sla_ttft,
                             sla_tpot=args.sla_tpot)

    req_args = RequestArgs(chat_url=f"{args.url}/v1/chat/completions",
                           model=args.model,
                           temperature=args.temperature,
                           stream=not args.disable_stream,
                           limit_min_tokens=args.limit_min_tokens,
                           limit_max_tokens=args.limit_max_tokens)

    return client_args, req_args


async def main_mp(client_args: ClientArgs,
                  req_args: RequestArgs,
                  bench_args: BenchmarkArgs,
                  tokenizer: AutoTokenizer,
                  input_conv: ConversationsMap) -> Tuple[ConversationsMap, List[RequestStats], Optional[List[ServerMetricsSample]]]:

    # An event that will trigger graceful termination of all the clients
    stop_event = mp.Event()

    # Queue for input conversations (from the input file/dataset)
    task_queue: mp.Queue = mp.Queue()

    # Queue for client measurements (TTFT, TPOT, etc. for each request)
    result_queue: mp.Queue = mp.Queue()

    # Queue for output conversations (with the LLM answers, sent by the server)
    conv_queue: mp.Queue = mp.Queue()
    output_conv: ConversationsMap = {}
    client_metrics: List[RequestStats] = []

    # Start all clients
    start_time = time.perf_counter_ns()
    logger.info(f"{COLOR_GREEN}Starting {bench_args.num_clients} clients{COLOR_RESET}")

    clients = []
    for client_id in range(bench_args.num_clients):
        client = mp.Process(name=f"client_{client_id}",
                            target=worker_function,
                            args=(client_id,
                                  tokenizer,
                                  client_args,
                                  req_args,
                                  stop_event,
                                  task_queue,
                                  result_queue,
                                  conv_queue))
        clients.append(client)
        client.start()

    # Submit all the input conversations as tasks for the clients
    for conv_id, messages in input_conv.items():
        task_queue.put((conv_id, messages))

    # Add termination signals for clients
    for _ in range(bench_args.num_clients):
        task_queue.put((TERM_SIGNAL, TERM_SIGNAL))

    if bench_args.metrics_interval_sec > 0:
        logger.info(f"{COLOR_BLUE}Starting the server metrics monitor thread{COLOR_RESET}")
        polling_thread = APIPollingThread(url=f"{bench_args.url}/metrics",
                                          interval_sec=bench_args.metrics_interval_sec)
        polling_thread.start()

    # Collect the updated conversations from all clients
    num_clients_finished = 0
    total_convs = len(input_conv)

    debug_stats = DebugStats(logger, min(15 * bench_args.num_clients, 500))

    while num_clients_finished < bench_args.num_clients:
        # Collect updated conversation
        conv_id, messages = conv_queue.get()

        # Collect results (measurements)
        while not result_queue.empty():
            new_data = result_queue.get()
            client_metrics.append(new_data)
            debug_stats.update(new_data)

        if conv_id is TERM_SIGNAL:
            num_clients_finished += 1
            logger.info(f"{COLOR_CYAN}{num_clients_finished} out of {bench_args.num_clients} clients finished{COLOR_RESET}")

            if bench_args.early_stop and not stop_event.is_set():
                # Once one client finished, stop all other clients.
                # there is no reason to continue the benchmark with fewer clients.
                logger.info(f"{COLOR_YELLOW}Sending termination signal to all clients{COLOR_RESET}")
                stop_event.set()
        else:
            output_conv[conv_id] = messages

            finished_convs = len(output_conv)
            percent = finished_convs / total_convs

            # Tuned to control the print rate (can be changed if required)
            print_cycle = max(3, int(bench_args.num_clients / 4))

            if finished_convs % print_cycle == 0:
                runtime_sec = nanosec_to_sec(time.perf_counter_ns() - start_time)
                logger.info(f"{COLOR_CYAN}Finished {finished_convs} out of {total_convs} conversations ({percent:.0%}), "
                            f"{num_clients_finished} out of {bench_args.num_clients} clients finished, collected {len(client_metrics)} measurements, runtime {runtime_sec:.3f} sec{COLOR_RESET}")

                rps: Union[str, float] = round(len(client_metrics) / runtime_sec, 3)
                if len(client_metrics) < (5 * bench_args.num_clients):
                    # Do not estimate the RPS if the number of samples is very low (threshold can be tuned if needed)
                    rps = "N/A"

                runtime_left_sec: Union[str, float] = round((runtime_sec / finished_convs) * (total_convs - finished_convs), 3)
                if percent < 0.05:
                    # If less than 5% of the conversations were not finished,
                    # the estimation will probably be very inaccurate (threshold can be tuned if needed).
                    runtime_left_sec = "N/A"

                logger.info(f"{COLOR_CYAN}Estimated req/sec {rps}, estimated runtime left {runtime_left_sec} sec{COLOR_RESET}")
                debug_stats.print()

    logger.info(f"{COLOR_CYAN}All {bench_args.num_clients} clients finished{COLOR_RESET}")

    # At this point all the clients finished,
    # collect results (TTFT, TPOT, etc.) from all the clients.
    # This needs to happens before calling join on the clients (result_queue should be emptied).
    while not result_queue.empty():
        client_metrics.append(result_queue.get())

    logger.info(f"Collected {len(client_metrics)} samples from all the clients")

    # Wait for all clients to finish
    for client in clients:
        logger.info(f"{COLOR_CYAN}Waiting for client {client.name} (is alive: {client.is_alive()}){COLOR_RESET}")

        client.join(timeout=120)

        if client.is_alive():
            logger.warning(f"{COLOR_YELLOW}Client {client.name} will be terminated{COLOR_RESET}")
            client.terminate()

        exitcode = client.exitcode
        if exitcode != 0:
            logger.error(f"{COLOR_RED}Client {client.name} exited with exit code {exitcode}{COLOR_RESET}")

    logger.info(f"All {bench_args.num_clients} clients exited (successfully finished {len(output_conv)} out of {total_convs} conversations)")

    server_metrics: Optional[List[ServerMetricsSample]] = None
    if bench_args.metrics_interval_sec > 0:
        logger.info(f"{COLOR_BLUE}Stopping the server metrics monitor thread{COLOR_RESET}")
        polling_thread.stop()
        polling_thread.join()

        # Collect the data from the thread
        server_metrics = polling_thread.get_all_data()
        logger.info(f"{COLOR_BLUE}Collected {len(server_metrics)} samples{COLOR_RESET}")

    return output_conv, client_metrics, server_metrics


def get_filename_with_timestamp(label: str, extension: str) -> str:
    time_now = datetime.now()
    timestamp = time_now.strftime("%d-%m-%Y_%H-%M-%S")
    filename = f"{label}__{timestamp}.{extension}"
    return filename


def server_metrics_report(server_metrics: List[ServerMetricsSample]) -> None:

    # Add "time" for use as x-axis in excel charts
    for index, item in enumerate(server_metrics):
        item["time"] = index

    df = pd.DataFrame(server_metrics)
    df.set_index("time", inplace=True)

    # Generate excel file with all the samples
    filename = get_filename_with_timestamp("server_metrics", "xlsx")
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:

        sheet_name = "Server metrics"
        df.to_excel(writer, sheet_name=sheet_name, index=True)

        # Create a chart object (to plot the metrics)
        workbook = writer.book
        chart = workbook.add_chart({"type": "line"})

        # Configure the series for the chart from the dataframe data
        num_rows = len(df)
        for col_num, column in enumerate(df.columns, start=1):
            chart.add_series({
                "name":       [sheet_name, 0, col_num],
                "categories": [sheet_name, 1, 0, num_rows, 0],  # Time column (index)
                "values":     [sheet_name, 1, col_num, num_rows, col_num],
            })

        chart.set_x_axis({"name": "Time"})
        chart.set_y_axis({"name": "Metric"})

        chart.set_title({"name": "Server metrics"})
        chart.set_legend({"position": "bottom"})

        chart.set_style(10)

        # Insert the chart into the worksheet
        worksheet = writer.sheets[sheet_name]
        worksheet.insert_chart("E2", chart)

    logger.info(f"{COLOR_GREEN}Server metrics exported to Excel file: {filename}{COLOR_RESET}")


def process_statistics(client_metrics: List[RequestStats],
                       warmup_percentages: List[float],
                       test_params: Dict,
                       verbose: bool,
                       gen_conv_args: Optional[GenConvArgs] = None,
                       excel_output: bool = False) -> None:
    if len(client_metrics) == 0:
        logger.info("No samples to process")
        return

    logger.info(f"Processing {len(client_metrics)} samples...")

    raw_data = pd.DataFrame(client_metrics)

    if verbose:
        # Calculate the time between user turns in each conversation (in a new column)
        raw_data = raw_data.sort_values(by=["conversation_id", "start_time_ms"])
        raw_data["time_between_user_turns_sec"] = raw_data.\
        groupby("conversation_id")["start_time_ms"].diff()

        # Convert milliseconds to seconds
        raw_data["time_between_user_turns_sec"] = raw_data["time_between_user_turns_sec"] / 1000.0

    # Final raw data should be sorted by time
    raw_data = raw_data.sort_values(by=["start_time_ms"])
    raw_data["end_time_ms"] = raw_data["start_time_ms"] + raw_data["latency_ms"]

    percentiles = [0.25, 0.5, 0.75, 0.9]

    # Add more percentiles if there are enough samples
    if len(raw_data) >= 100:
        percentiles.append(0.99)

    if len(raw_data) >= 1000:
        percentiles.append(0.999)

    if len(raw_data) >= 10000:
        percentiles.append(0.9999)

    # Set precision for numbers in the output text (the dataframes)
    pd.set_option("display.precision", 2)

    # Exclude parameters from RequestStats
    exclude = [
        "start_time_ms",
        "end_time_ms",
        "output_num_first_chunk_tokens",
        "approx_cached_percent",
        "conversation_id",
        "client_id"
    ]

    print(TEXT_SEPARATOR)
    print(f"{COLOR_YELLOW}Parameters:{COLOR_RESET}")
    for k, v in test_params.items():
        print(f"{k}={v}")

    # conversations generation parameters
    if gen_conv_args is not None:
        gen_params = {
            "text_files": ", ".join(gen_conv_args.text_files),
            "input_num_turns": str(gen_conv_args.input_num_turns),
            "input_common_prefix_num_tokens": str(gen_conv_args.input_common_prefix_num_tokens),
            "input_prefix_num_tokens": str(gen_conv_args.input_prefix_num_tokens),
            "input_num_tokens": str(gen_conv_args.input_num_tokens),
            "output_num_tokens": str(gen_conv_args.output_num_tokens)
        }

        print(f"{COLOR_YELLOW}Conversations Generation Parameters:{COLOR_RESET}")
        for k, v in gen_params.items():
            print(f"{k}={v}")

    print(TEXT_SEPARATOR)

    params_list = []
    df_list = []
    for percent in warmup_percentages:

        # Select samples from the end (tail) of the dataframe
        warmup_count = int(percent * len(raw_data))
        tail_count = len(raw_data) - warmup_count
        if tail_count == 0:
            # No reason to process if the count of samples is zero
            break

        df = raw_data.tail(tail_count)

        # Runtime is the diff between the end of the last request and the start of the first request
        runtime_sec = df["end_time_ms"].iloc[-1] - df["start_time_ms"].iloc[0]

        # Convert milliseconds to seconds
        runtime_sec = runtime_sec / 1000.0
        requests_per_sec = float(len(df)) / runtime_sec

        params = {
            "runtime_sec": runtime_sec,
            "requests_per_sec": requests_per_sec
        }

        # Generate a summary of relevant metrics (and drop irrelevant data)
        df = df.drop(columns=exclude).describe(percentiles=percentiles).transpose()

        # List for Excel file
        params_list.append(params)
        df_list.append(df)

        # Print the parameters and summary for the specific sample count
        print(f"{COLOR_YELLOW}Statistics summary (assuming {percent:.0%} warmup samples):{COLOR_RESET}")
        for k, v in params.items():
            if isinstance(v, float):
                print(f"{k} = {v:.3f}")
            else:
                print(f"{k} = {v}")
        print(TEXT_SEPARATOR)
        print(df)
        print(TEXT_SEPARATOR)

    if excel_output is True:
        prefix = f"statistics_{test_params['num_clients']}_clients"
        filename = get_filename_with_timestamp(prefix, "xlsx")

        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            startrow = 0
            test_params_df = pd.DataFrame([test_params])
            test_params_df.to_excel(writer, sheet_name="Summary",
                                    index=False, startrow=startrow)
            startrow += len(test_params_df) + 3

            if gen_conv_args is not None:
                gen_params_df = pd.DataFrame([gen_params])
                gen_params_df.to_excel(writer, sheet_name="Summary",
                                        index=False, startrow=(startrow - 1))
                startrow += len(gen_params_df) + 3

            for params, df_stats in zip(params_list, df_list):
                df_params = pd.DataFrame([params])
                df_params.to_excel(writer, sheet_name="Summary",
                                   index=False, startrow=startrow)
                startrow += len(df_params) + 2
                df_stats.to_excel(writer, sheet_name="Summary",
                                  index=True, startrow=startrow)
                startrow += len(df_stats) + 3

            raw_data.to_excel(writer, sheet_name="Raw data", index=False, startrow=0)

        logger.info(f"{COLOR_GREEN}Client metrics exported to Excel file: {filename}{COLOR_RESET}")


def conversations_list_to_dict(input_list: ShareGptConversations) -> ConversationsMap:
    conversations: ConversationsMap = {}

    for item in input_list:
        conv_id: str = item["id"]
        assert isinstance(conv_id, str)

        assert conv_id not in conversations, \
        f"Conversation ID {conv_id} found more than once in the input"

        messages: MessagesList = item["messages"]
        assert isinstance(messages, list), \
        f"Conversation messages should be a list (ID: {conv_id})"
        assert len(messages) > 0, f"Conversation with no messages (ID: {conv_id})"

        conversations[conv_id] = messages

    logger.info(f"Using {len(conversations)} unique conversations (IDs)")
    assert len(conversations) == len(input_list)

    # Print statistics about the selected conversations
    stats: List[Dict[str, Any]] = []
    for conv_data in conversations.values():
        stats.append({"num_turns": len(conv_data)})

    print(TEXT_SEPARATOR)
    print(f"{COLOR_YELLOW}Conversations statistics:{COLOR_RESET}")
    print(TEXT_SEPARATOR)
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999]
    conv_stats = pd.DataFrame(stats).describe(percentiles=percentiles)
    print(conv_stats.transpose())
    print(TEXT_SEPARATOR)

    return conversations


def conversations_dict_to_list(input_dict: ConversationsMap) -> ShareGptConversations:
    output: ShareGptConversations = []
    for conv_id, conv_data in input_dict.items():
        new_item = {"id": conv_id, "messages": conv_data}
        output.append(new_item)

    return output


async def get_server_info(url: str) -> None:
    logger.info(f"{COLOR_BLUE}Collecting information from server: {url}{COLOR_RESET}")
    async with aiohttp.ClientSession() as session:
        # Get server version (not mandatory, "version" endpoint may not exist)
        url_version = f"{url}/version"
        async with session.get(url_version) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{COLOR_BLUE}Server version: {text}{COLOR_RESET}")

        # Get available models
        url_models = f"{url}/v1/models"
        async with session.get(url_models) as response:
            if HTTPStatus(response.status) == HTTPStatus.OK:
                text = await response.text()
                logger.info(f"{COLOR_BLUE}Models:{COLOR_RESET}")
                models_data = json.loads(text)
                models_list = models_data["data"]
                for model in models_list:
                    model_id = model["id"]
                    max_model_len = model.get("max_model_len", "N/A")
                    logger.info(f"{COLOR_BLUE}\t{model_id=}, {max_model_len=}{COLOR_RESET}")
            else:
                logger.info(f"{COLOR_RED}Failed to get models{COLOR_RESET}")


def get_random_distribution(conf: dict,
                            section: str,
                            subsection: str,
                            optional: bool = False) -> Distribution:
    # section can be "prompt_input" or "prompt_output" (both required)
    conf = conf[section]

    if optional and subsection not in conf:
        # Optional subsection, if not found assume the value is always 0
        return ConstantDistribution(0)

    # subsection can be "num_turns", "num_tokens" or "prefix_num_tokens"
    assert subsection in conf, \
    f"Missing subsection {subsection} in section {section}"

    conf = conf[subsection]

    distribution = conf.get("distribution")
    assert distribution is not None, \
    f"Missing field 'distribution' in {section=} and {subsection=}"

    if distribution == "constant":
        assert "value" in conf, f"Missing field 'value' in {section=} and {subsection=}"
        return ConstantDistribution(conf["value"])

    elif distribution == "zipf":
        assert "alpha" in conf, f"Missing field 'alpha' in {section=} and {subsection=}"
        max_val = conf.get("max", None)
        return ZipfDistribution(conf["alpha"], max_val=max_val)

    elif distribution == "poisson":
        assert "alpha" in conf, f"Missing field 'alpha' in {section=} and {subsection=}"
        max_val = conf.get("max", None)
        return PoissonDistribution(conf["alpha"], max_val=max_val)

    elif distribution == "lognormal":
        assert "mean" in conf, f"Missing field 'mean' in {section=} and {subsection=}"
        assert "sigma" in conf, f"Missing field 'sigma' in {section=} and {subsection=}"
        max_val = conf.get("max", None)
        return LognormalDistribution(conf["mean"], conf["sigma"], max_val=max_val)

    elif distribution == "uniform":
        assert "min" in conf, f"Missing field 'min' in {section=} and {subsection=}"
        assert "max" in conf, f"Missing field 'max' in {section=} and {subsection=}"

        min_val = conf["min"]
        max_val = conf["max"]

        assert min_val > 0
        assert min_val <= max_val

        is_integer = isinstance(min_val, int) and isinstance(max_val, int)
        return UniformDistribution(min_val, max_val, is_integer)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def parse_input_json_file(conf: dict) -> GenConvArgs:

    # Validate the input file
    assert isinstance(conf, dict)
    required_fields = ["filetype", "num_conversations", "text_files",
                       "prompt_input", "prompt_output"]
    for field in required_fields:
        assert field in conf, \
        f"Missing field {field} in input {conf}"

    assert conf["filetype"] == "generate_conversations"

    assert conf["num_conversations"] > 0, \
    "num_conversations should be larger than zero"

    text_files = conf["text_files"]

    assert isinstance(text_files, list), \
    "Field 'text_files' should be a list"
    assert len(text_files) > 0, \
    "Field 'text_files' should be a list with at least one file"

    # Parse the parameters for the prompt input/output workload
    input_num_turns = get_random_distribution(conf, "prompt_input", "num_turns")
    input_num_tokens = get_random_distribution(conf, "prompt_input", "num_tokens")
    input_common_prefix_num_tokens = get_random_distribution(conf, "prompt_input",
                                                             "common_prefix_num_tokens",
                                                             optional=True)
    input_prefix_num_tokens = get_random_distribution(conf, "prompt_input", "prefix_num_tokens")
    output_num_tokens = get_random_distribution(conf, "prompt_output", "num_tokens")

    print_stats: bool = conf.get("print_stats", False)
    assert isinstance(print_stats, bool), \
    "Field 'print_stats' should be either 'true' or 'false'"

    args = GenConvArgs(num_conversations=conf["num_conversations"],
                       text_files=text_files,
                       input_num_turns=input_num_turns,
                       input_common_prefix_num_tokens=input_common_prefix_num_tokens,
                       input_prefix_num_tokens=input_prefix_num_tokens,
                       input_num_tokens=input_num_tokens,
                       output_num_tokens=output_num_tokens,
                       print_stats=print_stats)
    return args


def print_conv_stats(conversations: ConversationsMap,
                     tokenizer: AutoTokenizer) -> None:
    # Collect statistics
    conv_stats: List[Dict] = []
    req_stats: List[int] = []

    print("\nCollecting statistics...")
    for messages in conversations.values():
        # messages is a list of dicts
        user_tokens: List[int] = []
        assistant_tokens: List[int] = []
        request_tokens: List[int] = []

        req_tokens = 0
        for m in messages:
            content = m["content"]
            num_tokens = len(tokenizer(content).input_ids)

            if m["role"] == "user":
                user_tokens.append(num_tokens)
                # New user prompt including all chat history
                req_tokens += num_tokens
                request_tokens.append(req_tokens)

            elif m["role"] == "assistant":
                assistant_tokens.append(num_tokens)
                # Update assistant answer
                # (will be part of chat history for the next user prompt)
                req_tokens += num_tokens

        item_stats = {
            "conversation_turns": len(messages),
            "user_tokens": mean(user_tokens),
            "assistant_tokens": mean(assistant_tokens),
        }

        conv_stats.append(item_stats)
        req_stats.extend(request_tokens)

    # Print statistics
    percentiles = [0.25, 0.5, 0.75, 0.9, 0.99]

    print(TEXT_SEPARATOR)
    print(f"{COLOR_YELLOW}Conversations statistics:{COLOR_RESET}")
    print(TEXT_SEPARATOR)
    df = pd.DataFrame(conv_stats)
    print(df.describe(percentiles=percentiles).transpose())
    print(TEXT_SEPARATOR)
    print(f"{COLOR_YELLOW}Request statistics:{COLOR_RESET}")
    print(TEXT_SEPARATOR)
    df = pd.DataFrame(req_stats, columns=["request_tokens"])
    print(df.describe(percentiles=percentiles).transpose())
    print(TEXT_SEPARATOR)


def generate_conversations(args: GenConvArgs, tokenizer: AutoTokenizer) -> ConversationsMap:

    # Text for all user prompts (text from the input text files will be appended to this line)
    base_prompt_text = "Please rewrite the following text in a humorous style: "
    base_prompt_token_count = len(tokenizer.encode(base_prompt_text, add_special_tokens=False))

    logger.info(f"{COLOR_PURPLE}Generating conversations...{COLOR_RESET}")
    logger.info(args)

    list_of_tokens = []

    for filename in args.text_files:
        # Load text file that will be used to generate prompts
        with open(filename, "r") as file:
            data = file.read()
            tokens_in_file = tokenizer.encode(data, add_special_tokens=False)
            list_of_tokens.extend(tokens_in_file)

    conversations: ConversationsMap = {}
    conv_id = 0

    # Generate number of turns for every conversation
    turn_count: np.ndarray = args.input_num_turns.sample(args.num_conversations)

    # Turn count should be at least 2 (one user prompt and one assistant answer)
    turn_count = np.maximum(turn_count, 2)

    # Round up to an even number (every user prompt should have an answer)
    turn_count = turn_count + (turn_count % 2)

    # Generate number of prefix tokens for every conversation
    conv_prefix_tokens: np.ndarray = args.input_prefix_num_tokens.sample(args.num_conversations)

    # Used to reduce shared text between conversations (jump/skip over text sections between conversations)
    base_offset = 0

    # Common prefix size for all conversations (only 1 sample required)
    common_prefix_text = ""
    common_prefix_tokens: int = args.input_common_prefix_num_tokens.sample(1)[0]
    if common_prefix_tokens > 0:
        # Using "." at the end to separate sentences
        common_prefix_text = tokenizer.decode(list_of_tokens[:common_prefix_tokens - 2]) + "."
        base_offset += common_prefix_tokens

    for conv_id in range(args.num_conversations):
        # Generate a single conversation
        messages: MessagesList = []

        nturns = turn_count[conv_id]

        # User prompt token count per turn (with lower limit)
        input_token_count: np.ndarray = args.input_num_tokens.sample(nturns)
        input_token_count = np.maximum(input_token_count, base_prompt_token_count)

        # Assistant answer token count per turn (with lower limit)
        output_token_count: np.ndarray = args.output_num_tokens.sample(nturns)
        output_token_count = np.maximum(output_token_count, 1)

        user_turn = True
        for turn_id in range(nturns):
            if user_turn:
                role = "user"
                num_tokens = input_token_count[turn_id]

                # Generate the user prompt,
                # use a unique prefix (the conv_id) for each conversation
                # (to avoid shared prefix between conversations)
                content = f"{conv_id} is a nice number... "

                if len(common_prefix_text) > 0 and turn_id == 0:
                    content = common_prefix_text + content

                # Update the number of tokens left for the content
                num_tokens -= len(tokenizer.encode(content, add_special_tokens=False))

                if turn_id == 0:
                    prefix_num_tokens = conv_prefix_tokens[conv_id]
                    if prefix_num_tokens > 0:
                        # Add prefix text (context) to the first turn
                        start_offset = base_offset
                        end_offset = start_offset + prefix_num_tokens
                        assert len(list_of_tokens) > end_offset, \
                        f"Not enough input text to generate {prefix_num_tokens} tokens for the prefix text ({start_offset=}, {end_offset=})"

                        content += f"{conv_id}, " + tokenizer.decode(list_of_tokens[start_offset:end_offset])
                        base_offset += prefix_num_tokens

                # Add the actual user prompt/question after the prefix text
                content += base_prompt_text
                num_tokens -= base_prompt_token_count

                if num_tokens > 0:
                    # Add text from the input text file (to reach the desired token count)
                    start_offset = base_offset + turn_id * input_token_count.max()
                    end_offset = start_offset + num_tokens
                    assert len(list_of_tokens) > end_offset, \
                    f"Not enough input text to generate {num_tokens} tokens for the prompt ({start_offset=}, {end_offset=})"

                    # Convert tokens back to text
                    content += tokenizer.decode(list_of_tokens[start_offset:end_offset])
            else:
                role = "assistant"
                # This content will not be used as input to the LLM server (actual answers will be used instead).
                # Content is only required to determine the min_tokens/max_tokens (inputs to the LLM server).
                num_tokens = output_token_count[turn_id]
                assert len(list_of_tokens) > num_tokens, \
                f"Not enough input text to generate {num_tokens} tokens for assistant content"
                content = tokenizer.decode(list_of_tokens[:num_tokens])

            # Append the user/assistant message to the list of messages
            messages.append({"role": role, "content": content})
            user_turn = not user_turn

        # Add the new conversation
        conversations[f"CONV_ID_{conv_id}"] = messages

        # Increase base offset for the next conversation
        base_offset += nturns

    if args.print_stats:
        print_conv_stats(conversations, tokenizer)

    return conversations


def temperature_float(x: Any) -> float:
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} is not a valid float")
    if x < 0.0 or x > 2.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 2.0]")
    return x


async def main() -> None:
    user_name = getpass.getuser()
    parser = argparse.ArgumentParser(prog="Multi Turn Benchmark Serving",
                                     description="Benchmark LLM online inference using REST API")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")

    parser.add_argument("-i", "--input-file", type=str, default="openai.json",
                        help="Input JSON file containing conversations or configuration file for synthetic conversations")
    parser.add_argument("-o", "--output-file", type=str, default=None,
                        help="Output JSON file containing conversations with answers from the tested LLM")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="Seed for RNGs (default: 0)")
    parser.add_argument("-p", "--num-clients", type=int, default=1,
                        help="Number of clients that will send requests in parallel")
    parser.add_argument("-x", "--max-conversations", type=int, default=None,
                        help="Number of conversations to use from the input file")
    parser.add_argument("-k", "--max-active-conversations", type=int, default=None,
                        help="Max number of active conversations at a time (for all clients)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Maximum number of turns/messages per conversation, includes both user and assistant messages (a positive number, e.g: 2, 4, 6, etc.), disabled by default")
    parser.add_argument("--early-stop", default=False, action="store_true",
                        help="Stop the test if at least one client finished/exited")
    parser.add_argument("--warmup", default=False, action="store_true",
                        help="Run a warmup step (using only the first turn of every conversation), measurements will not be included in the final benchmark results")
    parser.add_argument("-n", "--num-requests", type=int, default=1,
                        help="Number of requests to simulate (default: 1)")
    parser.add_argument("-z", "--limit-max-tokens", type=int, default=None,
                        help="Set max_tokens for each request (use 0 to set max_tokens based on the assistant content from the input file)")
    parser.add_argument("-w", "--limit-min-tokens", type=int, default=None,
                        help="Set min_tokens for each request (use 0 to set min_tokens based on the assistant content from the input file)")
    parser.add_argument("-l", "--lambda", type=float, default=0.0, dest="lambda_param",
                        help="Lambda parameter (Poisson distribution) for the request rate of each client (use 0 for no delay between requests)")
    parser.add_argument("-c", "--conversation-sampling",
                        type=lambda s: ConversationSampling[s.upper()],
                        choices=list(ConversationSampling),
                        default=ConversationSampling.ROUND_ROBIN,
                        help="Conversation sampling from the input file (for each request)")
    parser.add_argument("-q", "--verify-output", default=False, action="store_true",
                        help="Verify the LLM output (compare to the answers in the input JSON file)")
    parser.add_argument("-m", "--model", type=str, default=f"/home/{user_name}/workspace/hf_models/meta-llama/Meta-Llama-3.1-8B-Instruct/",
                        help="Select LLM model for the LLM chatbot")
    parser.add_argument("-u", "--url", type=str, default="http://localhost:8000",
                        help="Base URL for the LLM API server")
    parser.add_argument("-j", "--disable-stream", default=False, action="store_true",
                        help="Disable stream/streaming mode (set 'stream' to False in the API request)")
    parser.add_argument("--temperature", type=temperature_float, default=0.0,
                        help="Float that controls the randomness of the sampling (sets the 'temperature' in the API request). Zero means greedy sampling.")

    parser.add_argument("--prefix-data-file", type=str, default=None,
                        help="Text file that will be used to generate prefix text (context) for every prompt")
    parser.add_argument("--prefix-num-words", type=int, default=1,
                        help="The number of words that will be used as prefix text for every prompt (requires --prefix-data-file)")

    parser.add_argument("-e", "--excel-output", default=False, action="store_true",
                        help="Export summary to Excel file (optional)")
    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--print-content", default=False, action="store_true",
                        help="Print the user prompts and the server's answers")
    parser.add_argument("-d", "--detailed-stats", default=False, action="store_true",
                        help="Enable detailed statistics (per client)")
    parser.add_argument("-g", "--metrics-interval-sec", type=int, default=0,
                        help="Collect metrics from the API server (metrics endpoint), this flag's value is the sampling interval in seconds")

    parser.add_argument("--collect-gpu-metrics", default=False, action="store_true",
                        help="Collect energy/power consumption metrics from all local GPU devices")

    # Target SLAs (can be average, 90% percentile, etc.) -
    # a warning will be printed in case of SLA violation.
    parser.add_argument("--sla-ttft", type=int, default=20000,
                        help="Target SLA for TTFT in milliseconds")
    parser.add_argument("--sla-tpot", type=int, default=400,
                        help="Target SLA for TPOT in milliseconds")

    parser.add_argument("--warmup-percentage", type=str, default="0%,25%,50%,75%",
                        help="Ignore the first X samples as warmup (X is a percentage). A comma separated list of percentages can be used (for example: --warmup-percentage=0%,50%)")

    args = parser.parse_args()

    logger.info(args)

    logger.info(f"{COLOR_GREEN}Input parameters:{COLOR_RESET}")
    logger.info(f"url={args.url}")
    logger.info(f"model={args.model}")
    logger.info(f"num_clients={args.num_clients}")

    if args.verify_output is True:
        logger.info(f"{COLOR_PURPLE}Verify is enabled{COLOR_RESET}")

    if args.prefix_data_file:
        assert os.path.isfile(args.prefix_data_file), \
        f"File not found: {args.prefix_data_file}"
        assert args.prefix_num_words > 0, \
        "Prefix num words must be larger than zero"
        logger.info(f"Using prefix data file (prefix num words {args.prefix_num_words}): {args.prefix_data_file}")

    if args.limit_min_tokens is not None:
        assert args.limit_min_tokens >= 0, "--limit-min-tokens can be zero or larger"
        text = args.limit_min_tokens
        if args.limit_min_tokens == NUM_TOKENS_FROM_DATASET:
            text = "based on input file"
        logger.info(f"{COLOR_PURPLE}Limit min tokens is enabled ({text}){COLOR_RESET}")

    if args.limit_max_tokens is not None:
        assert args.limit_max_tokens >= 0, "--limit-max-tokens can be zero or larger"
        text = args.limit_min_tokens
        if args.limit_min_tokens == NUM_TOKENS_FROM_DATASET:
            text = "based on input file"
        logger.info(f"{COLOR_PURPLE}Limit max tokens is enabled ({text}){COLOR_RESET}")

    # Calculate the amount of samples to filter (as warmup samples/measurements).
    try:
        if args.warmup:
            # Warmup percentage is not required is a separate warmup step was used
            warmup_percentages: List[float] = [0.0]
        else:
            warmup_strings: List[str] = args.warmup_percentage.split(",")
            warmup_strings = [x.replace("%", "") for x in warmup_strings]
            warmup_percentages = [float(x) / 100 for x in warmup_strings]

        # Check for valid range (0 to 1)
        for p in warmup_percentages:
            assert p >= 0.0 and p < 1.0

        # Sort from high to low warmup percentage
        warmup_percentages.sort()

    except Exception:
        logger.error(f"Invalid input --warmup-percentage={args.warmup_percentage}")
        exit(1)

    logger.info(f"Warmup percentages (percentage of samples): {warmup_percentages}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    assert os.path.exists(args.model), f"Path does not exist: {args.model}"
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    await get_server_info(args.url)

    # Load the input file (either conversations of configuration file)
    logger.info(f"Reading input file: {args.input_file}")
    with open(args.input_file, "r") as f:
        input_data = json.load(f)

    gen_conv_args = None
    if isinstance(input_data, list):
        # The conversations are stored as a list of dicts
        logger.info(f"Found {len(input_data)} items in the input file")

        if args.max_conversations is not None:
            # Limit the number of conversations to use
            sample_size = args.max_conversations
            if len(input_data) > sample_size:
                logger.info(f"Sampling {sample_size} conversations out of {len(input_data)}")
                input_data = random.sample(input_data, sample_size)

        # Convert the list to a ConversationsMap
        conversations = conversations_list_to_dict(input_data)

    elif isinstance(input_data, dict):
        # The input file is a configuration file (type is determined by the field 'filetype')
        if "filetype" not in input_data:
            raise Exception(f"Input file {args.input_file} is invalid (missing 'filetype')")

        logger.info(f"Using input file with filetype: {input_data['filetype']}")

        gen_conv_args = parse_input_json_file(input_data)

        # Disable warning from "huggingface/tokenizers" (when using python multiprocessing and tokenizers)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Generate synthetic conversations
        conversations = generate_conversations(gen_conv_args, tokenizer)

    else:
        raise Exception(f"Input file {args.input_file} is invalid")

    if args.max_turns is not None:
        assert args.max_turns > 0, \
        "Max turns must be a positive number"
        logger.info(f"{COLOR_PURPLE}Max turns per conversation is limited to {args.max_turns}{COLOR_RESET}")

    num_conversations = len(conversations)
    assert num_conversations >= args.num_clients, \
    "Number of conversations must be equal or larger than the number of clients"

    if args.collect_gpu_metrics:
        # Get total energy consumption before the test started
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        energy_consumptions = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # Retrieves total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded
            energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            energy_consumptions.append(energy)

    # Create benchmark configurations
    client_args, req_args = get_client_config(args, conversations)

    bench_args = BenchmarkArgs(url=args.url,
                               num_clients=args.num_clients,
                               early_stop=args.early_stop,
                               metrics_interval_sec=args.metrics_interval_sec)

    # Warm-up step
    if args.warmup:
        # Only send a single user prompt from every conversation.
        # max_active_conversations must be 1,
        # otherwise the clients may exit after sending a single request
        # (because the task queue is empty).
        warmup_client_args = client_args._replace(skip_first_turn=False,
                                                  max_turns=1,
                                                  max_active_conversations=1)

        # Early stop should be disabled,
        # all clients should finish their work before exiting
        warmup_bench_args = bench_args._replace(early_stop=False,
                                                metrics_interval_sec=0)

        logger.info(f"{COLOR_PURPLE}Warmup start{COLOR_RESET}")
        conversations, _, _ = await main_mp(warmup_client_args, req_args,
                                            warmup_bench_args,
                                            tokenizer, conversations)
        logger.info(f"{COLOR_PURPLE}Warmup done{COLOR_RESET}")

    # Run the benchmark
    start_time = time.perf_counter_ns()
    client_convs, client_metrics, server_metrics = await main_mp(client_args, req_args, bench_args,
                                                                 tokenizer, conversations)
    total_runtime_ms = nanosec_to_millisec(time.perf_counter_ns() - start_time)

    # Calculate requests per second
    total_runtime_sec = total_runtime_ms / 1000.0
    rps = len(client_metrics) / total_runtime_sec
    logger.info(f"{COLOR_GREEN}All clients finished, total runtime: {total_runtime_sec:.3f} sec ({total_runtime_ms:.3f} ms), requests per second: {rps:.3f}{COLOR_RESET}")

    # Benchmark parameters
    params = {
        "model": args.model,
        "num_clients": args.num_clients,
        "num_conversations": num_conversations,
        "active_conversations": args.max_active_conversations,
        "seed": args.seed,
    }

    if args.collect_gpu_metrics:
        # Get total energy consumed during the test
        total_energy_mj = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            if energy is not None:
                # None means that the device does not support energy consumption measurement
                energy_consumptions[i] = energy - energy_consumptions[i]
                total_energy_mj += energy_consumptions[i]

        pynvml.nvmlShutdown()

        total_power_watt = (total_energy_mj / 1000.0) / total_runtime_sec
        logger.info(f"{COLOR_GREEN}Total energy/power consumption (all GPUs): {total_energy_mj:,} mJ / {total_power_watt:.3f} W{COLOR_RESET}")
        logger.info(f"Energy consumption per GPU (mJ): {energy_consumptions}")

        params["total_energy_mj"] = total_energy_mj
        params["total_power_watt"] = total_power_watt

    if args.limit_min_tokens is not None:
        params["min_tokens"] = args.limit_min_tokens

    if args.limit_max_tokens is not None:
        params["max_tokens"] = args.limit_max_tokens

    # Process and print statistics (and save excel file with the statistics)
    process_statistics(client_metrics,
                       test_params=params,
                       warmup_percentages=warmup_percentages,
                       verbose=args.verbose,
                       gen_conv_args=gen_conv_args,
                       excel_output=args.excel_output)

    if server_metrics is not None:
        server_metrics_report(server_metrics)

    if args.output_file is not None:
        # Write a JSON file with the updated conversations
        # The "assistant" content will contain the answers from the tested LLM
        output_data: ShareGptConversations = conversations_dict_to_list(client_convs)
        logger.info(f"{COLOR_GREEN}Writing conversations file: {args.output_file}{COLOR_RESET}")
        with open(args.output_file, "w") as f:
            json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
