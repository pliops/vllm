# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
from unittest.mock import MagicMock

import numpy as np  # type: ignore
import pytest  # type: ignore
from bench_dataset import (
    ConstantDistribution,
    ConversationsMap,
    Distribution,
    GenConvArgs,
    LognormalDistribution,
    PoissonDistribution,
    UniformDistribution,
    ZipfDistribution,
    generate_conversations,
    parse_input_json_file,
)


@pytest.fixture(scope="session")
def mock_tokenizer():
    # Simple tokenizer mock: token is one char, decode/encode are reversible
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [
        ord(c) for c in text
    ]
    tokenizer.decode.side_effect = lambda tokens: "".join(chr(t) for t in tokens)
    return tokenizer


@pytest.mark.parametrize("input_file", ["generate_multi_turn.json"])
def test_json_parsing(input_file) -> None:
    print(f"Loading file: {input_file}")
    with open(input_file) as f:
        input_data = json.load(f)

    gen_conv_args = parse_input_json_file(input_data)

    # Basic check
    assert input_data["num_conversations"] == gen_conv_args.num_conversations

    # Test optional subsection
    input_data["prompt_input"].pop("common_prefix_num_tokens", None)

    gen_conv_args = parse_input_json_file(input_data)

    # Common prefix will always be zero if not found in the input JSON file
    assert isinstance(
        gen_conv_args.input_common_prefix_num_tokens, ConstantDistribution
    )
    assert gen_conv_args.input_common_prefix_num_tokens.sample(1)[0] == 0


@pytest.mark.parametrize("seed", [0, 10, 999, 54321])
def test_distribution_lognormal(seed) -> None:

    # Use the same seed for reproducibility
    np.random.seed(seed)

    # Averages to test
    averages = [
        1000,
        2000,
        4000,
        6000,
        8000,
        10000,
        12000,
        16000,
        18000,
        20000,
        24000,
        28000,
        32000,
    ]

    # Allow 3% of error
    tolerance = 0.03

    for median_ratio in [None, 0.85, 0.5, 0.3, 0.1]:
        for target_average in averages:
            dist = LognormalDistribution(average=target_average,
                                         median_ratio=median_ratio)

            print(f"Testing: {dist}")

            # Sample from the random distribution
            num_samples = 500
            samples = dist.sample(num_samples)

            actual_average = samples.mean()

            assert math.isclose(actual_average, target_average, rel_tol=tolerance), (
                f"Relative difference of 'average' is more than {tolerance:.2%}, "
                f"{actual_average=}, {target_average=}"
            )

            # Check that the min is "much smaller" than the average
            # (The samples should not be too concentrated around the average)
            min_val = samples.min()
            assert min_val < actual_average * 0.25, \
            f"Min value {min_val} is not much smaller than the average {actual_average}"


def verify_conversations(
    mock_tokenizer: MagicMock,
    args: GenConvArgs,
    conversations: ConversationsMap,
    input_prefix_num_tokens: Distribution,
    input_num_tokens: Distribution,
    output_num_tokens: Distribution,
) -> None:
    assert len(conversations) == args.num_conversations

    for conv_id, messages in conversations.items():
        assert isinstance(conv_id, str), "Conversation ID should be a string"
        assert len(messages) > 0 and len(messages) % 2 == 0, (
            "Message count should be non zero and even"
        )

        for turn_id, message in enumerate(messages):
            assert "role" in message
            assert "content" in message

            # Convert text to tokens
            content = message["content"]
            token_count = len(mock_tokenizer.encode(content))

            assert token_count > 0, f"Token count is zero in {conv_id=} and {turn_id=}"

            role = message["role"]
            if turn_id % 2 == 0:
                assert role == "user"
                if (
                    turn_id > 0 or input_prefix_num_tokens.max_val == 0
                ) and input_num_tokens.max_val is not None:
                    # User turn without a prefix, verify max token count if relevant
                    assert token_count <= input_num_tokens.max_val, (
                        f"{conv_id=}, {turn_id=}, invalid input token count "
                        f"{token_count} (above max)"
                    )

                if hasattr(input_num_tokens, "min_val") and input_num_tokens.min_val:
                    assert token_count >= input_num_tokens.min_val, (
                        f"{conv_id=}, {turn_id=}, invalid input token count "
                        f"{token_count} (below min)"
                    )

            else:
                assert role == "assistant"
                if hasattr(output_num_tokens, "max_val") and output_num_tokens.max_val:
                    assert token_count <= output_num_tokens.max_val, (
                        f"{conv_id=}, {turn_id=}, invalid output token count "
                        f"{token_count} (above max)"
                    )

                if hasattr(output_num_tokens, "min_val") and output_num_tokens.min_val:
                    assert token_count >= output_num_tokens.min_val, (
                        f"{conv_id=}, {turn_id=}, invalid output token count "
                        f"{token_count} (below min)"
                    )


def test_prefix(mock_tokenizer) -> None:
    text_files = ["pg1184.txt"]
    num_conversations = 30
    input_num_turns = ConstantDistribution(10)

    # No prefix
    input_common_prefix_num_tokens = ConstantDistribution(0)
    input_prefix_num_tokens = ConstantDistribution(0)

    input_num_tokens = ConstantDistribution(1)
    output_num_tokens = ConstantDistribution(1)

    args = GenConvArgs(
        num_conversations=num_conversations,
        text_files=text_files,
        input_num_turns=input_num_turns,
        input_common_prefix_num_tokens=input_common_prefix_num_tokens,
        input_prefix_num_tokens=input_prefix_num_tokens,
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        print_stats=False,
    )

    print(f"Running config: {args}")
    conversations = generate_conversations(args, mock_tokenizer)

    # Get lower/upper limit for a user prompt when there is no prefix
    no_prefix_max = 0
    no_prefix_min = 0
    for conv_id, messages in conversations.items():
        for turn_id, message in enumerate(messages):
            # Prefix is relevant only for user content
            if message["role"] == "user":
                # Convert text to tokens
                content = message["content"]
                token_count = len(mock_tokenizer.encode(content))
                assert token_count > 0

                if no_prefix_max == 0:
                    # Initial values
                    no_prefix_max = token_count
                    no_prefix_min = token_count
                else:
                    # Update min/max
                    no_prefix_max = max(token_count, no_prefix_max)
                    no_prefix_min = min(token_count, no_prefix_min)

    # Check common/unique prefix (for first user turn)
    input_common_prefix_num_tokens = ConstantDistribution(321)
    input_prefix_num_tokens = ConstantDistribution(123)

    args = GenConvArgs(
        num_conversations=num_conversations,
        text_files=text_files,
        input_num_turns=input_num_turns,
        input_common_prefix_num_tokens=input_common_prefix_num_tokens,
        input_prefix_num_tokens=input_prefix_num_tokens,
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        print_stats=False,
    )

    print(f"Running config: {args}")
    conversations = generate_conversations(args, mock_tokenizer)

    # Total prefix size
    prefix_token_count = (
        input_common_prefix_num_tokens.value + input_prefix_num_tokens.value
    )

    for conv_id, messages in conversations.items():
        for turn_id, message in enumerate(messages):
            # Prefix is relevant only for user content
            if message["role"] == "user":
                # Convert text to tokens
                content = message["content"]
                token_count = len(mock_tokenizer.encode(content))

                if turn_id == 0:
                    # Only first user turn should have a prefix
                    assert token_count >= (prefix_token_count + no_prefix_min), (
                        f"{conv_id=}, {turn_id=}, invalid input token count "
                        f"{token_count} ({prefix_token_count=}, {no_prefix_min=})"
                    )
                else:
                    # All other turns should not include a prefix
                    assert token_count <= no_prefix_max, (
                        f"{conv_id=}, {turn_id=}, invalid input token count "
                        f"{token_count} ({prefix_token_count=}, {no_prefix_max=})"
                    )


@pytest.mark.parametrize("num_conversations", [10, 200])
@pytest.mark.parametrize(
    "input_num_turns",
    [
        ConstantDistribution(50),
        UniformDistribution(10, 20),
        PoissonDistribution(3, 30),
        ZipfDistribution(1.3, 30),
    ],
)
@pytest.mark.parametrize(
    "input_common_prefix_num_tokens", [UniformDistribution(50, 100)]
)
@pytest.mark.parametrize(
    "input_prefix_num_tokens",
    [
        ConstantDistribution(0),
        UniformDistribution(300, 600),
        LognormalDistribution(3, 2),
    ],
)
@pytest.mark.parametrize(
    "input_num_tokens", [UniformDistribution(200, 600), PoissonDistribution(200, 600)]
)
@pytest.mark.parametrize(
    "output_num_tokens", [ConstantDistribution(1), UniformDistribution(50, 150)]
)
def test_generation(
    mock_tokenizer,
    num_conversations,
    input_num_turns,
    input_common_prefix_num_tokens,
    input_prefix_num_tokens,
    input_num_tokens,
    output_num_tokens,
) -> None:
    text_files = ["pg1184.txt"]

    args = GenConvArgs(
        num_conversations=num_conversations,
        text_files=text_files,
        input_num_turns=input_num_turns,
        input_common_prefix_num_tokens=input_common_prefix_num_tokens,
        input_prefix_num_tokens=input_prefix_num_tokens,
        input_num_tokens=input_num_tokens,
        output_num_tokens=output_num_tokens,
        print_stats=False,
    )

    print(f"Running config: {args}")
    conversations = generate_conversations(args, mock_tokenizer)

    verify_conversations(
        mock_tokenizer,
        args,
        conversations,
        input_prefix_num_tokens,
        input_num_tokens,
        output_num_tokens,
    )
