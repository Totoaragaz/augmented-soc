import base64
import io
import json
import os
import time
import traceback
import sys

import easyocr
import ollama
import pkg_resources
from PIL import Image
from ultralytics import YOLO

from operate.config import Config
from operate.exceptions import ModelNotRecognizedException
from operate.models.prompts import (
    get_system_prompt,
    get_user_first_message_prompt,
    get_user_prompt,
)
from operate.utils.label import (
    add_labels,
    get_click_position_in_percent,
    get_label_coordinates,
)
from operate.utils.ocr import get_text_coordinates, get_text_element
from operate.utils.screenshot import capture_screen_with_cursor, compress_screenshot
from operate.utils.style import ANSI_BRIGHT_MAGENTA, ANSI_GREEN, ANSI_RED, ANSI_RESET

# Load configuration
config = Config()


async def get_next_action(model, messages, objective, session_id):
    if config.verbose:
        print("[Self-Operating Computer][get_next_action]")
        print("[Self-Operating Computer][get_next_action] model", model)
    if model == "gpt-4":
        return call_gpt_4o(messages), None
    if model == "qwen-vl":
        operation = await call_qwen_vl_with_ocr(messages, objective, model)
        return operation, None
    if model == "gpt-4-with-ocr":
        operation = await call_gpt_4o_with_ocr(messages, objective, model)
        return operation, None
    if model == "agent-1":
        return "coming soon"
    if model == "gemini-pro-vision":
        operation = await call_gemini_pro_vision(messages, objective, model)
        return operation, None
    if model == "llava":
        operation = await call_ollama_llava(messages, objective, model)
        return operation, None
    if model == "claude-3":
        operation = await call_claude_3_with_ocr(messages, objective, model)
        return operation, None
    raise ModelNotRecognizedException(model)


def call_gpt_4o(messages):
    if config.verbose:
        print("[call_gpt_4_v]")
    time.sleep(1)
    client = config.initialize_openai()
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        with open(screenshot_filename, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        if config.verbose:
            print(
                "[call_gpt_4_v] user_prompt",
                user_prompt,
            )

        vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }
        messages.append(vision_message)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            presence_penalty=1,
            frequency_penalty=1,
        )

        content = response.choices[0].message.content

        content = clean_json(content)

        assistant_message = {"role": "assistant", "content": content}
        if config.verbose:
            print(
                "[call_gpt_4_v] content",
                content,
            )
        content = json.loads(content)

        messages.append(assistant_message)

        return content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[Operate] That did not work. Trying again {ANSI_RESET}",
            e,
        )
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] AI response was {ANSI_RESET}",
            content,
        )
        if config.verbose:
            traceback.print_exc()
        sys.exit()


async def call_qwen_vl_with_ocr_old(messages, objective, model):
    if config.verbose:
        print("[call_qwen_vl_with_ocr]")

    # Construct the path to the file within the package
    try:
        time.sleep(1)
        client = config.initialize_qwen()

        confirm_system_prompt(messages, objective, model)
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        # Call the function to capture the screen with the cursor
        raw_screenshot_filename = os.path.join(screenshots_dir, "raw_screenshot.png")
        capture_screen_with_cursor(raw_screenshot_filename)

        # Compress screenshot image to make size be smaller
        screenshot_filename = os.path.join(screenshots_dir, "screenshot.jpeg")
        compress_screenshot(raw_screenshot_filename, screenshot_filename)

        with open(screenshot_filename, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        vision_message = {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"{user_prompt}**REMEMBER** Only output json format, do not append any other text."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }
        messages.append(vision_message)

        response = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=messages,
        )

        content = response.choices[0].message.content

        content = clean_json(content)

        # used later for the messages
        content_str = content

        content = json.loads(content)

        processed_content = []

        for operation in content:
            if operation.get("operation") == "click":
                text_to_click = operation.get("text")
                if config.verbose:
                    print(
                        "[call_qwen_vl_with_ocr][click] text_to_click",
                        text_to_click,
                    )
                # Initialize EasyOCR Reader
                reader = easyocr.Reader(["en"])

                # Read the screenshot
                result = reader.readtext(screenshot_filename)

                text_element_index = get_text_element(
                    result, text_to_click, screenshot_filename
                )
                coordinates = get_text_coordinates(
                    result, text_element_index, screenshot_filename
                )

                # add `coordinates`` to `content`
                operation["x"] = coordinates["x"]
                operation["y"] = coordinates["y"]

                if config.verbose:
                    print(
                        "[call_qwen_vl_with_ocr][click] text_element_index",
                        text_element_index,
                    )
                    print(
                        "[call_qwen_vl_with_ocr][click] coordinates",
                        coordinates,
                    )
                    print(
                        "[call_qwen_vl_with_ocr][click] final operation",
                        operation,
                    )
                processed_content.append(operation)

            else:
                processed_content.append(operation)

        # wait to append the assistant message so that if the `processed_content` step fails we don't append a message and mess up message history
        assistant_message = {"role": "assistant", "content": content_str}
        messages.append(assistant_message)

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Trying another method {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()

def call_gemini_pro_vision_old(messages, objective):
    """
    Get the next action for Self-Operating Computer using Gemini Pro Vision
    """
    if config.verbose:
        print(
            "[Self Operating Computer][call_gemini_pro_vision]",
        )
    # sleep for a second
    time.sleep(1)
    try:
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)
        # sleep for a second
        time.sleep(1)
        prompt = get_system_prompt("gemini-pro-vision", objective)

        model = config.initialize_google()
        if config.verbose:
            print("[call_gemini_pro_vision] model", model)

        response = model.generate_content([prompt, Image.open(screenshot_filename)])

        content = clean_json(response.text)
        if config.verbose:
            print("[call_gemini_pro_vision] response", response)
            print("[call_gemini_pro_vision] content", content)

        content = json.loads(content)
        if config.verbose:
            print(
                "[get_next_action][call_gemini_pro_vision] content",
                content,
            )

        return content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[Operate] That did not work. Trying another method {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()


async def call_gpt_4o_with_ocr_old(messages, objective, model):
    if config.verbose:
        print("[call_gpt_4o_with_ocr]")

    # Construct the path to the file within the package
    try:
        time.sleep(1)
        client = config.initialize_openai()

        confirm_system_prompt(messages, objective, model)
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        with open(screenshot_filename, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        vision_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ],
        }

        messages.append(vision_message)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        content = response.choices[0].message.content

        content = clean_json(content)

        # used later for the messages
        content_str = content

        content = json.loads(content)

        processed_content = []

        for operation in content:
            if operation.get("operation") == "click":
                text_to_click = operation.get("text")
                if config.verbose:
                    print(
                        "[call_gpt_4o_with_ocr][click] text_to_click",
                        text_to_click,
                    )
                # Initialize EasyOCR Reader
                reader = easyocr.Reader(["en"])

                # Read the screenshot
                result = reader.readtext(screenshot_filename)

                text_element_index = get_text_element(
                    result, text_to_click, screenshot_filename
                )
                coordinates = get_text_coordinates(
                    result, text_element_index, screenshot_filename
                )

                # add `coordinates`` to `content`
                operation["x"] = coordinates["x"]
                operation["y"] = coordinates["y"]

                if config.verbose:
                    print(
                        "[call_gpt_4o_with_ocr][click] text_element_index",
                        text_element_index,
                    )
                    print(
                        "[call_gpt_4o_with_ocr][click] coordinates",
                        coordinates,
                    )
                    print(
                        "[call_gpt_4o_with_ocr][click] final operation",
                        operation,
                    )
                processed_content.append(operation)

            else:
                processed_content.append(operation)

        # wait to append the assistant message so that if the `processed_content` step fails we don't append a message and mess up message history
        assistant_message = {"role": "assistant", "content": content_str}
        messages.append(assistant_message)

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Trying another method {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()


def call_ollama_llava_old(messages):
    if config.verbose:
        print("[call_ollama_llava]")
    time.sleep(1)
    try:
        model = config.initialize_ollama()
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        # Call the function to capture the screen with the cursor
        capture_screen_with_cursor(screenshot_filename)

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        if config.verbose:
            print(
                "[call_ollama_llava] user_prompt",
                user_prompt,
            )

        vision_message = {
            "role": "user",
            "content": user_prompt,
            "images": [screenshot_filename],
        }
        messages.append(vision_message)

        response = model.chat(
            model="llava",
            messages=messages,
        )

        # Important: Remove the image path from the message history.
        # Ollama will attempt to load each image reference and will
        # eventually timeout.
        messages[-1]["images"] = None

        content = response["message"]["content"].strip()

        content = clean_json(content)

        assistant_message = {"role": "assistant", "content": content}
        if config.verbose:
            print(
                "[call_ollama_llava] content",
                content,
            )
        content = json.loads(content)

        messages.append(assistant_message)

        return content

    except ollama.ResponseError as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Operate] Couldn't connect to Ollama. With Ollama installed, run `ollama pull llava` then `ollama serve`{ANSI_RESET}",
            e,
        )

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[llava] That did not work. Trying again {ANSI_RESET}",
            e,
        )
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] AI response was {ANSI_RESET}",
            content,
        )
        if config.verbose:
            traceback.print_exc()
        return call_ollama_llava(messages)


async def call_claude_3_with_ocr_old(messages, objective, model):
    if config.verbose:
        print("[call_claude_3_with_ocr]")

    try:
        time.sleep(1)
        client = config.initialize_anthropic()

        confirm_system_prompt(messages, objective, model)
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
        capture_screen_with_cursor(screenshot_filename)

        # downsize screenshot due to 5MB size limit
        with open(screenshot_filename, "rb") as img_file:
            img = Image.open(img_file)

            # Convert RGBA to RGB
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # Calculate the new dimensions while maintaining the aspect ratio
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            new_width = 2560  # Adjust this value to achieve the desired file size
            new_height = int(new_width / aspect_ratio)
            if config.verbose:
                print("[call_claude_3_with_ocr] resizing claude")

            # Resize the image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the resized and converted image to a BytesIO object for JPEG format
            img_buffer = io.BytesIO()
            img_resized.save(
                img_buffer, format="JPEG", quality=85
            )  # Adjust the quality parameter as needed
            img_buffer.seek(0)

            # Encode the resized image as base64
            img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        if len(messages) == 1:
            user_prompt = get_user_first_message_prompt()
        else:
            user_prompt = get_user_prompt()

        vision_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data,
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt
                    + "**REMEMBER** Only output json format, do not append any other text.",
                },
            ],
        }
        messages.append(vision_message)

        # anthropic api expect system prompt as an separate argument
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=3000,
            system=messages[0]["content"],
            messages=messages[1:],
        )

        content = response.content[0].text
        content = clean_json(content)
        content_str = content
        try:
            content = json.loads(content)
        # rework for json mode output
        except json.JSONDecodeError as e:
            if config.verbose:
                print(
                    f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Error] JSONDecodeError: {e} {ANSI_RESET}"
                )
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=3000,
                system=f"This json string is not valid, when using with json.loads(content) \
                it throws the following error: {e}, return correct json string. \
                **REMEMBER** Only output json format, do not append any other text.",
                messages=[{"role": "user", "content": content}],
            )
            content = response.content[0].text
            content = clean_json(content)
            content_str = content
            content = json.loads(content)

        if config.verbose:
            print(
                f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] content: {content} {ANSI_RESET}"
            )
        processed_content = []

        for operation in content:
            if operation.get("operation") == "click":
                text_to_click = operation.get("text")
                if config.verbose:
                    print(
                        "[call_claude_3_ocr][click] text_to_click",
                        text_to_click,
                    )
                # Initialize EasyOCR Reader
                reader = easyocr.Reader(["en"])

                # Read the screenshot
                result = reader.readtext(screenshot_filename)

                # limit the text to extract has a higher success rate
                text_element_index = get_text_element(
                    result, text_to_click[:3], screenshot_filename
                )
                coordinates = get_text_coordinates(
                    result, text_element_index, screenshot_filename
                )

                # add `coordinates`` to `content`
                operation["x"] = coordinates["x"]
                operation["y"] = coordinates["y"]

                if config.verbose:
                    print(
                        "[call_claude_3_ocr][click] text_element_index",
                        text_element_index,
                    )
                    print(
                        "[call_claude_3_ocr][click] coordinates",
                        coordinates,
                    )
                    print(
                        "[call_claude_3_ocr][click] final operation",
                        operation,
                    )
                processed_content.append(operation)

            else:
                processed_content.append(operation)

        assistant_message = {"role": "assistant", "content": content_str}
        messages.append(assistant_message)

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Trying another method {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
            print("message before convertion ", messages)

        # Convert the messages to the GPT-4 format
        gpt4_messages = [messages[0]]  # Include the system message
        for message in messages[1:]:
            if message["role"] == "user":
                # Update the image type format from "source" to "url"
                updated_content = []
                for item in message["content"]:
                    if isinstance(item, dict) and "type" in item:
                        if item["type"] == "image":
                            updated_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{item['source']['data']}"
                                    },
                                }
                            )
                        else:
                            updated_content.append(item)

                gpt4_messages.append({"role": "user", "content": updated_content})
            elif message["role"] == "assistant":
                gpt4_messages.append(
                    {"role": "assistant", "content": message["content"]}
                )

        sys.exit()


def get_last_assistant_message(messages):
    """
    Retrieve the last message from the assistant in the messages array.
    If the last assistant message is the first message in the array, return None.
    """
    for index in reversed(range(len(messages))):
        if messages[index]["role"] == "assistant":
            if index == 0:  # Check if the assistant message is the first in the array
                return None
            else:
                return messages[index]
    return None  # Return None if no assistant message is found


def gpt_4_fallback(messages, objective, model):
    if config.verbose:
        print("[gpt_4_fallback]")
    system_prompt = get_system_prompt("gpt-4o", objective)
    new_system_message = {"role": "system", "content": system_prompt}
    # remove and replace the first message in `messages` with `new_system_message`

    messages[0] = new_system_message

    if config.verbose:
        print("[gpt_4_fallback][updated]")
        print("[gpt_4_fallback][updated] len(messages)", len(messages))

    sys.exit()


def confirm_system_prompt(messages, objective, model):
    """
    On `Exception` we default to `call_gpt_4_vision_preview` so we have this function to reassign system prompt in case of a previous failure
    """
    if config.verbose:
        print("[confirm_system_prompt] model", model)

    system_prompt = get_system_prompt(model, objective)
    new_system_message = {"role": "system", "content": system_prompt}
    # remove and replace the first message in `messages` with `new_system_message`

    messages[0] = new_system_message

    if config.verbose:
        print("[confirm_system_prompt]")
        print("[confirm_system_prompt] len(messages)", len(messages))
        for m in messages:
            if m["role"] != "user":
                print("--------------------[message]--------------------")
                print("[confirm_system_prompt][message] role", m["role"])
                print("[confirm_system_prompt][message] content", m["content"])
                print("------------------[end message]------------------")


def clean_json(content):
    if config.verbose:
        print("\n\n[clean_json] content before cleaning", content)
    if content.startswith("```json"):
        content = content[
            len("```json") :
        ].strip()  # Remove starting ```json and trim whitespace
    elif content.startswith("```"):
        content = content[
            len("```") :
        ].strip()  # Remove starting ``` and trim whitespace
    if content.endswith("```"):
        content = content[
            : -len("```")
        ].strip()  # Remove ending ``` and trim whitespace

    # Normalize line breaks and remove any unwanted characters
    content = "\n".join(line.strip() for line in content.splitlines())

    if config.verbose:
        print("\n\n[clean_json] content after cleaning", content)

    return content


def interpret_model_response(operations, screenshot_path, verbose=False):
    """
    Process the raw operations returned by an AI model and resolve any 'click by text' actions to coordinates.

    Args:
        operations (list or str): Raw JSON string or already-parsed list of operations.
        screenshot_path (str): Path to the screenshot to be analyzed via OCR.
        verbose (bool): Whether to print debug information.

    Returns:
        list: List of resolved operation dictionaries, with coordinates added where needed.
    """

    # Parse if still in string form
    if isinstance(operations, str):
        operations = clean_json(operations)
        operations = json.loads(operations)
    
    if isinstance(operations, dict):
        operations = [operations]

    processed = []
    reader = easyocr.Reader(["en"])

    # OCR scan of the screenshot once
    ocr_results = reader.readtext(screenshot_path)

    for operation in operations:
        if operation.get("operation") == "click" and "text" in operation:
            text_to_click = operation["text"]

            if verbose:
                print("[interpret_model_response] Resolving text:", text_to_click)

            try:
                index = get_text_element(ocr_results, text_to_click, screenshot_path)
                coords = get_text_coordinates(ocr_results, index, screenshot_path)
                operation["x"] = coords["x"]
                operation["y"] = coords["y"]
            except Exception as e:
                if verbose:
                    print(f"[interpret_model_response] Text not found: {text_to_click}")
                    print(f"[interpret_model_response] Skipping click or marking as failed: {e}")
                operation["x"] = None
                operation["y"] = None
                operation["error"] = "text_not_found"

        processed.append(operation)

    return processed


def prepare_vision_prompt(messages):
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    screenshot_filename = os.path.join(screenshots_dir, "screenshot.png")
    capture_screen_with_cursor(screenshot_filename)

    with open(screenshot_filename, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    if len(messages) == 1:
        user_prompt = get_user_first_message_prompt()
    else:
        user_prompt = get_user_prompt()

    # Vision-style message must have 'role': 'user', 'content': [TEXT, IMAGE]
    vision_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ],
    }

    return screenshot_filename, vision_message

async def call_gpt_4o_with_ocr(messages, objective, model):
    if config.verbose:
        print("[call_gpt_4o_with_ocr]")

    try:
        time.sleep(1)
        client = config.initialize_openai()
        confirm_system_prompt(messages, objective, model)

        screenshot_path, vision_message = prepare_vision_prompt(messages)
        messages.append(vision_message)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        raw_content = response.choices[0].message.content
        processed_content = interpret_model_response(raw_content, screenshot_path)

        # Append full JSON as string to message history
        messages.append({"role": "assistant", "content": json.dumps(processed_content)})

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Skipping model. {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()

async def call_qwen_vl_with_ocr(messages, objective, model):
    if config.verbose:
        print("[call_qwen_vl_with_ocr]")

    try:
        time.sleep(1)
        client = config.initialize_qwen()
        confirm_system_prompt(messages, objective, model)

        screenshot_path, vision_message = prepare_vision_prompt(messages)
        messages.append(vision_message)

        response = client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=messages,
        )

        raw_content = response.choices[0].message.content
        processed_content = interpret_model_response(raw_content, screenshot_path)

        # Append full JSON as string to message history
        messages.append({"role": "assistant", "content": json.dumps(processed_content)})

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Skipping model. {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()

async def call_claude_3_with_ocr(messages, objective, model):
    if config.verbose:
        print("[call_claude_3_with_ocr]")

    try:
        time.sleep(1)
        client = config.initialize_anthropic()
        confirm_system_prompt(messages, objective, model)

        # Claude-specific: capture, resize, and encode image manually
        screenshot_path = os.path.join("screenshots", "screenshot.png")
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        capture_screen_with_cursor(screenshot_path)

        with open(screenshot_path, "rb") as img_file:
            img = Image.open(img_file)

            if img.mode == "RGBA":
                img = img.convert("RGB")

            aspect_ratio = img.width / img.height
            new_width = 2560
            new_height = int(new_width / aspect_ratio)

            if config.verbose:
                print("[call_claude_3_with_ocr] resizing image")

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_buffer = io.BytesIO()
            img_resized.save(img_buffer, format="JPEG", quality=85)
            img_buffer.seek(0)

            img_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

        user_prompt = (
            get_user_first_message_prompt()
            if len(messages) == 1
            else get_user_prompt()
        )

        vision_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_data,
                    },
                },
                {
                    "type": "text",
                    "text": user_prompt + "**REMEMBER** Only output json format, do not append any other text.",
                },
            ],
        }
        messages.append(vision_message)

        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=3000,
            system=messages[0]["content"],
            messages=messages[1:],
        )

        raw_content = response.content[0].text
        processed_content = interpret_model_response(raw_content, screenshot_path)

        messages.append({"role": "assistant", "content": json.dumps(processed_content)})

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Skipping model. {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()

async def call_gemini_pro_vision(messages, objective, model):
    if config.verbose:
        print("[call_gemini_pro_vision_with_ocr]")

    try:
        time.sleep(1)
        client = config.initialize_google()
        confirm_system_prompt(messages, objective, model)

        screenshot_path = os.path.join("screenshots", "screenshot.png")
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        capture_screen_with_cursor(screenshot_path)

        user_prompt = (
            get_user_first_message_prompt()
            if len(messages) == 1
            else get_user_prompt()
        )

        # Gemini expects system prompt + image directly in generate_content
        system_prompt = messages[0]["content"]
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        if config.verbose:
            print("[call_gemini_pro_vision_with_ocr] full prompt", full_prompt)

        response = client.generate_content([full_prompt, Image.open(screenshot_path)])

        raw_content = response.text
        processed_content = interpret_model_response(raw_content, screenshot_path)

        messages.append({"role": "assistant", "content": json.dumps(processed_content)})

        return processed_content

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Skipping model. {ANSI_RESET}"
        )
        if config.verbose:
            print("[Self-Operating Computer][Operate] error", e)
            traceback.print_exc()
        sys.exit()

async def call_ollama_llava(messages, objective, model):
    if config.verbose:
        print("[call_ollama_llava_with_ocr]")

    try:
        time.sleep(1)
        client = config.initialize_ollama()
        confirm_system_prompt(messages, objective, model)

        screenshot_path = os.path.join("screenshots", "screenshot.png")
        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

        capture_screen_with_cursor(screenshot_path)

        user_prompt = (
            get_user_first_message_prompt()
            if len(messages) == 1
            else get_user_prompt()
        )

        if config.verbose:
            print("[call_ollama_llava_with_ocr] user_prompt", user_prompt)

        vision_message = {
            "role": "user",
            "content": user_prompt,
            "images": [screenshot_path],
        }
        messages.append(vision_message)

        response = client.chat(
            model="llava",
            messages=messages,
        )

        # Ollama will try to re-load images if they're left in message history
        messages[-1]["images"] = None

        raw_content = response["message"]["content"].strip()
        processed_content = interpret_model_response(raw_content, screenshot_path)

        messages.append({"role": "assistant", "content": json.dumps(processed_content)})

        return processed_content

    except ollama.ResponseError as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_RED}[Operate] Couldn't connect to Ollama. Make sure 'llava' is pulled and the server is running.{ANSI_RESET}",
            e,
        )
        sys.exit()

    except Exception as e:
        print(
            f"{ANSI_GREEN}[Self-Operating Computer]{ANSI_BRIGHT_MAGENTA}[{model}] That did not work. Skipping model. {ANSI_RESET}",
            e,
        )
        if config.verbose:
            traceback.print_exc()
        sys.exit()