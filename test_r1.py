import asyncio


# -*- coding: utf-8 -*-

import openai
from openai import AsyncOpenAI
import re
import json
import os
from retrying import retry
import requests
# from key import (
#     client_key,
#     client_url,
#     client_qwen_key,
#     client_qwen_url,
#     openai_key,
#     openai_org,
# )


class Generator:
    def __init__(self, request_times=3):

        client_url = "https://ark.cn-beijing.volces.com/api/v3"
        client_key = "4d553e3a-752e-44a3-b6a0-889c3a8c54d0"
        
        client_qwen_key = ""
        client_qwen_url = ""
        openai_key = ""
        openai_org = ""
        self.request_times = request_times
        self.CLIENT = AsyncOpenAI(api_key=client_key, base_url=client_url)
        self.QWEN_CLIENT = AsyncOpenAI(
            base_url=client_qwen_url, api_key=client_qwen_key
        )
        self.OPENAI_CLIENT = AsyncOpenAI(api_key=openai_key, organization=openai_org)
        self.model_map = {
            "Doubao-lite-32k": "ep-20241017105528-gm84f",
            "QwQ-32B-Preview": "8085",
            "Qwen2.5-14B-Instruct-1M": "8083",
            "DeepSeek-R1-Distill-Qwen-32B": "8082",
            "DeepSeek-R1-Distill-Qwen-7B": "8084",
            "Doubao-pro-32k": "ep-20241019064955-dj94p",
            "OpenAI-o1": "o1-2024-12-17",
            "deepseek-v3": "ep-20250207151339-9688m",
            "OpenAI-o1-mini": "o1-mini-2024-09-12",
            "OpenAI-o3-mini": "o3-mini-2025-01-31",
            "OpenAI-gpt-4o": "gpt-4o-2024-08-06",
            "OpenAI-gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "deepseek-r1-qwen-32b": "ep-20250207151459-4bqpx",
            "deepseek-r1-qwen-7b": "ep-20250207151415-bvzrr",
            "deepseek-r1": "ep-20250207151258-v586g",
            "Qwen-plus": "qwen-plus",
            "Qwen-max": "qwen-max",
        }
        # use for rag
        self.rag_url = 'http://47.101.166.200:8086/search_qa'
        self.header = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.search_k = 5


    @retry(stop_max_attempt_number=3)
    async def request_llm(
        self,
        model,
        request_message,
        save_path
    ):
        print("requesting... output: ", save_path)
        success = False
        for i in range(self.request_times):
            if success:
                break

            try:
                if model in [
                    "OpenAI-gpt-4o",
                    "OpenAI-gpt-4o-mini",
                    "OpenAI-o3-mini",
                    "OpenAI-o1-mini",
                    "OpenAI-o1",
                ]:
                    responses = await self.OPENAI_CLIENT.chat.completions.create(
                        model=self.model_map[model],
                        messages=request_message,
                    )
                    responses = responses.choices[0].message.content

                elif model in [
                    "Doubao-lite-32k",
                    "Doubao-pro-32k",
                    "deepseek-r1-qwen-32b",
                    "deepseek-r1-qwen-7b",
                    "deepseek-v3",
                    "deepseek-r1",
                ]:
                    responses = await self.CLIENT.chat.completions.create(
                        model=self.model_map[model],
                        messages=request_message,
                        temperature=0.5,
                    )
                    responses = responses.json()

                elif model in ["Qwen-plus", "Qwen-max"]:
                    responses = await self.QWEN_CLIENT.chat.completions.create(
                        model=self.model_map[model],
                        messages=request_message,
                        temperature=0.1,
                        top_p=0.1,
                    )
                    responses = responses.choices[0].message.content

                else:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": "Basic YWdlbnRfd3djOkxhaUAyMDI0ISEh",
                    }
                    prompts = {
                        "model": model,
                        "stream": False,
                        "messages": request_message,
                    }
                    model_url = "http://47.101.166.200:{}/v1/chat/completions"
                    responses = requests.post(
                        model_url.format(self.model_map[model]),
                        headers=headers,
                        data=json.dumps(prompts),
                    )
                    responses = responses.json()
                    # responses = responses["choices"][0]["message"]["content"]

                success = True
            except:
                pass

        if not success:
            return {
                "status_message": "请求失败",
                "answer": "",
            }
        else:
            with open(save_path, 'w') as fp:
                json.dump(json.loads(responses), fp, sort_keys=True, indent=4, ensure_ascii=False)
            return  {
                "status_message": "请求成功",
                "answer": responses,
            }
            
# Make sure the Generator class and its imports are in scope
# from your_module import Generator  # Uncomment and modify if needed

# import asyncio
# from test_r1 import Generator

# async def main():
#     gen = Generator()

#     request_message = [
#         {
#             "role": "user",
#             "content": """
#             Solve the problem, the problem contains a image but instead of the image you are given a detailed description of the image, return only the final result in one word.

#             ## problem

#             Four people can be seated at a square table. How many people at most could be seated if we pushed four tables of this kind together in one row?

#             ## image description

#             The image appears to be a line drawing of a table setting for four people. The table is rectangular and has four chairs, one at each corner. On the table, there are four place settings, each consisting of a plate, a bowl, a pair of chopsticks, and a spoon. The plates are circular with a small crescent shape cut out from the top edge. The bowls are also circular but slightly smaller than the plates. The chopsticks are placed parallel to each other on the left side of each plate, and the spoons are placed on the right side of each plate. The overall layout is symmetrical, with each place setting mirroring the others. The drawing is simple and uses clean lines without any shading or color.
#             """
#         }
#     ]

#     model_name = "deepseek-r1"

#     # Create tasks for all calls
#     task1 = asyncio.create_task(gen.request_llm(model_name, request_message, "./tmp.json"))
#     task2 = asyncio.create_task(gen.request_llm(model_name, request_message, "./tmp1.json"))
#     task3 = asyncio.create_task(gen.request_llm(model_name, request_message, "./tmp2.json"))
#     task4 = asyncio.create_task(gen.request_llm(model_name, request_message, "./tmp3.json"))

#     # Wait for all tasks concurrently
#     responses = await asyncio.gather(task1, task2, task3, task4)
#     # print("LLM Responses:", responses)

# await main()