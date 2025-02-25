from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenVLWorker:
    def __init__(self,):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "/home/agent_h/data/llms/Qwen2-VL-72B-Instruct",
            torch_dtype="auto",
            load_in_8bit=True,
            device_map="auto")
            # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            #     "Qwen/Qwen2.5-VL-72B-Instruct",
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     device_map="auto",
            # )

        self.processor = AutoProcessor.from_pretrained("/home/agent_h/data/llms/Qwen2-VL-72B-Instruct",min_pixels=896*896,max_pixels=1792*1792)

    def discribe_image(self,img_path):
        print(img_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text",
                     "text": "please discribe the image in detail to a blind person, include as much detail as possible."},
                ],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
