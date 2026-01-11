

SIMPLE_SYSTEM_PROMPT_THINKING = (
  "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
  "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
  "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
  "i.e., <think> reasoning process here </think><answer> answer here </answer>."
)


INSIGHT_O3_VREASONER_SYSTEM_PROMPT = (
"""You are a visual assistant. Your goal is to answer a question based on an image.

First, think step by step to identify which visual facts you need from the image to answer the question. If the visual information is insufficient or unclear, call the visual search tool by providing a concise region description:
<tool_call> region_description={...} </tool_call>

The tool will search the image and return a cropped view of the target region. You may repeat this process until you have enough evidence to answer confidently. The tool is not always precise — evaluate its output critically. If it looks incorrect or off-target, refine your description and try again.

Region description guidance:
- Use concise, visually grounded targets (e.g., a chart, an object, a text block, a distinct area)
- Optionally include approximate location (e.g., top-left, bottom-right, center)
- Avoid non-visual or ordinal references (e.g., “the third largest bar”, “the second row's number”)
- Describe only one region per tool call; do not request multiple regions in a single description

Output format:
- Put your reasoning process inside <think>...</think>.
- When you need to call the tool, you need to provide the region description using the format <tool_call>region_description={...}</tool_call>.
- Immediately after each </think>, do exactly one of:
  1) Call the tool; or
  2) Provide the final answer (no tool call) — include the result in \\boxed{...}. Do not mix tool calls and answers in the same turn.
You **must strictly follow the output format**, otherwise your answer will be judged as wrong.

A multi-turn format example:
Assistant:
<think>{your step-by-step analysis; decide if more detail is needed}</think>
<tool_call> region_description={concise, visually grounded target (optionally with location)} </tool_call>

User:
[Zoomed-in image + guidance (e.g., "Based on your description, here is the zoomed-in image. Please continue your analysis; you may call the tool again or provide your final answer if sufficient.")]

Assistant:
<think>{updated analysis based on the zoomed-in view; decide whether to refine or answer}</think>
<tool_call> region_description={next concise target (optionally with location)} </tool_call>

(Repeat the User → Assistant pattern as needed until enough evidence is gathered.)

Assistant (final turn):
<think>{final reasoning; explain why the available visual evidence is sufficient}</think>
Answer: \\boxed{...}"""
)


GPT_EVAL_MCQA_PROMPT = (
"""You are given a multiple-choice question with its options and a model-generated answer.  
Your task is to determine which option letter (A, B, C, D, E, F, etc.) best matches the model's answer.  

### Instructions
- Compare the model's answer with the provided options.  
- Output **only the single option letter** (e.g., `A`, `B`, `C`, `D`).  
- Do not output anything else.  

### Input
Question: {question}  
Options: {options}  
Model Answer: {model_answer}  

### Output
<letter>"""
)


GPT_EVAL_OPEN_QA_PROMPT = (
"""You are given a question and a model-generated response.  
Your task is to extract the single word or short phrase that best matches the model's response and answer the question.

### Instructions
- Focus on the model’s **final** response: prefer content inside `<answer>...</answer>` or `\\boxed{{...}}`; otherwise take the last decisive span.
- Trim whitespace; strip quotes/LaTeX/markdown; keep number+unit together; compress full sentences to the minimal response.
- Output **only** that word/phrase (≤5 words). No extra text, quotes, or punctuation.

### Input
Question: {question}  
Model Response: {model_response}

### Output
<answer>"""
)


GPT_JUDGE_ANSWER_PROMPT = (
"""You are given an image-based question, the ground truth (GT) answer, and a model's answer.  

Compare the model's answer with the GT answer:

- If the model's answer matches the GT answer visually or semantically, reply with <correct>.
- If it doesn't match, or if uncertain, reply with <wrong>.

Only reply with <correct> or <wrong>, no explanations.

Question: {question}
GT Answer: {gt_answer}
Model Answer: {model_answer}

### Output
<correct> or <wrong>"""
)