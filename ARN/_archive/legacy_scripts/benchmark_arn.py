import os
import time
import torch
import tracemalloc
import numpy as np
from config import ARNconfig
from tokenizer_utils import TOKENIZER
from model import AetherRoutingNetwork
from inference import predict

# Rigorous N=100 Out-of-Domain Conversational / Chit-Chat / Ambiguous Test Prompts
IRRELEVANT_PROMPTS = [
    # General Greetings & Politeness (15)
    "Hello there!", "How's the weather today?", "Tell me a joke.", "Who is the president of the United States?",
    "Can you explain quantum physics?", "Let's grab a coffee sometime.", "Nice to meet you.", "What is your name?",
    "Goodbye!", "Just thinking out loud about some coding ideas.", "Good morning how are you doing today?",
    "Thanks a lot for your help!", "Have a wonderful day ahead", "What are your favorite movies?", "Hey assistant how is it going?",
    
    # Casual Chit-Chat & Questions (25)
    "Why is the sky blue?", "Do you know who won the World Cup in 2022?", "What is the capital of France?",
    "Can dogs eat chocolate?", "How far is the moon from Earth?", "Tell me a story about a dragon.",
    "What is the meaning of life?", "Do you like pizza or burgers better?", "How do airplanes stay in the air?",
    "What is your opinion on artificial intelligence?", "Can you write a poem about autumn?", "Who wrote Hamlet?",
    "Is it going to rain this weekend?", "What is the speed of light?", "How do batteries store electricity?",
    "Tell me an interesting science fact.", "What are the primary colors?", "How do trees produce oxygen?",
    "What time do grocery stores usually close?", "Why do cats purr when they are happy?", "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?", "How many continents are there?", "What is the difference between fruit and vegetable?",
    "Can you translate hello into Spanish?",
    
    # Statement / Thought Fragment Inputs (25)
    "I really enjoy listening to classical music in the evening.", "Debugging code can be quite challenging sometimes.",
    "Python is one of the most popular programming languages today.", "Machine learning requires a lot of high quality data.",
    "I need to drink more water throughout the day.", "The sunset yesterday was absolutely breathtaking.",
    "Virtual reality technology has come a long way in recent years.", "Reading books helps improve vocabulary and focus.",
    "Exercise is essential for maintaining good physical health.", "Automated testing saves a lot of time in software development.",
    "I am thinking about learning how to play the acoustic guitar.", "Coffee tastes great on a chilly winter morning.",
    "Space exploration has always fascinated human beings.", "Clean architecture makes software codebases much easier to maintain.",
    "Open source software powers most of the internet infrastructure.", "Electric vehicles are becoming more common every year.",
    "Typography plays a crucial role in user interface design.", "I love the smell of fresh rain on dry earth.",
    "Cooking your own meals is healthier and more affordable.", "Consistency is key to mastering any new skill.",
    "Digital privacy is an important topic in modern technology.", "Traveling exposes you to new cultures and perspectives.",
    "Recursion is a programming concept where a function calls itself.", "A balanced diet provides steady energy throughout the day.",
    "Mindfulness practice can help reduce daily stress levels.",
    
    # Technical & Math Enquiries without Tool Intent (20)
    "What is the formula for calculating the area of a circle?", "How does a binary search algorithm work?",
    "What is the difference between TCP and UDP networking protocols?", "Can you explain what a REST API is?",
    "What is the complexity of quicksort in the worst case?", "How does garbage collection work in Java?",
    "What is the Pythagorean theorem?", "What is a hash table and how does collision resolution work?",
    "What is the purpose of Docker containers in DevOps?", "How does public key cryptography work?",
    "What is a database index and why is it used?", "What is the difference between process and thread in OS?",
    "How does gradient descent optimize neural network weights?", "What is the derivative of x squared?",
    "What is a pull request in Git version control?", "How does SSL TLS encryption protect web traffic?",
    "What is the difference between synchronous and asynchronous execution?", "What is a closure in JavaScript?",
    "What is the CAP theorem in distributed systems?", "How does a compiler differ from an interpreter?",
    
    # Conversational Commands without Actionable Tools (15)
    "Be quiet for a second.", "Think deeply about this problem.", "Hold on a moment while I check something.",
    "Never mind about that.", "Forget what I just said.", "Listen to me carefully.",
    "Pay attention to this detail.", "Take your time.", "Don't worry about it.",
    "Keep that in mind for later.", "Let me know what you think.", "Ponder this question for a bit.",
    "Consider the following scenario.", "Imagine a world without internet.", "Stay tuned for further updates."
]

def main():
    print("=" * 65)
    print("ARN HARDWARE & SAFETY EVALUATION (N=100 False Positive Benchmark)")
    print("=" * 65)
    
    model_path = "checkpoints/arn_best_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train.py first.")
        return
        
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[+] Model Weights File Size: {file_size_mb:.2f} MB")
    
    config = ARNconfig()
    tracemalloc.start()
    
    model = AetherRoutingNetwork(config)
    try:
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    except Exception:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    peak_mem_mb = peak_mem / (1024 * 1024)
    print(f"[+] Peak Memory Usage (Model Load): {peak_mem_mb:.2f} MB")
    
    tracemalloc.reset_peak()
    
    sample_text = "send a WhatsApp message to Rahul saying I am late"
    
    for _ in range(10):
        _ = predict(sample_text, model, TOKENIZER, config)
        
    latencies = []
    iterations = 100
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = predict(sample_text, model, TOKENIZER, config)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    _, peak_infer_mem = tracemalloc.get_traced_memory()
    peak_infer_mem_mb = peak_infer_mem / (1024 * 1024)
    
    print(f"[+] Average Inference Latency: {avg_latency:.2f} ms")
    print(f"    - 95th Percentile (p95):  {p95_latency:.2f} ms")
    print(f"    - 99th Percentile (p99):  {p99_latency:.2f} ms")
    print(f"[+] Peak Memory Usage (Inference): {peak_infer_mem_mb:.2f} MB")
    
    tracemalloc.stop()
    
    # ── Evaluate False Positive Rejection across N=100 Out-of-Domain Samples ──
    print("\n" + "=" * 65)
    print(f"EVALUATING FALSE POSITIVE REJECTION (N={len(IRRELEVANT_PROMPTS)} CONVERSATIONAL PROMPTS)")
    print("=" * 65)
    
    rejected_correctly = 0
    total_prompts = len(IRRELEVANT_PROMPTS)
    failures = []
    
    for idx, prompt in enumerate(IRRELEVANT_PROMPTS):
        res = predict(prompt, model, TOKENIZER, config)
        pred_tools = res["tools"]
        pred_slots = res["arguments"]
        
        is_rejected = (len(pred_tools) == 0)
        if is_rejected:
            rejected_correctly += 1
        else:
            failures.append((idx+1, prompt, pred_tools, pred_slots))
            
    rejection_rate = (rejected_correctly / total_prompts) * 100
    print(f"Correctly Rejected: {rejected_correctly} / {total_prompts}")
    print(f"False Positive Rejection Success Rate: {rejection_rate:.1f}%")
    
    if failures:
        print(f"\n[False Positive Triggers ({len(failures)})]:")
        for f_idx, p_text, p_tools, p_slots in failures:
            print(f"  - Sample {f_idx}: {repr(p_text[:45])} -> Triggered {p_tools}")
    else:
        print("  --> ALL 100 OUT-OF-DOMAIN PROMPTS CORRECTLY REJECTED! [100% PERFECT SAFETY]")
    print("=" * 65)

if __name__ == "__main__":
    main()
