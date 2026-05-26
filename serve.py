import os
import urllib.request
from align import RejectionSamplingSimulator

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None # Handled dynamically

MODEL_URL = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

def download_model_if_missing():
    if not os.path.exists(MODEL_FILENAME):
        print(f"Downloading Micro-Model ({MODEL_FILENAME})...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        print("Download complete.")

class LocalMicroServer:
    def __init__(self):
        download_model_if_missing()
        if Llama is None:
            raise ImportError("llama-cpp-python is required to run the local model. Run `pip install llama-cpp-python`")
            
        # Zero-GPU constraints: 
        # use_mmap=True prevents loading entire model into RAM
        # use_mlock=False prevents locking model in RAM
        self.llm = Llama(
            model_path=MODEL_FILENAME,
            n_ctx=512,
            n_threads=4, # CPU friendly
            n_gpu_layers=0, # STRICT CPU
            use_mmap=True,
            use_mlock=False,
            verbose=False
        )
        
    def generate_json_with_rejection_sampling(self, prompt: str) -> str:
        """
        Generates output token-by-token, applying the alignment module 
        to reject tokens that break structural rules.
        """
        aligner = RejectionSamplingSimulator()
        
        full_prompt = f"Convert to JSON with keys 'urgency' and 'summary':\n{prompt}\n\n```json\n{{"
        
        generator = self.llm(
            full_prompt,
            max_tokens=64,
            stop=["```", "}"],
            stream=True
        )
        
        final_output = "{"
        for chunk in generator:
            token = chunk["choices"][0]["text"]
            if aligner.evaluate_token(token):
                final_output += token
            else:
                pass # Rejection sampling logic
                
        if not final_output.strip().endswith("}"):
            final_output += "}"
            
        return final_output

if __name__ == "__main__":
    server = LocalMicroServer()
    out = server.generate_json_with_rejection_sampling("My computer is on fire!")
    print(out)
