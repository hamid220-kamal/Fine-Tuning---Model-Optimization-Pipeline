import json

class RejectionSamplingSimulator:
    """
    Simulates token-by-token log-probit biasing by validating the structural
    integrity of the generation buffer. If a token violates the JSON schema,
    it rejects it and forces a retry.
    """
    
    def __init__(self):
        self.buffer = ""
    
    def evaluate_token(self, new_token: str) -> bool:
        """
        Validates if adding `new_token` to the buffer keeps it structurally viable.
        Returns True if valid (accept token), False if invalid (reject token).
        """
        temp_buffer = self.buffer + new_token
        
        # A simple heuristic check for JSON structure.
        open_braces = temp_buffer.count('{')
        close_braces = temp_buffer.count('}')
        
        if close_braces > open_braces:
            return False # Structural violation: closing brace before opening
        
        # Accept the token
        self.buffer = temp_buffer
        return True
        
    def validate_final_output(self, generated_text: str) -> bool:
        try:
            json.loads(generated_text)
            return True
        except json.JSONDecodeError:
            return False

def demo_align():
    aligner = RejectionSamplingSimulator()
    tokens = ["{\n", '  "status": ', '"ok"\n', "}"]
    
    for token in tokens:
        if aligner.evaluate_token(token):
            pass # token accepted
        else:
            print(f"Rejected token: {token}")
            
    print(f"Final valid? {aligner.validate_final_output(aligner.buffer)}")

if __name__ == "__main__":
    demo_align()
