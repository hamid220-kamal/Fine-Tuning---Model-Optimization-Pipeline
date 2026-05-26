import asyncio
import json
import random
from dataclasses import dataclass, asdict
from typing import AsyncGenerator

@dataclass
class StructuredData:
    user_id: int
    complaint_summary: str
    urgency_level: str
    requires_followup: bool

class TeacherModelMock:
    """Mocks a massive Teacher Model turning messy text into strict JSON."""
    
    async def process_transcript(self, raw_text: str) -> str:
        # Simulate API latency
        await asyncio.sleep(random.uniform(0.01, 0.05))
        
        # Process the input intelligently
        urgency = "high" if "urgent" in raw_text.lower() or "broken" in raw_text.lower() else "low"
        
        data = StructuredData(
            user_id=random.randint(1000, 9999),
            complaint_summary=f"Processed: {raw_text[:20]}...",
            urgency_level=urgency,
            requires_followup=urgency == "high"
        )
        
        # Return strict, unescaped JSON string
        return json.dumps(asdict(data))

async def synthetic_data_generator(transcripts: list[str]) -> AsyncGenerator[str, None]:
    """Yields perfect JSON objects continuously without blowing up RAM."""
    teacher = TeacherModelMock()
    for transcript in transcripts:
        result = await teacher.process_transcript(transcript)
        yield result

async def demo_distill():
    messy_inputs = [
        "My phone screen is broken and it's urgent!",
        "The battery drains too fast",
        "I need a refund asap, completely broken",
        "How do I change my password?"
    ] * 25  # 100 items
    
    print("Starting Distillation Engine...")
    async for structured_json in synthetic_data_generator(messy_inputs):
        pass # In a real pipeline, we'd stream this directly to a file or trainer
    print("Distillation Complete. Zero RAM spikes.")

if __name__ == "__main__":
    asyncio.run(demo_distill())
