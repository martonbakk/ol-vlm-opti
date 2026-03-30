from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq
import logging 
import time 
import json


class JudgeWrapper(DeepEvalBaseLLM):
    def __init__(self, 
    model_name: str, 
    api_key: str, 
    temperature: float
    ):
        self.model_name = model_name
        self.temperature = temperature
        logging.info(f"Initializing JudgeWrapper with model {model_name} and temperature {temperature}")
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        max_retries = 5
        attempt = 0
        
        system_instruction = (
            "You are a helpful evaluation assistant. "
            "You must ALWAYS respond with a valid JSON object. "
            "The 'verdict' field must be strictly 'yes', 'no', or 'idk'. "
            "Do not use 'none'."
        )

        while attempt < max_retries:
            attempt += 1
            try:
                # Groq Free Tier limitjei miatt érdemes várni a kérések között
                # Ha 429-et kapsz, ezt emelheted, de 2-5 mp általában elég soros futtatásnál
                time.sleep(5) 
                
                logging.info(f"Groq hívás - {attempt}. próbálkozás...")
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                response_content = chat_completion.choices[0].message.content

                # 1. Ellenőrizzük, hogy érvényes JSON-e
                parsed_json = json.loads(response_content)
                
                # 2. 'verdict' javítása, ha 'none' csúszott be (a Pydantic miatt)
                if 'verdict' in parsed_json:
                    v = str(parsed_json['verdict']).lower()
                    if v not in ['yes', 'no', 'idk']:
                        logging.warning(f"Hibás verdict: {v}. Javítás 'no'-ra.")
                        parsed_json['verdict'] = 'no'
                
                logging.info(f"Sikeres válasz és valid JSON a(z) {attempt}. körben.")
                return json.dumps(parsed_json)

            except json.JSONDecodeError:
                logging.error(f"Kör {attempt}: A modell nem valid JSON-t küldött. Újrapróbálkozás...")
            except Exception as e:
                logging.error(f"Kör {attempt}: Hiba a hívás során: {e}")
                # Rate limit (429) esetén érdemes többet várni a következő kör előtt
                if "429" in str(e):
                    logging.info("Rate limit észlelve, 15 másodperc pihenő...")
                    time.sleep(15)

        # Ha elfogyott a 10 próbálkozás
        error_msg = "A modell 10 próbálkozás után sem adott valid JSON választ."
        logging.critical(error_msg)
        return json.dumps({"verdict": "idk", "reason": error_msg})

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return f"Groq {self.model_name}"