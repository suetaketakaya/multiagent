import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional
from config import OllamaConfig

class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.base_url = config.base_url
        self.timeout = aiohttp.ClientTimeout(total=config.timeout)
    
    async def check_models(self) -> Dict[str, Any]:
        """利用可能なモデル一覧を取得"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        raise Exception(f"Failed to get models: {response.status}")
            except Exception as e:
                raise Exception(f"Error connecting to Ollama: {e}")
    
    async def generate_response(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Ollamaモデルにプロンプトを送信してレスポンスを取得"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Generation failed: {response.status} - {error_text}")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error: {e}")
            except asyncio.TimeoutError:
                raise Exception("Request timeout")
            except Exception as e:
                raise Exception(f"Unexpected error: {e}")
    
    async def test_connection(self) -> bool:
        """Ollamaサーバーとの接続をテスト"""
        try:
            models = await self.check_models()
            print(f"✅ Ollama接続成功: {len(models.get('models', []))}個のモデルが利用可能")
            return True
        except Exception as e:
            print(f"❌ Ollama接続失敗: {e}")
            return False 