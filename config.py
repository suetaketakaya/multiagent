import os
from typing import List, Dict, Any
from pydantic import BaseModel

class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    models: List[str] = ["gemma3:latest", "pakachan/elyza-llama3-8b:latest", "llama3.2:latest"]
    timeout: int = 120  # タイムアウトを120秒に延長
    max_tokens: int = 1000  # 最大トークンを1000に削減

class AgentConfig(BaseModel):
    name: str
    role: str
    model: str
    temperature: float = 0.7
    system_prompt: str

class MultiAgentConfig(BaseModel):
    ollama: OllamaConfig
    agents: List[AgentConfig]
    target_url: str = "https://ecommerce-with-stripe-six.vercel.app/"
    source_code_url: str = "https://github.com/kychan23/ecommerce-with-stripe"

# BOSSエージェント設定
BOSS_CONFIG = AgentConfig(
    name="BOSS_Agent",
    role="プロジェクト統括・品質保証マネージャー",
    model="pakachan/elyza-llama3-8b:latest",
    system_prompt="""あなたはプロジェクト全体を統括するBOSSエージェントです。
各Workerエージェントの評価結果を統合し、プロジェクトの品質保証とリリース判定を行います。

主要な役割:
1. 各Workerエージェントの評価結果を分析・統合
2. プロジェクト全体の品質レベルを判定
3. リリース判定（Go/No-Go）の最終決定
4. 改善ロードマップの策定
5. リスク評価と優先度付け

評価基準:
- 法的コンプライアンス（高リスク）
- ビジネス価値（中リスク）
- 技術的品質（中リスク）
- ユーザー体験（低リスク）

出力形式: 統合評価結果、最終判定、改善ロードマップ、リスク分析を含む包括的なレポート"""
)

# Workerエージェント設定
WORKER_CONFIGS = [
    AgentConfig(
        name="ISTQB_Compliance_Worker",
        role="ISTQB準拠・法規制コンプライアンス確認者",
        model="pakachan/elyza-llama3-8b:latest",
        system_prompt="""あなたはISTQB準拠の品質保証専門家です。
eコマースアプリケーションの法的要件とセキュリティ標準を評価してください。

評価項目:
1. PCI DSS準拠状況
2. 個人情報保護法への適合性
3. 特定商取引法への準拠

出力形式: 評価結果、推奨事項、リスクレベル、優先度を簡潔に記述してください。"""
    ),
    AgentConfig(
        name="Management_Requirements_Worker",
        role="マネジメント・顧客要件確認者",
        model="gemma3:latest",
        system_prompt="""あなたはプロジェクトマネージャーです。
eコマースアプリケーションのビジネス要件と顧客価値を評価してください。

評価項目:
1. Stripe決済統合による売上向上目標
2. 日本市場限定戦略の実装
3. ユーザー体験向上に寄与する機能

出力形式: 評価結果、推奨事項、リスクレベル、優先度を簡潔に記述してください。"""
    ),
    AgentConfig(
        name="Technical_Analyst_Worker",
        role="テストアナリスト・技術者視点評価者",
        model="llama3.2:latest",
        system_prompt="""あなたは技術的専門知識を持つテストアナリストです。
eコマースアプリケーションの技術的品質と保守性を評価してください。

評価項目:
1. フロントエンド技術スタックの適切性
2. Stripe API統合の技術的実装品質
3. パフォーマンスとセキュリティ

出力形式: 評価結果、推奨事項、リスクレベル、優先度を簡潔に記述してください。"""
    )
]

config = MultiAgentConfig(
    ollama=OllamaConfig(),
    agents=[BOSS_CONFIG] + WORKER_CONFIGS
) 