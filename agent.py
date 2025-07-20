import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
from ollama_client import OllamaClient
from config import AgentConfig

@dataclass
class AgentResult:
    agent_name: str
    role: str
    evaluation: str
    recommendations: List[str]
    risk_level: str
    priority: str

@dataclass
class BossResult:
    agent_name: str
    role: str
    overall_evaluation: str
    final_decision: str
    risk_analysis: str
    improvement_roadmap: List[str]
    worker_summary: Dict[str, Any]

class BaseAgent:
    def __init__(self, config: AgentConfig, ollama_client: OllamaClient):
        self.config = config
        self.client = ollama_client
        self.name = config.name
        self.role = config.role
        self.model = config.model
        self.system_prompt = config.system_prompt
        self.temperature = config.temperature
    
    async def evaluate(self, target_info: Dict[str, Any]) -> AgentResult:
        """エージェント固有の評価を実行"""
        raise NotImplementedError
    
    def create_prompt(self, target_info: Dict[str, Any]) -> str:
        """評価用のプロンプトを作成"""
        return f"""対象アプリケーション: {target_info.get('url', 'N/A')}
概要: {target_info.get('description', 'N/A')}

あなたの役割: {self.role}

上記のアプリケーションについて評価し、以下の形式で回答してください：

## 評価結果
[専門分野での評価]

## 推奨事項
- [改善提案1]
- [改善提案2]

## リスクレベル
[高/中/低] - [理由]

## 優先度
[高/中/低] - [理由]"""

    def _parse_worker_response(self, response: str) -> AgentResult:
        """Workerエージェント共通のレスポンス解析"""
        lines = response.split('\n')
        evaluation = ""
        recommendations = []
        risk_level = "中"
        priority = "中"
        
        in_evaluation = False
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if "## 評価結果" in line:
                in_evaluation = True
                in_recommendations = False
                continue
            elif "## 推奨事項" in line:
                in_evaluation = False
                in_recommendations = True
                continue
            elif "## リスクレベル" in line:
                in_recommendations = False
                if "高" in line:
                    risk_level = "高"
                elif "低" in line:
                    risk_level = "低"
                continue
            elif "## 優先度" in line:
                if "高" in line:
                    priority = "高"
                elif "低" in line:
                    priority = "低"
                continue
            
            if in_evaluation and line:
                evaluation += line + "\n"
            elif in_recommendations and line.startswith("-"):
                recommendations.append(line[1:].strip())
        
        if not evaluation.strip():
            evaluation = response
        
        if not recommendations:
            recommendations = ["詳細な評価が必要です"]
        
        return AgentResult(
            agent_name=self.name,
            role=self.role,
            evaluation=evaluation.strip(),
            recommendations=recommendations,
            risk_level=risk_level,
            priority=priority
        )

class BossAgent(BaseAgent):
    async def evaluate_workers(self, worker_results: List[AgentResult], target_info: Dict[str, Any]) -> BossResult:
        """Workerエージェントの結果を統合して最終評価を実行"""
        
        # Worker結果のサマリーを作成
        worker_summary = {}
        for result in worker_results:
            worker_summary[result.agent_name] = {
                "role": result.role,
                "evaluation": result.evaluation,
                "recommendations": result.recommendations,
                "risk_level": result.risk_level,
                "priority": result.priority
            }
        
        # BOSS用のプロンプトを作成
        prompt = self._create_boss_prompt(worker_results, target_info)
        
        try:
            response = await self.client.generate_response(
                model=self.model,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature
            )
            
            return self._parse_boss_response(response, worker_summary)
        except Exception as e:
            return BossResult(
                agent_name=self.name,
                role=self.role,
                overall_evaluation=f"BOSS評価中にエラーが発生しました: {str(e)}",
                final_decision="No-Go",
                risk_analysis="エラーにより評価できません",
                improvement_roadmap=["システムエラーの解決が必要です"],
                worker_summary=worker_summary
            )
    
    def _create_boss_prompt(self, worker_results: List[AgentResult], target_info: Dict[str, Any]) -> str:
        """BOSS用のプロンプトを作成"""
        worker_summary = "\n\n".join([
            f"## {result.agent_name} ({result.role})\n"
            f"評価: {result.evaluation}\n"
            f"推奨事項: {', '.join(result.recommendations)}\n"
            f"リスクレベル: {result.risk_level}\n"
            f"優先度: {result.priority}"
            for result in worker_results
        ])
        
        return f"""対象アプリケーション: {target_info.get('url', 'N/A')}
概要: {target_info.get('description', 'N/A')}

## Workerエージェントの評価結果

{worker_summary}

上記のWorkerエージェントの評価結果を統合し、以下の形式で最終評価を行ってください：

## 統合評価結果
[プロジェクト全体の品質評価]

## 最終判定
[Go/No-Go] - [理由]

## リスク分析
[高/中/低リスクの詳細分析]

## 改善ロードマップ
- [短期改善項目1]
- [中期改善項目1]
- [長期改善項目1]"""

    def _parse_boss_response(self, response: str, worker_summary: Dict[str, Any]) -> BossResult:
        """BOSSのレスポンスを解析"""
        lines = response.split('\n')
        overall_evaluation = ""
        final_decision = "No-Go"
        risk_analysis = ""
        improvement_roadmap = []
        
        in_evaluation = False
        in_decision = False
        in_risk = False
        in_roadmap = False
        
        for line in lines:
            line = line.strip()
            if "## 統合評価結果" in line:
                in_evaluation = True
                in_decision = False
                in_risk = False
                in_roadmap = False
                continue
            elif "## 最終判定" in line:
                in_evaluation = False
                in_decision = True
                in_risk = False
                in_roadmap = False
                continue
            elif "## リスク分析" in line:
                in_evaluation = False
                in_decision = False
                in_risk = True
                in_roadmap = False
                continue
            elif "## 改善ロードマップ" in line:
                in_evaluation = False
                in_decision = False
                in_risk = False
                in_roadmap = True
                continue
            
            if in_evaluation and line:
                overall_evaluation += line + "\n"
            elif in_decision and line:
                if "Go" in line:
                    final_decision = "Go"
                elif "No-Go" in line:
                    final_decision = "No-Go"
            elif in_risk and line:
                risk_analysis += line + "\n"
            elif in_roadmap and line.startswith("-"):
                improvement_roadmap.append(line[1:].strip())
        
        # デフォルト値の設定
        if not overall_evaluation.strip():
            overall_evaluation = "Workerエージェントの評価を統合した結果、プロジェクトの品質を総合的に評価しました。"
        if not risk_analysis.strip():
            risk_analysis = "中リスク - 改善が必要だがリリースは可能"
        if not improvement_roadmap:
            improvement_roadmap = ["短期: セキュリティ強化", "中期: UI/UX改善", "長期: 機能拡張"]
        
        return BossResult(
            agent_name=self.name,
            role=self.role,
            overall_evaluation=overall_evaluation.strip(),
            final_decision=final_decision,
            risk_analysis=risk_analysis.strip(),
            improvement_roadmap=improvement_roadmap,
            worker_summary=worker_summary
        )

class ISTQBComplianceWorker(BaseAgent):
    async def evaluate(self, target_info: Dict[str, Any]) -> AgentResult:
        prompt = self.create_prompt(target_info)
        
        try:
            response = await self.client.generate_response(
                model=self.model,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature
            )
            
            return self._parse_worker_response(response)
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                role=self.role,
                evaluation=f"評価中にエラーが発生しました: {str(e)}",
                recommendations=["システムエラーのため評価を再実行してください", f"エラー詳細: {str(e)}"],
                risk_level="不明",
                priority="高"
            )

class ManagementRequirementsWorker(BaseAgent):
    async def evaluate(self, target_info: Dict[str, Any]) -> AgentResult:
        prompt = self.create_prompt(target_info)
        
        try:
            response = await self.client.generate_response(
                model=self.model,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature
            )
            
            return self._parse_worker_response(response)
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                role=self.role,
                evaluation=f"評価中にエラーが発生しました: {str(e)}",
                recommendations=["システムエラーのため評価を再実行してください", f"エラー詳細: {str(e)}"],
                risk_level="不明",
                priority="高"
            )

class TechnicalAnalystWorker(BaseAgent):
    async def evaluate(self, target_info: Dict[str, Any]) -> AgentResult:
        prompt = self.create_prompt(target_info)
        
        try:
            response = await self.client.generate_response(
                model=self.model,
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=self.temperature
            )
            
            return self._parse_worker_response(response)
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                role=self.role,
                evaluation=f"評価中にエラーが発生しました: {str(e)}",
                recommendations=["システムエラーのため評価を再実行してください", f"エラー詳細: {str(e)}"],
                risk_level="不明",
                priority="高"
            )

def create_agent(config: AgentConfig, ollama_client: OllamaClient) -> BaseAgent:
    """エージェント設定に基づいて適切なエージェントインスタンスを作成"""
    if "BOSS" in config.name:
        return BossAgent(config, ollama_client)
    elif "ISTQB" in config.name:
        return ISTQBComplianceWorker(config, ollama_client)
    elif "Management" in config.name:
        return ManagementRequirementsWorker(config, ollama_client)
    elif "Technical" in config.name:
        return TechnicalAnalystWorker(config, ollama_client)
    else:
        return BaseAgent(config, ollama_client) 