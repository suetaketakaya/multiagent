import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import config, BOSS_CONFIG, WORKER_CONFIGS
from ollama_client import OllamaClient
from agent import create_agent, AgentResult, BossResult

console = Console()

class MultiAgentSystem:
    def __init__(self):
        self.ollama_client = OllamaClient(config.ollama)
        self.boss_agent = None
        self.worker_agents = []
        self.worker_results = []
        self.boss_result = None
        
        # BOSSエージェントを初期化
        self.boss_agent = create_agent(BOSS_CONFIG, self.ollama_client)
        
        # Workerエージェントを初期化
        for worker_config in WORKER_CONFIGS:
            worker = create_agent(worker_config, self.ollama_client)
            self.worker_agents.append(worker)
    
    async def test_connection(self) -> bool:
        """Ollamaサーバーとの接続をテスト"""
        return await self.ollama_client.test_connection()
    
    async def run_evaluation(self, target_info: Dict[str, Any]) -> BossResult:
        """BOSS-Worker構造で評価を実行"""
        console.print(Panel.fit(
            f"[bold blue]BOSS-Worker評価システム開始[/bold blue]\n"
            f"対象: {target_info.get('url', 'N/A')}\n"
            f"BOSS: 1名\n"
            f"Worker: {len(self.worker_agents)}名",
            border_style="blue"
        ))
        
        # Step 1: Workerエージェントによる並行評価
        console.print("\n[bold]🔧 Step 1: Workerエージェントによる専門評価[/bold]")
        worker_results = await self._run_worker_evaluation(target_info)
        
        if not worker_results:
            console.print("[red]❌ Workerエージェントの評価が失敗しました[/red]")
            return None
        
        # Step 2: BOSSエージェントによる統合評価
        console.print("\n[bold]👑 Step 2: BOSSエージェントによる統合評価[/bold]")
        boss_result = await self._run_boss_evaluation(worker_results, target_info)
        
        self.worker_results = worker_results
        self.boss_result = boss_result
        
        return boss_result
    
    async def _run_worker_evaluation(self, target_info: Dict[str, Any]) -> List[AgentResult]:
        """Workerエージェントによる並行評価を実行"""
        tasks = []
        for worker in self.worker_agents:
            task = asyncio.create_task(worker.evaluate(target_info))
            tasks.append(task)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Workerエージェント評価中...", total=len(tasks))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            worker_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    console.print(f"[red]Worker {self.worker_agents[i].name} でエラー: {result}[/red]")
                else:
                    worker_results.append(result)
                progress.advance(task)
        
        return worker_results
    
    async def _run_boss_evaluation(self, worker_results: List[AgentResult], target_info: Dict[str, Any]) -> BossResult:
        """BOSSエージェントによる統合評価を実行"""
        try:
            boss_result = await self.boss_agent.evaluate_workers(worker_results, target_info)
            return boss_result
        except Exception as e:
            console.print(f"[red]BOSSエージェントでエラー: {e}[/red]")
            return None
    
    def generate_report(self) -> Dict[str, Any]:
        """評価結果から統合レポートを生成"""
        if not self.boss_result:
            return {"error": "BOSS評価結果がありません"}
        
        # Worker結果の分類
        high_priority_workers = []
        medium_priority_workers = []
        low_priority_workers = []
        
        high_risk_workers = []
        medium_risk_workers = []
        low_risk_workers = []
        
        for result in self.worker_results:
            if result.priority == "高":
                high_priority_workers.append(result)
            elif result.priority == "中":
                medium_priority_workers.append(result)
            else:
                low_priority_workers.append(result)
            
            if result.risk_level == "高":
                high_risk_workers.append(result)
            elif result.risk_level == "中":
                medium_risk_workers.append(result)
            else:
                low_risk_workers.append(result)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_structure": {
                "boss_agent": self.boss_agent.name,
                "worker_agents": [worker.name for worker in self.worker_agents],
                "total_agents": len(self.worker_agents) + 1
            },
            "boss_evaluation": {
                "agent_name": self.boss_result.agent_name,
                "role": self.boss_result.role,
                "overall_evaluation": self.boss_result.overall_evaluation,
                "final_decision": self.boss_result.final_decision,
                "risk_analysis": self.boss_result.risk_analysis,
                "improvement_roadmap": self.boss_result.improvement_roadmap
            },
            "worker_summary": {
                "total_workers": len(self.worker_results),
                "high_priority_issues": len(high_priority_workers),
                "medium_priority_issues": len(medium_priority_workers),
                "low_priority_issues": len(low_priority_workers),
                "high_risk_issues": len(high_risk_workers),
                "medium_risk_issues": len(medium_risk_workers),
                "low_risk_issues": len(low_risk_workers)
            },
            "worker_results": [
                {
                    "name": result.agent_name,
                    "role": result.role,
                    "evaluation": result.evaluation,
                    "recommendations": result.recommendations,
                    "risk_level": result.risk_level,
                    "priority": result.priority
                }
                for result in self.worker_results
            ],
            "issues_by_priority": {
                "high": [{"agent": r.agent_name, "recommendations": r.recommendations} for r in high_priority_workers],
                "medium": [{"agent": r.agent_name, "recommendations": r.recommendations} for r in medium_priority_workers],
                "low": [{"agent": r.agent_name, "recommendations": r.recommendations} for r in low_priority_workers]
            },
            "issues_by_risk": {
                "high": [{"agent": r.agent_name, "recommendations": r.recommendations} for r in high_risk_workers],
                "medium": [{"agent": r.agent_name, "recommendations": r.recommendations} for r in medium_risk_workers],
                "low": [{"agent": r.agent_name, "recommendations": r.recommendations} for r in low_risk_workers]
            }
        }
        
        return report
    
    def display_results(self):
        """結果をリッチな形式で表示"""
        if not self.boss_result:
            console.print("[red]表示する結果がありません[/red]")
            return
        
        # BOSS結果の表示
        console.print(Panel.fit(
            f"[bold]👑 BOSS評価結果[/bold]\n\n"
            f"[bold]統合評価:[/bold]\n{self.boss_result.overall_evaluation}\n\n"
            f"[bold]最終判定:[/bold] {self.boss_result.final_decision}\n"
            f"[bold]リスク分析:[/bold]\n{self.boss_result.risk_analysis}\n\n"
            f"[bold]改善ロードマップ:[/bold]\n" + "\n".join([f"• {item}" for item in self.boss_result.improvement_roadmap]),
            title=f"BOSS: {self.boss_result.agent_name}",
            border_style="yellow"
        ))
        
        # Worker結果のサマリーテーブル
        if self.worker_results:
            summary_table = Table(title="Workerエージェント評価サマリー")
            summary_table.add_column("Worker", style="cyan")
            summary_table.add_column("役割", style="magenta")
            summary_table.add_column("リスクレベル", style="red")
            summary_table.add_column("優先度", style="yellow")
            
            for result in self.worker_results:
                risk_color = {"高": "red", "中": "yellow", "低": "green"}.get(result.risk_level, "white")
                priority_color = {"高": "red", "中": "yellow", "低": "green"}.get(result.priority, "white")
                
                summary_table.add_row(
                    result.agent_name,
                    result.role,
                    f"[{risk_color}]{result.risk_level}[/{risk_color}]",
                    f"[{priority_color}]{result.priority}[/{priority_color}]"
                )
            
            console.print(summary_table)
            
            # Worker詳細結果
            for result in self.worker_results:
                console.print(Panel(
                    f"[bold]{result.agent_name}[/bold] - {result.role}\n\n"
                    f"[bold]評価結果:[/bold]\n{result.evaluation}\n\n"
                    f"[bold]推奨事項:[/bold]\n" + "\n".join([f"• {rec}" for rec in result.recommendations]),
                    title=f"Worker: {result.agent_name}",
                    border_style="blue"
                ))
    
    def save_report(self, filename: str = None):
        """レポートをJSONファイルに保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"boss_worker_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        console.print(f"[green]レポートを保存しました: {filename}[/green]")
        return filename 