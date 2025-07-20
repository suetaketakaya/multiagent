#!/usr/bin/env python3
"""
BOSS-Workerマルチエージェントシステム メイン実行ファイル
Ollamaモデルを使用してeコマースアプリケーションの多角的評価を実行
"""

import asyncio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from multi_agent_system import MultiAgentSystem
from config import config, BOSS_CONFIG, WORKER_CONFIGS

console = Console()
app = typer.Typer()

@app.command()
def run(
    url: str = typer.Option(
        "https://ecommerce-with-stripe-six.vercel.app/",
        "--url",
        "-u",
        help="評価対象のアプリケーションURL"
    ),
    source_code: str = typer.Option(
        "https://github.com/kychan23/ecommerce-with-stripe",
        "--source",
        "-s",
        help="ソースコードリポジトリURL"
    ),
    save_report: bool = typer.Option(
        True,
        "--save/--no-save",
        help="レポートをファイルに保存するかどうか"
    ),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="出力ファイル名（指定しない場合は自動生成）"
    ),
    show_conversations: bool = typer.Option(
        True,
        "--conversations/--no-conversations",
        help="Worker間の会話をリアルタイムで表示するかどうか"
    )
):
    """BOSS-Workerマルチエージェントシステムを実行してeコマースアプリケーションを評価"""
    
    console.print(Panel.fit(
        "[bold blue]BOSS-Workerマルチエージェント評価システム[/bold blue]\n"
        "Ollamaモデルを使用した階層的品質評価",
        border_style="blue"
    ))
    
    # 対象情報を設定
    target_info = {
        "url": url,
        "source_code": source_code,
        "description": "Stripe決済統合を持つeコマースサイト（JPY通貨、日本限定取引、レスポンシブUI対応）"
    }
    
    async def main():
        # システム初期化
        system = MultiAgentSystem()
        
        # 接続テスト
        console.print("\n[bold]🔍 Ollamaサーバー接続テスト中...[/bold]")
        if not await system.test_connection():
            console.print("[red]❌ Ollamaサーバーに接続できません。サーバーが起動しているか確認してください。[/red]")
            return
        
        # 設定確認
        console.print(f"\n[bold]📋 評価設定[/bold]")
        console.print(f"対象URL: {target_info['url']}")
        console.print(f"ソースコード: {target_info['source_code']}")
        console.print(f"BOSS: {BOSS_CONFIG.name} ({BOSS_CONFIG.model})")
        console.print(f"Worker: {len(WORKER_CONFIGS)}名")
        for worker in WORKER_CONFIGS:
            console.print(f"  • {worker.name} ({worker.model})")
        console.print(f"会話表示: {'有効' if show_conversations else '無効'}")
        
        if not Confirm.ask("評価を開始しますか？"):
            console.print("[yellow]評価をキャンセルしました[/yellow]")
            return
        
        # 評価実行
        console.print("\n[bold]🚀 BOSS-Worker評価開始[/bold]")
        boss_result = await system.run_evaluation(target_info)
        
        if not boss_result:
            console.print("[red]❌ 評価結果がありません[/red]")
            return
        
        # 結果表示
        console.print("\n[bold]📊 評価結果[/bold]")
        system.display_results()
        
        # レポート保存
        if save_report:
            filename = system.save_report(output_file)
            console.print(f"\n[green]✅ レポートが保存されました: {filename}[/green]")
        
        # 最終判定
        decision = boss_result.final_decision
        decision_color = "green" if decision == "Go" else "red"
        
        console.print(Panel.fit(
            f"[bold]最終判定: [{decision_color}]{decision}[/{decision_color}][/bold]\n"
            f"リスク分析: {boss_result.risk_analysis}\n"
            f"改善項目数: {len(boss_result.improvement_roadmap)}件",
            title="BOSS最終判定",
            border_style=decision_color
        ))
    
    # 非同期実行
    asyncio.run(main())

@app.command()
def test_connection():
    """Ollamaサーバーとの接続をテスト"""
    async def test():
        system = MultiAgentSystem()
        success = await system.test_connection()
        
        if success:
            console.print("[green]✅ Ollamaサーバーに正常に接続できました[/green]")
            
            # 利用可能なモデルを表示
            try:
                models = await system.ollama_client.check_models()
                console.print(f"\n[bold]利用可能なモデル:[/bold]")
                for model in models.get('models', []):
                    console.print(f"• {model['name']} ({model['details']['parameter_size']})")
            except Exception as e:
                console.print(f"[yellow]⚠️ モデル情報の取得に失敗: {e}[/yellow]")
        else:
            console.print("[red]❌ Ollamaサーバーに接続できません[/red]")
            console.print("以下の点を確認してください:")
            console.print("• Ollamaサーバーが起動しているか")
            console.print("• http://localhost:11434 にアクセスできるか")
            console.print("• ファイアウォール設定")
    
    asyncio.run(test())

@app.command()
def show_config():
    """現在の設定を表示"""
    console.print(Panel.fit(
        f"[bold]BOSS-Worker設定情報[/bold]\n\n"
        f"Ollama URL: {config.ollama.base_url}\n"
        f"タイムアウト: {config.ollama.timeout}秒\n"
        f"最大トークン: {config.ollama.max_tokens}\n\n"
        f"[bold]BOSSエージェント:[/bold]\n"
        f"• {BOSS_CONFIG.name} ({BOSS_CONFIG.model}) - {BOSS_CONFIG.role}\n\n"
        f"[bold]Workerエージェント:[/bold]\n" +
        "\n".join([
            f"• {worker.name} ({worker.model}) - {worker.role}"
            for worker in WORKER_CONFIGS
        ]),
        title="BOSS-Workerシステム設定",
        border_style="blue"
    ))

@app.command()
def show_structure():
    """システム構造を表示"""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    # システム構造の説明
    structure_text = Text()
    structure_text.append("🏗️ BOSS-Workerマルチエージェントシステム構造\n\n", style="bold blue")
    
    structure_text.append("👑 BOSSエージェント\n", style="bold yellow")
    structure_text.append("   • BOSS_Agent (プロジェクト統括・品質保証マネージャー)\n", style="yellow")
    structure_text.append("   • 役割: Worker結果統合、最終判定、改善ロードマップ策定\n\n", style="dim")
    
    structure_text.append("🔧 Workerエージェント群\n", style="bold green")
    structure_text.append("   • ISTQB_Compliance_Worker (ISTQB準拠・法規制コンプライアンス確認者)\n", style="green")
    structure_text.append("   • Management_Requirements_Worker (マネジメント・顧客要件確認者)\n", style="green")
    structure_text.append("   • Technical_Analyst_Worker (テストアナリスト・技術者視点評価者)\n", style="green")
    structure_text.append("   • UX_Design_Worker (UX/UI設計・ユーザビリティ評価者)\n", style="green")
    structure_text.append("   • Security_Audit_Worker (セキュリティ監査・脆弱性評価者)\n\n", style="green")
    
    structure_text.append("🔄 処理フロー\n", style="bold magenta")
    structure_text.append("   Step 1: Workerエージェント並行評価\n", style="magenta")
    structure_text.append("   Step 1.5: Workerエージェント間会話・協調\n", style="magenta")
    structure_text.append("   Step 2: BOSSエージェント統合評価\n", style="magenta")
    structure_text.append("   Step 3: 最終判定・改善ロードマップ生成\n\n", style="magenta")
    
    structure_text.append("💬 会話タイプ\n", style="bold cyan")
    structure_text.append("   • ❓ 質問: 専門分野間の情報交換\n", style="cyan")
    structure_text.append("   • 💬 回答: 質問への専門的回答\n", style="cyan")
    structure_text.append("   • 🤝 協力: 共同改善提案の検討\n", style="cyan")
    structure_text.append("   • ⚖️ 議論: 異なる観点の建設的議論\n", style="cyan")
    
    panel = Panel(structure_text, title="システム構造", border_style="blue")
    console.print(panel)
    
    # エージェント詳細テーブル
    table = Table(title="エージェント詳細", show_header=True, header_style="bold magenta")
    table.add_column("エージェント名", style="cyan", no_wrap=True)
    table.add_column("役割", style="green")
    table.add_column("使用モデル", style="yellow")
    table.add_column("専門分野", style="blue")
    
    # BOSSエージェント
    table.add_row(
        "BOSS_Agent",
        "プロジェクト統括・品質保証マネージャー",
        "pakachan/elyza-llama3-8b:latest",
        "統合評価・最終判定・改善ロードマップ"
    )
    
    # Workerエージェント
    table.add_row(
        "ISTQB_Compliance_Worker",
        "ISTQB準拠・法規制コンプライアンス確認者",
        "pakachan/elyza-llama3-8b:latest",
        "PCI DSS・個人情報保護法・特定商取引法"
    )
    
    table.add_row(
        "Management_Requirements_Worker",
        "マネジメント・顧客要件確認者",
        "gemma3:latest",
        "ビジネスモデル・市場適合性・ROI分析"
    )
    
    table.add_row(
        "Technical_Analyst_Worker",
        "テストアナリスト・技術者視点評価者",
        "llama3.2:latest",
        "アーキテクチャ・パフォーマンス・技術的品質"
    )
    
    table.add_row(
        "UX_Design_Worker",
        "UX/UI設計・ユーザビリティ評価者",
        "gemma3:latest",
        "ユーザビリティ・アクセシビリティ・UI/UX設計"
    )
    
    table.add_row(
        "Security_Audit_Worker",
        "セキュリティ監査・脆弱性評価者",
        "pakachan/elyza-llama3-8b:latest",
        "認証・暗号化・入力検証・APIセキュリティ"
    )
    
    console.print(table)

@app.command()
def preview_conversations():
    """Worker間の会話プレビューを表示"""
    console.print(Panel.fit(
        "[bold]Worker間会話プレビュー[/bold]\n\n"
        "この機能では、Workerエージェント間の以下の会話が表示されます：\n\n"
        "❓ 質問: 専門分野間の情報交換\n"
        "💬 回答: 質問への専門的回答\n"
        "🤝 協力: 共同改善提案の検討\n"
        "⚖️ 議論: 異なる観点の建設的議論\n\n"
        "会話はリアルタイムで表示され、各Workerの専門性を活かした\n"
        "建設的な議論が行われます。",
        title="会話プレビュー機能",
        border_style="cyan"
    ))

if __name__ == "__main__":
    app() 