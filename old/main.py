import platform
from typing import Optional
from config import *
from bot import LoanAssistantBot, Colors

def get_numeric_input(prompt: str, current_value: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Optional[float]:
    """数値入力を受け付け、検証する"""
    try:
        value_input = input(prompt).strip()
        if not value_input:
            return None
        
        value = float(value_input)
        
        if min_value is not None and value < min_value:
            print(f"{Colors.YELLOW}値は{min_value}以上である必要があります{Colors.END}")
            return None
        
        if max_value is not None and value > max_value:
            print(f"{Colors.YELLOW}値は{max_value}以下である必要があります{Colors.END}")
            return None
        
        return value
        
    except ValueError:
        print(f"{Colors.YELLOW}有効な数値を入力してください{Colors.END}")
        return None

def main():
    """メイン実行関数"""
    try:
        # Windows環境でANSIエスケープシーケンスを有効化
        if platform.system() == "Windows":
            import os
            os.system("color")

        # ボットの初期化
        bot = LoanAssistantBot(
            model_provider=MODEL_PROVIDER,
            model_name=MODEL_NAME,
            embedding_provider=EMBEDDING_PROVIDER,
            embedding_model_name=EMBEDDING_MODEL,
            reranker_model_name=RERANKER_MODEL,
            prompt_path=DEFAULT_PROMPT_FILE
        )
        bot.initialize()
        
        # 初期メッセージの表示
        print(f"\n{Colors.BOLD}融資業務Q&Aボット{Colors.END}")
        print(f"ドキュメントは {Colors.GREEN}{DOCS_DIRECTORY}{Colors.END} から再帰的に読み込まれます。")
        print(f"\n{Colors.BOLD}コマンド一覧:{Colors.END}")
        print(f"  {Colors.YELLOW}quit{Colors.END}: 終了")
        print(f"  {Colors.YELLOW}params{Colors.END}: 現在の検索パラメータを表示")
        print(f"  {Colors.YELLOW}set{Colors.END}: 検索パラメータを設定")
        print("それ以外の入力は質問として処理されます")
        
        print(f"\n{Colors.GRAY}使用中の埋め込みモデル: {bot.embeddings.model_name}")
        print(f"使用中のリランカーモデル: {RERANKER_MODEL}{Colors.END}")
        
        # メインループ
        while True:
            command = input(f"\n{Colors.BOLD}入力{Colors.END}: ").strip().lower()
            
            if command == 'quit':
                break
            
            elif command == 'params':
                bot.show_search_params()
            
            elif command == 'set':
                try:
                    print(f"\n{Colors.BOLD}=== 検索パラメータの設定 ==={Colors.END}")
                    print(f"{Colors.GRAY}(変更しない場合は空欄のままEnterを押してください){Colors.END}")
                    
                    initial_top_k = get_numeric_input(
                        f"初期検索件数 (現在: {Colors.GREEN}{bot.search_params.initial_top_k}{Colors.END}): ",
                        bot.search_params.initial_top_k,
                        min_value=1
                    )
                    
                    final_top_k = get_numeric_input(
                        f"最終検索件数 (現在: {Colors.GREEN}{bot.search_params.final_top_k}{Colors.END}): ",
                        bot.search_params.final_top_k,
                        min_value=1
                    )
                    
                    semantic_weight = get_numeric_input(
                        f"意味的類似度の重み (0-1) (現在: {Colors.GREEN}{bot.search_params.semantic_weight:.2f}{Colors.END}): ",
                        bot.search_params.semantic_weight,
                        min_value=0,
                        max_value=1
                    )
                    
                    if any(x is not None for x in [initial_top_k, final_top_k, semantic_weight]):
                        bot.update_search_params(
                            initial_top_k=initial_top_k,
                            final_top_k=final_top_k,
                            semantic_weight=semantic_weight
                        )
                        print(f"{Colors.GREEN}パラメータを更新しました{Colors.END}")
                    else:
                        print(f"{Colors.GRAY}パラメータは変更されませんでした{Colors.END}")
                    
                except (ValueError, Exception) as e:
                    print(f"{Colors.YELLOW}エラー: {e}{Colors.END}")
                    print(f"{Colors.YELLOW}パラメータの更新に失敗しました{Colors.END}")
            
            else:
                try:
                    answer = bot.ask(command)
                    print(f"\n{Colors.BOLD}回答:{Colors.END} {answer}")
                except Exception as e:
                    print(f"{Colors.YELLOW}エラー: 質問の処理中にエラーが発生しました: {e}{Colors.END}")
            
    except FileNotFoundError as e:
        print(f"{Colors.YELLOW}エラー: {e}")
        print("プロンプトテンプレートファイルが見つかりません。")
        print(f"以下のパスにプロンプトファイルを配置してください: {DEFAULT_PROMPT_FILE}{Colors.END}")
    
    except Exception as e:
        print(f"{Colors.YELLOW}エラー: アプリケーションの実行中にエラーが発生しました: {e}{Colors.END}")

if __name__ == "__main__":
    main()