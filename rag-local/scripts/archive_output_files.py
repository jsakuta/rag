"""
出力ファイルを年月別にアーカイブするスクリプト
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

def archive_output_files(base_dir: str):
    """2025年以前のファイルを年月別にアーカイブ"""
    output_dir = Path(base_dir) / "output"
    archive_dir = output_dir / "archive"
    
    if not output_dir.exists():
        print(f"出力ディレクトリが見つかりません: {output_dir}")
        return
    
    # アーカイブディレクトリ作成
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    archived_count = 0
    skipped_count = 0
    
    # output直下のファイルのみ対象（既存のarchiveは除外）
    for file_path in output_dir.iterdir():
        if not file_path.is_file():
            continue
        
        # ファイルの更新日時を取得
        mtime = file_path.stat().st_mtime
        file_date = datetime.fromtimestamp(mtime)
        
        # 2025年以前のファイルのみアーカイブ
        if file_date.year <= 2025:
            # YYYYMM形式のディレクトリ
            year_month = file_date.strftime("%Y%m")
            dest_dir = archive_dir / year_month
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイル移動
            dest_path = dest_dir / file_path.name
            try:
                shutil.move(str(file_path), str(dest_path))
                print(f"アーカイブ: {file_path.name} → archive/{year_month}/")
                archived_count += 1
            except Exception as e:
                print(f"エラー: {file_path.name} - {e}")
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"\n完了: {archived_count}件アーカイブ, {skipped_count}件スキップ")

if __name__ == "__main__":
    base_dir = str(Path(__file__).parent.parent)
    archive_output_files(base_dir)
