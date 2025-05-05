@echo off
echo ===================================
echo 居眠り検知システム ビルドスクリプト
echo ===================================

echo 必要なライブラリをインストールしています...
pip install -r requirements.txt
pip install pyinstaller

echo EXEファイルをビルドしています...
pyinstaller drowsiness_watcher.spec

echo ビルド完了！
echo dist\drowsiness_watcher.exe が作成されました

echo 完成したファイルをデスクトップにコピーしますか？ (Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    echo デスクトップにコピーしています...
    copy dist\drowsiness_watcher.exe %USERPROFILE%\Desktop\
    copy .env.template %USERPROFILE%\Desktop\.env.template
    echo コピー完了！
    echo デスクトップの .env.template を .env にリネームして設定を編集してください。
)

echo ビルドプロセスが完了しました。
pause