import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyaudio
import os
import socket
import getpass
import csv
import datetime
import atexit
import glob
from pathlib import Path
import logging
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# --- ロギングの設定 ---
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- 環境変数の読み込み ---
load_dotenv()
SLACK_TOKEN = os.getenv('SLACK_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# --- CSVログの設定 ---
def get_log_filename():
    """現在の日時に基づいたログファイル名を生成"""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_log.csv")

# CSVログファイルの保存パス
csv_log_path = get_log_filename()

# CSVログのヘッダー
csv_headers = [
    'timestamp', 
    'ear_value', 
    'status', 
    'closed_duration', 
    'alert_triggered', 
    'device_name', 
    'user_name', 
    'mode'
]

# システム情報の取得
device_name = socket.gethostname()
user_name = getpass.getuser()
mode = "auto"  # 固定値

# CSVファイルの初期化
def init_csv_log():
    """CSVログファイルを初期化し、ヘッダーを書き込む"""
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    # バックアップディレクトリの作成と初期ログのコピー
    backup_dir = Path('.hidden_backup')
    backup_dir.mkdir(exist_ok=True)
    with open(backup_dir / csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    logging.info(f"CSV ログファイルを初期化しました: {csv_log_path}")

# CSVにログを書き込む
def write_csv_log(ear_value, status, closed_duration, alert_triggered):
    """現在のステータスをCSVログに記録"""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    log_row = [
        timestamp,
        f"{ear_value:.3f}",
        status,
        f"{closed_duration:.1f}" if closed_duration is not None else "0.0",
        str(alert_triggered),
        device_name,
        user_name,
        mode
    ]
    
    # 本体CSVに書き込み
    with open(csv_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_row)
    
    # バックアップにも書き込み
    backup_path = Path('.hidden_backup') / csv_log_path
    with open(backup_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_row)

# --- Slack送信処理 ---
def send_log_to_slack():
    """プログラム終了時にCSVログをSlackに送信"""
    files = sorted(glob.glob("*_log.csv"), reverse=True)
    if not files:
        logging.warning("送信するCSVログファイルが見つかりません")
        return
    
    latest_file = files[0]
    logging.info(f"Slackに送信するファイル: {latest_file}")
    
    if not SLACK_TOKEN or not SLACK_CHANNEL:
        logging.error("Slack設定が不完全です。.envファイルを確認してください")
        with open("slack_failed_log.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} - 設定エラー: {latest_file}\n")
        return
    
    try:
        client = WebClient(token=SLACK_TOKEN)
        response = client.files_upload_v2(
            channel=SLACK_CHANNEL,
            file=latest_file,
            title=f"居眠り検知ログ: {latest_file}",
            initial_comment=f"デバイス: {device_name} | ユーザー: {user_name}"
        )
        logging.info(f"Slackにログを送信しました: {latest_file}")
    except SlackApiError as e:
        error_msg = f"Slack送信失敗: {e.response['error']}"
        logging.error(error_msg)
        with open("slack_failed_log.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} - {error_msg}: {latest_file}\n")

# プログラム終了時にSlack送信を登録
atexit.register(send_log_to_slack)

# --- 以前の送信失敗ログがあれば、再送信を試みる ---
def check_and_resend_failed_logs():
    """前回の実行で送信に失敗したログがあれば再送信を試みる"""
    if not os.path.exists("slack_failed_log.txt"):
        return
    
    with open("slack_failed_log.txt", "r") as f:
        failed_logs = f.readlines()
    
    if not failed_logs:
        return
    
    logging.info("前回送信に失敗したログの再送信を試みます")
    
    # 失敗ログファイルを空にする
    open("slack_failed_log.txt", "w").close()
    
    # 各ログファイルの送信を試みる
    for line in failed_logs:
        parts = line.strip().split(" - ")
        if len(parts) < 2:
            continue
            
        filename = parts[-1]
        if os.path.exists(filename) and filename.endswith("_log.csv"):
            try:
                client = WebClient(token=SLACK_TOKEN)
                response = client.files_upload_v2(
                    channel=SLACK_CHANNEL,
                    file=filename,
                    title=f"[再送]居眠り検知ログ: {filename}",
                    initial_comment=f"デバイス: {device_name} | ユーザー: {user_name} | 再送信"
                )
                logging.info(f"失敗したログを再送信しました: {filename}")
            except SlackApiError as e:
                error_msg = f"再送信失敗: {e.response['error']}"
                logging.error(error_msg)
                with open("slack_failed_log.txt", "a") as f:
                    f.write(f"{datetime.datetime.now()} - {error_msg}: {filename}\n")

# --- PyAudio の設定 ---
p = pyaudio.PyAudio()
fs = 44100            # サンプリングレート
duration = 0.1        # ビープ音の長さ（秒）
frequency = 1000.0    # 周波数（Hz）

# 正弦波トーンを生成（振幅0.5）
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
beep = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

# 出力用ストリームを開始
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# --- 警報関連のグローバル変数 ---
alarm_state = False         # True のとき警報発動
closed_start_time = None    # 目が閉じ始めた時刻
closed_threshold_time = 1.0 # 1秒以上閉じたら警報状態

# --- Mediapipe FaceMesh の設定 ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Mediapipe で使用する目のランドマークのインデックス
RIGHT_EYE_TOP    = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_EYE_LEFT   = 33
RIGHT_EYE_RIGHT  = 133

LEFT_EYE_TOP     = 386
LEFT_EYE_BOTTOM  = 374
LEFT_EYE_LEFT    = 362
LEFT_EYE_RIGHT   = 263

def eye_aspect_ratio(eye_top, eye_bottom, eye_left, eye_right):
    """目のアスペクト比（EAR）を計算"""
    vertical = np.linalg.norm(np.array(eye_top) - np.array(eye_bottom))
    horizontal = np.linalg.norm(np.array(eye_left) - np.array(eye_right))
    return vertical / horizontal if horizontal != 0 else 0

# --- 警報ループ（バックグラウンドスレッド） ---
def alarm_loop():
    """警報状態を監視し、警報を鳴らすバックグラウンドスレッド"""
    global alarm_state
    while True:
        if alarm_state:
            # 警報状態ならビープ音を再生
            stream.write(beep.tobytes())
        else:
            time.sleep(0.1)

def main():
    """メイン処理"""
    global alarm_state, closed_start_time
    
    # CSVログの初期化
    init_csv_log()
    
    # 前回の失敗ログを確認し再送信
    check_and_resend_failed_logs()
    
    # 警報スレッドの開始
    alarm_thread = threading.Thread(target=alarm_loop, daemon=True)
    alarm_thread.start()
    
    # Webカメラの開始
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logging.error("カメラを開けませんでした。デバイスを確認してください。")
        return
    
    logging.info("居眠り検知システムを開始しました")
    
    last_log_time = time.time()
    log_interval = 1.0  # 1秒ごとにログを記録
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error("フレームの取得に失敗しました")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            status = "Open"
            color = (0, 255, 0)
            current_time = time.time()
            avg_ear = 0
            closed_duration = 0.0
            
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    
                    # 右目のランドマーク取得
                    r_top = (int(landmarks.landmark[RIGHT_EYE_TOP].x * w),
                             int(landmarks.landmark[RIGHT_EYE_TOP].y * h))
                    r_bottom = (int(landmarks.landmark[RIGHT_EYE_BOTTOM].x * w),
                                int(landmarks.landmark[RIGHT_EYE_BOTTOM].y * h))
                    r_left = (int(landmarks.landmark[RIGHT_EYE_LEFT].x * w),
                              int(landmarks.landmark[RIGHT_EYE_LEFT].y * h))
                    r_right = (int(landmarks.landmark[RIGHT_EYE_RIGHT].x * w),
                               int(landmarks.landmark[RIGHT_EYE_RIGHT].y * h))
                    
                    # 左目のランドマーク取得
                    l_top = (int(landmarks.landmark[LEFT_EYE_TOP].x * w),
                             int(landmarks.landmark[LEFT_EYE_TOP].y * h))
                    l_bottom = (int(landmarks.landmark[LEFT_EYE_BOTTOM].x * w),
                                int(landmarks.landmark[LEFT_EYE_BOTTOM].y * h))
                    l_left = (int(landmarks.landmark[LEFT_EYE_LEFT].x * w),
                              int(landmarks.landmark[LEFT_EYE_LEFT].y * h))
                    l_right = (int(landmarks.landmark[LEFT_EYE_RIGHT].x * w),
                               int(landmarks.landmark[LEFT_EYE_RIGHT].y * h))
                    
                    # 両目の EAR を計算
                    r_ear = eye_aspect_ratio(r_top, r_bottom, r_left, r_right)
                    l_ear = eye_aspect_ratio(l_top, l_bottom, l_left, l_right)
                    avg_ear = (r_ear + l_ear) / 2
                    
                    threshold = 0.25
                    if avg_ear < threshold:
                        if closed_start_time is None:
                            closed_start_time = current_time
                        
                        closed_duration = current_time - closed_start_time
                        if closed_duration >= closed_threshold_time:
                            status = "Closed"
                            color = (0, 0, 255)
                            alarm_state = True
                        else:
                            status = "Blinking"
                            color = (0, 165, 255)
                            alarm_state = False
                    else:
                        closed_start_time = None
                        closed_duration = 0.0
                        status = "Open"
                        color = (0, 255, 0)
                        alarm_state = False
                    
                    # 画面に情報を表示
                    cv2.putText(frame, f"Eye: {status}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    if alarm_state:
                        cv2.putText(frame, f"Alert! Wake up!", (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 各ランドマークを描画
                    for pt in [r_top, r_bottom, r_left, r_right, l_top, l_bottom, l_left, l_right]:
                        cv2.circle(frame, pt, 2, (255, 0, 0), -1)
            
            # 一定間隔でCSVログに記録
            if current_time - last_log_time >= log_interval:
                write_csv_log(avg_ear, status, closed_duration, alarm_state)
                last_log_time = current_time
                
            # 画面表示
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("ユーザーによる終了")
                break
    
    except KeyboardInterrupt:
        logging.info("キーボード割り込みによる終了")
    except Exception as e:
        logging.error(f"エラーが発生しました: {str(e)}")
    finally:
        # リソースの解放
        cap.release()
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.info("プログラムを終了します")

if __name__ == "__main__":
    main()