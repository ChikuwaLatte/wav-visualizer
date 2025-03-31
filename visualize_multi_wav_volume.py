import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import glob
import os

# コマンドライン引数の処理
parser = argparse.ArgumentParser(description="フォルダ内の全WAVファイルの音量を1つの動画で可視化（緑背景）")
parser.add_argument("input_folder", type=str, help="WAVファイルが格納されているフォルダ")
parser.add_argument("output_file", type=str, help="出力する動画ファイル名 (例: output.mp4)")
args = parser.parse_args()

# 入力フォルダ内のすべてのWAVファイルを取得（ソートして処理）
wav_files = sorted(glob.glob(os.path.join(args.input_folder, "*.wav")))

# WAVファイルが見つからない場合の処理
if not wav_files:
    print("⚠️ WAVファイルが見つかりませんでした。正しいフォルダを指定してください。")
    exit(1)

# FPS（20FPSで動画を作成）
fps = 20

# 各WAVファイルの音量データを取得
rms_list = []
max_duration = 0  # 最大のWAVの長さを求める

for wav_file in wav_files:
    y, sr = librosa.load(wav_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    max_duration = max(max_duration, duration)  # 最大の長さを更新

    # フレームごとの音量（RMS: 二乗平均平方根）を計算
    hop_length = int(sr / fps)
    rms = librosa.feature.rms(y=y, frame_length=hop_length, hop_length=hop_length)[0]
    rms_list.append(rms)

# フレーム数を最大のWAVの長さに基づいて統一
num_frames = int(max_duration * fps)

# 短いWAVの音量データを0埋めして統一
for i in range(len(rms_list)):
    if len(rms_list[i]) < num_frames:
        rms_list[i] = np.pad(rms_list[i], (0, num_frames - len(rms_list[i])), mode='constant')

# グラフの設定
num_bars = len(wav_files)  # 棒グラフの本数
bar_width = 0.8 / num_bars  # 棒の幅を動的に調整
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor('#00FF00')  # 背景をクロマキー用の緑
ax.set_facecolor('#00FF00')  # 軸背景も緑

# 棒グラフの初期設定
x_positions = np.linspace(-0.4, 0.4, num_bars)  # 左から順に並べるためのX座標
bars = ax.bar(x_positions, [0] * num_bars, width=bar_width, color='white', align='center')

# 軸の非表示
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 縦軸のスケールを最大音量に合わせる
max_rms = max(max(rms) for rms in rms_list)
ax.set_ylim(0, max_rms * 1.1)

# アニメーション更新関数
def update(frame):
    for i, bar in enumerate(bars):
        bar.set_height(rms_list[i][frame])  # 各バーの高さを更新
    return bars

# tqdm のプログレスバー
with tqdm(total=num_frames, desc="Rendering video", unit="frame") as pbar:
    writer = FFMpegWriter(fps=fps, codec="libx264")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=True)

    # 出力動画を保存
    ani.save(args.output_file, writer=writer)
    pbar.update(num_frames)

print(f"🎬 動画を {args.output_file} に保存しました！")
