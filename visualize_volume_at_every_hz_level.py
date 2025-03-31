import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

# コマンドライン引数を処理
parser = argparse.ArgumentParser(description="WAVファイルの周波数成分を棒グラフで動画化")
parser.add_argument("filename", type=str, help="解析するWAVファイルのパス")
parser.add_argument("output", type=str, help="出力する動画ファイルのパス (例: output.mp4)")
args = parser.parse_args()

# WAVファイルを読み込む
y, sr = librosa.load(args.filename, sr=None)

# STFT（短時間フーリエ変換）を適用
D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

# 0〜2000Hz の範囲のデータを抽出
frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
valid_indices = (frequencies >= 0) & (frequencies <= 2000)
frequencies = frequencies[valid_indices]
D = D[valid_indices, :]

# 50Hz 刻みで 40 本の棒グラフを表示するためにリサンプリング
num_bars = 40
freq_bins = np.linspace(0, 2000, num_bars)
bar_heights = np.zeros(num_bars)

# WAVファイルの長さ（秒）とFPS（10に設定）
duration = librosa.get_duration(y=y, sr=sr)  # WAVファイルの長さ (秒)
fps = 10  # FPSを指定

# 動画のフレーム数は、WAVの長さにFPSを掛けた値として計算
num_frames = int(duration * fps)

# ここでフレームレートに基づいて適切な間隔を設定します
interval = 1000 / fps  # インターバルを設定（ms単位）

# 動画設定（黒背景、白棒グラフ）
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
bars = ax.bar(freq_bins, bar_heights, width=40, color='white', bottom=0)

# 軸ラベル・目盛りを非表示
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xlim(0, 2000)
ax.set_ylim(0, 100)  # 上方向にグラフが伸びるように設定

# アニメーション更新関数
def update(frame):
    amp_db = librosa.amplitude_to_db(D[:, frame], ref=np.max)  # dBスケールに変換
    for i in range(num_bars - 1):
        idx_start = np.abs(frequencies - freq_bins[i]).argmin()
        idx_end = np.abs(frequencies - freq_bins[i + 1]).argmin()
        bar_heights[i] = np.mean(amp_db[idx_start:idx_end])  # 周波数帯域ごとに平均を取る

    # 各棒グラフの高さを更新
    for bar, height in zip(bars, bar_heights):
        bar.set_height(height)

    return bars

# tqdm のプログレスバーを使う
with tqdm(total=num_frames, desc="Processing frames", unit="frame") as pbar:
    writer = FFMpegWriter(fps=fps, codec="libx264")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)

    # tqdm のプログレスバーを適切に更新
    for i in range(num_frames):
        update(i)  # 1フレームずつ処理
        pbar.update(1)  # プログレスバーを進める

    ani.save(args.output, writer=writer)

print(f"動画を {args.output} に保存しました！ 🎬")
