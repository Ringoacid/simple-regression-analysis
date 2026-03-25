import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データセットのURL
DAVIS_CSV_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Davis.csv"

# DataFrameの散布図を描画する
def plot_scatter(df : pd.DataFrame, output_file : str = None, params : tuple[float, float] = None) -> None:
    plt.scatter(df['weight'], df['height'])
    plt.xlabel('体重(kg)') # 元の単位に戻す
    plt.ylabel('身長(cm)') # 元の単位に戻す
    plt.grid(True)

    if params is not None:
        a, b = params

        min_x = df['weight'].min()
        max_x = df['weight'].max()

        x = [min_x, max_x]
        y = [model(a, b, min_x), model(a, b, max_x)]
        plt.plot(x, y, color='red', label='線形モデル')
        plt.title('体重と身長の散布図と線形モデル')
        plt.legend()
    else:
        plt.title('体重と身長の散布図')
    
    if output_file is not None:
        # output_fileのディレクトリが存在しない場合は作成する
        if output_file.rsplit('/', 1)[0] != '' and not os.path.exists(output_file.rsplit('/', 1)[0]):
            os.makedirs(output_file.rsplit('/', 1)[0])
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

# 身長と体重のデータを読み込む
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DAVIS_CSV_URL, index_col=False) # データの読み込み
    df = df[['weight', 'height']] # 必要な列だけを抽出
    mask = (df['weight'] < 110) # 外れ値を除去するために、体重が110kg未満のデータのみを残す
    df = df[mask]
    return df

# 線形モデル
def model(a : float, b : float, x : float) -> float:
    return a * x + b


def main(output_dir : str = 'output_with_normalization'):
    # 既存のデータの読み込み (元のスケールを保持)
    df_orig = load_data()
    n = df_orig.shape[0] # データの数

    # 平均と標準偏差を計算して保存
    mean_w = df_orig['weight'].mean()
    std_w = df_orig['weight'].std()
    mean_h = df_orig['height'].mean()
    std_h = df_orig['height'].std()

    # 学習用に正規化されたデータフレームを作成
    df_norm = df_orig.copy()
    df_norm['weight'] = (df_norm['weight'] - mean_w) / std_w
    df_norm['height'] = (df_norm['height'] - mean_h) / std_h

    # パラメーターの初期値 (正規化空間用)
    a_norm = 0
    b_norm = 0

    # 学習の設定
    epochs_count = 20 # 学習の繰り返し回数
    eta = 0.1 # 学習率

    a_list = [] # 学習の過程での元のaの値を保存する
    b_list = [] # 学習の過程での元のbの値を保存する
    mse_list = [] # 学習の過程での平均二乗誤差を保存する

    for epoch in range(epochs_count):
        print(f"エポック {epoch + 1}/{epochs_count}")
        sse = 0 # 誤差の二乗和
        da = 0 # aの偏微分の値
        db = 0 # bの偏微分の値

        # シグマ(Σ)の計算 (正規化データを使用)
        for i in range(n):
            xi = df_norm.iloc[i]['weight'] # i番目の体重
            yi = df_norm.iloc[i]['height'] # i番目の身長
            
            pred_height = model(a_norm, b_norm, xi) # 体重から身長を予測
            sse += (yi - pred_height) ** 2 # 誤差を二乗して加算
            da += -2 * xi * (yi - pred_height)
            db += -2 * (yi - pred_height)

        da /= n # aの偏微分の平均値
        db /= n # bの偏微分の平均値

        # パラメーターの更新 (正規化空間)
        a_norm -= eta * da
        b_norm -= eta * db

        # --- 正規化されたパラメーターを元のスケールに逆変換 ---
        a_orig = a_norm * (std_h / std_w)
        b_orig = std_h * (b_norm - a_norm * (mean_w / std_w)) + mean_h
        # ------------------------------------------------------

        # 散布図と線形モデルを描画して保存 (元のデータと元のパラメーターを使用)
        plot_scatter(df_orig, output_file=f'{output_dir}/scatter_epoch_{epoch + 1}.svg', params=(a_orig, b_orig))
        
        # 平均二乗誤差の計算と保存
        mse = sse / n # ※正規化空間でのMSE
        mse_list.append(mse)
        
        # 最終的なリストには元のスケールに変換したa, bを保存する
        a_list.append(a_orig)
        b_list.append(b_orig)


    # 学習結果(MSEの遷移と最終的なパラメーター)を保存
    with open(f'{output_dir}/learning_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"最終的なパラメーター: a = {a_list[-1]}, b = {b_list[-1]}\n")
        f.write("エポック数\taの値\tbの値\t平均二乗誤差(正規化)\n")
        for epoch, (a_val, b_val, mse) in enumerate(zip(a_list, b_list, mse_list)):
            f.write(f"{epoch + 1}\t{a_val}\t{b_val}\t{mse}\n")

    # 学習の過程での平均二乗誤差をプロット
    plt.plot(mse_list)
    plt.xlabel('エポック数')
    plt.ylabel('平均二乗誤差 (正規化空間)')
    plt.title('学習の過程での平均二乗誤差の推移')
    plt.grid(True)
    plt.savefig(f'{output_dir}/mse_transition.svg')
    plt.show()
   
if __name__ == "__main__":
    main()