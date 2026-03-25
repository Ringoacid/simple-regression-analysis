import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# データセットのURL
DAVIS_CSV_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Davis.csv"

# DataFrameの散布図を描画する
def plot_scatter(df : pd.DataFrame, output_file : str = None, params : tuple[float, float] = None) -> None:
    plt.scatter(df['weight'], df['height'])
    plt.xlabel('体重(kg)')
    plt.ylabel('身長(cm)')
    plt.grid(True)

    if params is not None:
        a, b = params

        min_x = df['weight'].min()
        max_x = df['weight'].max()

        x = [min_x, max_x]
        y = [model(a, b, min_x), model(a, b, max_x)]
        plt.plot(x, y, color='red', label='線形モデル')
        plt.title('体重と身長の散布図と線形モデル')
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


def main(output_dir : str = 'output_without_normalization'):
    # 既存のデータの読み込み
    df = load_data()
    n = df.shape[0] # データの数

    # パラメーターの初期値
    a = 0
    b = 0

    # 学習の設定
    epochs_count = 20 # 学習の繰り返し回数
    eta = 0.00002 # 学習率

    a_list = [] # 学習の過程でのaの値を保存する
    b_list = [] # 学習の過程でのbの値を保存する
    mse_list = [] # 学習の過程での平均二乗誤差を保存する

    for epoch in range(epochs_count):
        print(f"エポック {epoch + 1}/{epochs_count}")
        sse = 0 # 誤差の二乗和
        da = 0 # aの偏微分の値
        db = 0 # bの偏微分の値

        # シグマ(Σ)の計算
        for i in range(n):
            xi = df.iloc[i]['weight'] # i番目の体重
            yi = df.iloc[i]['height'] # i番目の身長
            
            pred_height = model(a, b, xi) # 体重から身長を予測 (pred_height = ax_i + b)
            sse += (yi - pred_height) ** 2 # 誤差を二乗して加算
            da += -2 * xi * (yi - pred_height)
            db += -2 * (yi - pred_height)

        plot_scatter(df, output_file=f'{output_dir}/scatter_epoch_{epoch + 1}.svg', params=(a, b)) # 散布図と線形モデルを描画して保存
        
        da /= n # aの偏微分の平均値
        db /= n # bの偏微分の平均値

        # パラメーターの更新
        a -= eta * da
        b -= eta * db

        # 平均二乗誤差の計算と保存
        mse = sse / n
        mse_list.append(mse)
        a_list.append(a)
        b_list.append(b)


    # 学習結果(MSEの遷移と最終的なパラメーター)を保存
    with open(f'{output_dir}/learning_results.txt', 'w', encoding='utf-8') as f:
        f.write(f"最終的なパラメーター: a = {a}, b = {b}\n")
        f.write("エポック数\taの値\tbの値\t平均二乗誤差\n")
        for epoch, (a_val, b_val, mse) in enumerate(zip(a_list, b_list, mse_list)):
            f.write(f"{epoch + 1}\t{a_val}\t{b_val}\t{mse}\n")

    # 学習の過程での平均二乗誤差をプロット
    plt.plot(mse_list)
    plt.xlabel('エポック数')
    plt.ylabel('平均二乗誤差')
    plt.title('学習の過程での平均二乗誤差の推移')
    plt.grid(True)
    plt.savefig(f'{output_dir}/mse_transition.svg')
    plt.show()
   
if __name__ == "__main__":
    main()
