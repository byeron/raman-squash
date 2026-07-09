import click
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import json
from pathlib import Path


def plot_dendrogram(Z, th, labels, img_path):
    n_leaves = len(labels)
    min_width = 4
    max_width = 36
    width = min(max(n_leaves * 0.1, min_width), max_width)

    if n_leaves > 1000:
        fontsize = 2
    else:
        fontsize = 4

    fig = plt.figure(figsize=(width, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    hierarchy.dendrogram(
        Z,
        ax=ax,
        leaf_rotation=90,
        color_threshold=th,
        leaf_font_size=fontsize,
        labels=labels,
    )
    plt.tight_layout()
    fig.savefig(img_path)
    print(f"output: medium dendrogram to {img_path}")
    return


def _bulkspectra(df, corr_th=0.2, dist_th=15, weighted=False):
    '''
    低解像度化処理の手順
    1. 偽相関をもつラマンスペクトルを集約するために、ラマンスペクトル間の相関を計算しクラスタリングする
        - 隣り合うバンドは偽相関により相関係数が高くなる傾向を持つため、バンド間の非類似度は小さくなる
        - 階層型クラスタリングにより偽相関を持つ変数が集約される
    2. 1で得られたクラスタ内で、隣り合わないバンドを別のクラスタとして分割する
        - カイザーを距離とし、非類似度が高い要素を分割する
    3. 1と2のステップで得られたクラスタに対して、代表値を決定する
        - 代表値は、平均値、加重平均、中央値などが候補となる
    4. 計算された各クラスタの代表値を新たな特徴量とし、元のラマンバンドの並びに基づきソートする
    '''

    def corr_clustering(df, threshold=0.2, img_path="img/corr_dendrogram.png"):
        '''
        非類似度: D = 1 - (corr(x, y) + 1) / 2
        - 相関係数の範囲の変更: -1 <= corr <= +1 から 0 <= (corr + 1) / 2 <= 1
        - 1 から補正後の相関係数を引くことで、正の相関を持つときに非類似度が低くなる指標に変換
        '''
        # ピアソンの相関係数を非類似度に変換する
        dissimilarity = 1 - (df.corr() + 1) / 2

        # 階層型クラスタリングの実行
        dissimilarity = squareform(dissimilarity)
        Z = hierarchy.linkage(dissimilarity, method="average")
        plot_dendrogram(Z, threshold, df.columns, img_path)
        cluster_ids = hierarchy.fcluster(Z, threshold, criterion="distance")

        # 非類似度のカットオフしきい値に基づくクラスタの分割
        agg_ramanshifts = {}
        for n, _id in enumerate(cluster_ids):
            _id = int(_id)
            agg_ramanshifts.setdefault(_id, [])
            features = df.columns.astype(int)
            agg_ramanshifts[_id].append(features[n])

        return agg_ramanshifts

    def calc_band_distance(each: list):
        # 各クラスタ: each におけるバンド間の物理的距離に基づくクラスタリングを実施
        # 初期化
        distance = np.zeros((len(each), len(each)))
        for i, lband in enumerate(each):
            for j, rband in enumerate(each):
                lband = int(lband)
                rband = int(rband)
                distance[i, j] = np.abs(lband - rband)
        return distance  # 各クラスタのバンド間の距離行列

    def bandwidth_clustering(distance, part_ramanbands, threshold):
        distance = squareform(distance)
        Z = hierarchy.linkage(
            distance,
            method="average",
            metric="euclidean"
        )
        ids = hierarchy.fcluster(
            Z,
            threshold,
            criterion="distance",
        )

        ids_ramanbands = {}
        for _id, r in zip(ids, part_ramanbands):
            _id = int(_id)
            ids_ramanbands.setdefault(_id, [])
            ids_ramanbands[_id].append(r)
        return ids_ramanbands, Z

    def divide_discontinuos(agg_ramanshifts, dist_th):
        # 各クラスタid についてラマンバンド間の距離に基づくクラスタリングを実施する
        max_id = max(agg_ramanshifts.keys())
        update_clusters = []
        remove_ids = []
        for cluster_id, ramanbands in agg_ramanshifts.items():

            # クラスタの要素が1の場合はスキップ
            if len(ramanbands) == 1:
                continue

            # クラスタ内のカイザーに基づく距離行列を計算
            distance = calc_band_distance(ramanbands)

            # 分割対象かもしれないクラスタを抽出する
            if distance.mean() > 10:
                subcluster, Z = bandwidth_clustering(
                    distance,
                    ramanbands,
                    dist_th,
                )
                plot_dendrogram(Z, dist_th, ramanbands, f"img/dist_dendrogram_{cluster_id}.png")

                # 分割対象であるクラスタは、すべてのサブクラスタに新規のクラスタidを払い出す
                for _id, r in subcluster.items():
                    update_clusters.append({max_id + _id: r})
                # 分割により増えたクラスタidの数を更新する
                max_id += len(subcluster)

                # 分割対象の元のクラスタidを削除対象として記録する
                remove_ids.append(cluster_id)

                if img_path is not None:
                    p = Path(img_path)
                    img_path = str(p.with_name(f"{p.stem}_{cluster_id}{p.suffix}"))  # 出力ファイルにクラスタidを挿入
                    plot_dendrogram(Z, dist_th, ramanbands, img_path)

        return update_clusters, remove_ids

    # 相関係数を用いてクラスタリングする
    # クラスタIDに基づいて同一クラスタのラマンシフトをまとめる
    agg_ramanshifts = corr_clustering(df, corr_th)

    # Step 2-after: 分割後のクラスタの再登録と分割前のクラスタの削除
    for s in subs:
        agg_ramanshifts.update(s)
    for d in duplicate_ids:
        agg_ramanshifts.pop(d)

    # ラマンバンドの波長が小さい順にソートし、クラスタidを振りなおす
    agg_ramanshifts = sorted(agg_ramanshifts.items(), key=lambda x: min(x[1]))
    agg_ramanshifts = {i: v[1] for i, v in enumerate(agg_ramanshifts)}

    # Step 3: クラスタ内の代表値を用いて低解像度なラマンスペクトルの再構築
    reduced = pd.DataFrame()
    highests = []
    for _, ramanshifts in agg_ramanshifts.items():
        ramanshifts = [str(r) for r in ramanshifts]

        d = df.loc[:, ramanshifts]

        # クラスタ内ラマンシフトの平均強度の大きさに基づく重みを計算
        weights = d.mean() / d.mean().sum()

        # 代表ラマンシフトとして、重みが最大のラマンシフトを選択
        highests.append(weights.index[weights.argmax()])

        if weighted:
            # 平均化前に重みづけする
            for n, (column, _) in enumerate(d.items()):
                d.loc[:, column] = weights.iloc[n] * d.loc[:, column]

        reduced = pd.concat(
            [reduced, d.mean(axis=1)],
            axis=1,
        )
    reduced.columns = highests
    print(reduced)

    return reduced, agg_ramanshifts


def _peakpick(df, label, distance, img_path=None):
    # 特定の行のみに着目したい場合は、このブロックが実行される
    if label:
        if set(label) <= set(df.index):
            print(label)
        else:
            diff = set(label) - set(df.index)
            raise ValueError(f"{diff} is not in Dataframe")
        _df = df.loc[label, :].copy()
    else:
        _df = df.copy()
    click.echo(_df)

    # 各ラベルで平均化する... 必要はなかった
    '''
    tmp = pd.DataFrame()
    for ll, d in df.groupby(level=0):
        a = d.mean().to_frame().T
        a.index = [ll]
        tmp = pd.concat([tmp, a])
    print(tmp)
    '''

    mean = _df.mean()
    loc, _ = find_peaks(mean, height=0, distance=distance)

    df = df.iloc[:, loc]

    if img_path is not None:
        fig = plt.figure(figsize=(12, 6), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(mean)
        ax.plot(loc, mean.iloc[loc], "x")
        fig.savefig(img_path)
        click.echo(f"Output to {img_path}")

    return df


@click.group()
@click.pass_context
def cmd(ctx):
    # ctx.obj["DEBUG"] = path
    pass


@cmd.command()
@click.argument("path")
@click.option("--label", "-l", multiple=True)
@click.option("--distance", "-d", default=10)
@click.option("--vis", is_flag=True)
@click.option("--output_path", "-op", default="output/peakpick.csv")
@click.option("--img_path", "-ip", default="img/peakpick.png")
@click.pass_context
def peakpick(ctx, path, vis, output_path, label, distance, img_path):
    df = pd.read_csv(path, header=0, index_col=0)
    click.echo(df)

    if not vis:
        img_path = None

    reduced = _peakpick(df, label, distance, img_path)
    reduced.to_csv(output_path)


@cmd.command()
@click.argument("path")
@click.option("--corr_th", default=0.2)
@click.option("--dist_th", default=15)
@click.option("--weighted", is_flag=False)
@click.option("--vis", is_flag=True)
@click.option("--output_path", "-op", default="output/bulkspectra.csv")
@click.option("--img_path", "-ip", default="img/bulkspectra.png")
@click.option("--comp_path", "-cp", default="comp/bulkspectra.json")
@click.pass_context
def bulkspectra(ctx, path, corr_th, dist_th, weighted, vis, output_path, img_path, comp_path):
    df = pd.read_csv(path, header=0, index_col=0)
    click.echo(df)

    reduced, agg_ramanshifts = _bulkspectra(
        df,
        corr_th=corr_th,
        dist_th=dist_th,
        weighted=weighted,
    )
    reduced.to_csv(output_path)

    if comp_path is not None:
        agg_ramanshifts = {k: [int(vv) for vv in v] for k, v in agg_ramanshifts.items()}
        with open(comp_path, "w") as f:
            json.dump(agg_ramanshifts, f, indent=4)

    if img_path is not None:
        import matplotlib.ticker as ticker
        fig = plt.figure(figsize=(12, 12), dpi=300)
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(df.mean())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax = fig.add_subplot(2, 1, 2)
        ax.plot(reduced.mean())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        fig.savefig(img_path)


@cmd.command()
@click.option("--path", "-p", default="testdata.csv")
@click.option("--row", "-r", default=600)
@click.option("--col", "-c", default=1200)
@click.option("--random_seed", "-rs", default=12345)
@click.pass_context
def testdata(ctx, path: str, row: int, col: int, random_seed: int):
    rng = np.random.default_rng(random_seed)
    df = pd.DataFrame(
        rng.random((row, col)),
        columns=[f"{i}" for i in range(col)],
        index=np.random.choice(["A", "B", "C"], row, p=[0.5, 0.3, 0.2])
    )
    df = df.sort_index()
    click.echo(df)
    df.to_csv(path)
    click.echo(f"save to {path}")


def main():
    cmd(obj={})


if __name__ == "__main__":
    main()
