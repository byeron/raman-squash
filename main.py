import click
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
import json


def _bulkspectra(df, threshold=15, weighted=False):
    '''
    処理の手順
    1. ラマンスペクトル間の相関を計算し、クラスタリングを実施する
        隣り合うバンドは物理的に相関が高くなる傾向を持つ
    2. 1で得られたクラスタに対して、物理的な距離が近い要素を更にまとめる
        1のクラスタでは飛び地のバンドも同一クラスタに属する可能性がある
        飛び地のバンドを別クラスタに分割する
    3. 2段階の処理で得られたクラスタについて、クラスタ内変数の重みを計算する
        加重平均 of 平均
    4. 各クラスタを新たな特徴量とし、重みに基づいて代表値を計算する
    5. ラマンバンドの並びに基づき、新たな特徴量を昇順で並べる
    '''

    def corr_clustering(df, threshold=0.2):
        dissimilarity = 1 - (df.corr() + 1) / 2
        dissimilarity = squareform(dissimilarity)
        Z = hierarchy.linkage(dissimilarity, method="average")
        cluster_ids = hierarchy.fcluster(Z, threshold, criterion="distance")

        # IDに基づいて同一クラスタのラマンシフトをまとめる
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
        return ids_ramanbands

    def divide_discontinuos(agg_ramanshifts):
        # 各クラスタid についてラマンバンド間の距離に基づくクラスタリングを実施する
        max_id = max(agg_ramanshifts.keys())
        update_clusters = []
        remove_ids = []
        for cluster_id, ramanbands in agg_ramanshifts.items():

            # あるクラスタidに属するラマンバンドが1つの場合は分割不要なのでスキップ
            if len(ramanbands) == 1:
                continue

            distance = calc_band_distance(ramanbands)
            if distance.mean() > 10:
                # もし平均距離が大きい場合は分割対象とみなす
                subcluster = bandwidth_clustering(
                    distance,
                    ramanbands,
                    threshold,
                )

                # サブクラスタに新たなidを払い出す
                for _id, r in subcluster.items():
                    update_clusters.append({max_id + _id: r})
                # 分割されたことで増えたクラスタidの数を更新する
                max_id += len(subcluster)

                # 分割前のクラスタidは重複しないように削除する
                remove_ids.append(cluster_id)
        return update_clusters, remove_ids

    # 相関係数を用いてクラスタリングする
    # クラスタIDに基づいて同一クラスタのラマンシフトをまとめる
    agg_ramanshifts = corr_clustering(df)

    # 各クラスタのバンド間の距離に基づきクラスタを分割し、
    # 分割後のサブクラスタ群と、重複するクラスタIDを返却する
    subs, duplicate_ids = divide_discontinuos(agg_ramanshifts)
    for s in subs:
        agg_ramanshifts.update(s)
    for d in duplicate_ids:
        agg_ramanshifts.pop(d)

    # ラマンバンドの波長が小さい順にソートする
    agg_ramanshifts = sorted(agg_ramanshifts.items(), key=lambda x: min(x[1]))

    # cluster idを振り直す
    agg_ramanshifts = {i: v[1] for i, v in enumerate(agg_ramanshifts)}

    reduced = pd.DataFrame()
    highests = []
    for _, ramanshifts in agg_ramanshifts.items():
        ramanshifts = [str(r) for r in ramanshifts]

        d = df.loc[:, ramanshifts]

        # 各クラスタにおける平均強度ベースの重みを計算する
        weights = d.mean() / d.mean().sum()
        # クラスタ内で最も強度が高いラマンバンドをスペクトル名の代表値とする
        highests.append(weights.index[weights.argmax()])

        if weighted:
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
@click.option("--dist_th", default=15)
@click.option("--weighted", is_flag=False)
@click.option("--vis", is_flag=True)
@click.option("--output_path", "-op", default="output/bulkspectra.csv")
@click.option("--img_path", "-ip", default="img/bulkspectra.png")
@click.option("--comp_path", "-cp", default="comp/bulkspectra.json")
@click.pass_context
def bulkspectra(ctx, path, dist_th, weighted, vis, output_path, img_path, comp_path):
    df = pd.read_csv(path, header=0, index_col=0)
    click.echo(df)

    reduced, agg_ramanshifts = _bulkspectra(
        df,
        threshold=dist_th,
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
