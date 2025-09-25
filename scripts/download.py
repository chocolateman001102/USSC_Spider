from pathlib import Path
from convokit import Corpus, download

import ssl, certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

def get_valid_corpus_path() -> Path:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) 目标下载到项目 data 目录
    p = Path(download("supreme-corpus", data_dir=str(data_dir), use_newest_version=True))
    if p.exists():
        return p

    # 2) 若路径不存在，尝试使用本地缓存定位（如无本地缓存则继续）
    try:
        p = Path(download("supreme-corpus", data_dir=str(data_dir), use_local=True))
        if p.exists():
            return p
    except FileNotFoundError:
        pass

    # 3) 清掉全局缓存记录，强制重新下载
    cached = Path("~/.convokit/downloads/downloaded.txt").expanduser()
    if cached.exists():
        try:
            cached.unlink()
        except Exception:
            pass
    p = Path(download("supreme-corpus", data_dir=str(data_dir), use_newest_version=True))

    # 4) 某些版本可能返回父目录；尝试常见的子目录组合
    if not p.exists():
        candidates = [
            p,
            p / "supreme-corpus",
            p.parent / "supreme-corpus",
        ]
        for c in candidates:
            if c.exists():
                return c
    return p


if __name__ == "__main__":
    corpus_path = get_valid_corpus_path()
    if not corpus_path.exists():
        raise FileNotFoundError(f"ConvoKit returned a non-existent path: {corpus_path}")
    corpus = Corpus(filename=str(corpus_path))