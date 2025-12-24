import os
import gzip
import argparse
from collections import defaultdict, Counter, deque
from typing import Tuple, Optional, List, Dict, Set
import random
from tqdm import tqdm


FB_PREFIX = "http://rdf.freebase.com/ns/"


def parse_token(tok: str) -> str:
    """
    解析一个 token：
    - 去掉尖括号 <> 和末尾的句点 .
    - 如果以 FB_PREFIX 开头，则返回去前缀后的内容；否则原样返回
    """
    tok = tok.strip()
    if tok.endswith('.'):
        tok = tok[:-1]
    if tok.startswith('<') and tok.endswith('>'):
        tok = tok[1:-1]
    if tok.startswith(FB_PREFIX):
        tok = tok[len(FB_PREFIX):]
    return tok


def parse_triple_line(line: bytes) -> Optional[Tuple[str, str, str]]:
    """
    从 gzip 行解析三元组。兼容两种格式：
    1) RDF-like：<http://rdf.freebase.com/ns/m.abc> <http://rdf.freebase.com/ns/rel.xyz> <http://rdf.freebase.com/ns/m.def> .
    2) 简单空格分隔：m.abc rel.xyz m.def
    解析后返回 (h, r, t)。若无法解析，返回 None。
    """
    try:
        s = line.decode('utf-8').strip()
    except Exception:
        return None
    if not s:
        return None

    parts = s.split()
    if len(parts) < 3:
        return None

    h = parse_token(parts[0])
    r = parse_token(parts[1])
    t = parse_token(parts[2])
    return h, r, t


def stream_gz(path: str):
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield line


def build_stats(input_gz: str,
                require_fb_prefix: bool = True) -> Tuple[Counter, Counter, int]:
    """
    第一遍：统计实体度数（按出现次数近似）与关系频次，只保留满足前缀条件的三元组进入统计。
    返回：entity_counter, relation_counter, kept_triple_count
    """
    ent_cnt = Counter()
    rel_cnt = Counter()
    kept = 0
    total_bytes = os.path.getsize(input_gz)
    with gzip.open(input_gz, 'rb') as f, tqdm(
        total=total_bytes, unit='B', unit_scale=True, desc='Pass-1 stats'
    ) as pbar:
        for line in f:
            triple = parse_triple_line(line)
            pbar.update(len(line))  # 以压缩字节为基准更新进度
            if not triple:
                continue
            h, r, t = triple
            if require_fb_prefix:
                # 仅保留具有 FB 前缀（或已被剥离成 m.* / g.* 的）实体；
                # 若非 URI，放宽为以 "m." 或 "g." 开头的实体；关系同理放宽为包含点号结构。
                cond_ent = (h.startswith('m.') or h.startswith('g.')) and (t.startswith('m.') or t.startswith('g.'))
                cond_rel = (r.startswith('m.') or r.startswith('g.') or ('.' in r))
                if not (cond_ent and cond_rel):
                    continue
            ent_cnt[h] += 1
            ent_cnt[t] += 1
            rel_cnt[r] += 1
            kept += 1
    return ent_cnt, rel_cnt, kept


def build_adjacency_and_triples(input_gz: str,
                                eligible_ents: Set[str],
                                eligible_rels: Set[str]) -> Tuple[Dict[str, List[Tuple[str, str]]], List[Tuple[str, str, str]]]:
    """
    第二遍：仅收集 (h,t,r) 都在可选集合中的边，构建无向邻接以供 BFS；
    同时保留三元组列表。
    返回：adj, triples
    adj[u] = list of (v, r)
    """
    adj = defaultdict(list)
    triples = []
    total_bytes = os.path.getsize(input_gz)
    with gzip.open(input_gz, 'rb') as f, tqdm(
        total=total_bytes, unit='B', unit_scale=True, desc='Pass-2 filter'
    ) as pbar:
        for line in f:
            triple = parse_triple_line(line)
            pbar.update(len(line))
            if not triple:
                continue
            h, r, t = triple
            if h in eligible_ents and t in eligible_ents and r in eligible_rels:
                adj[h].append((t, r))
                adj[t].append((h, r))  # 供无向 BFS
                triples.append((h, r, t))
    return adj, triples


def bfs_select_entities(adj: Dict[str, List[Tuple[str, str]]],
                        seeds: List[str],
                        target_entities: int) -> Set[str]:
    visited = set()
    q = deque()
    for s in seeds:
        if s in adj:
            visited.add(s)
            q.append(s)
            if len(visited) >= target_entities:
                return visited
    while q and len(visited) < target_entities:
        u = q.popleft()
        for v, _ in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.append(v)
                if len(visited) >= target_entities:
                    break
    return visited


def remap_and_split(triples: List[Tuple[str, str, str]],
                    kept_entities: Set[str],
                    out_dir: str,
                    train_ratio=0.8,
                    valid_ratio=0.1,
                    seed=2025):
    os.makedirs(out_dir, exist_ok=True)

    # 仅保留端点都在 kept_entities 的三元组
    triples = [tr for tr in triples if tr[0] in kept_entities and tr[2] in kept_entities]

    # 映射实体与关系
    entities = sorted({h for h, _, t in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    # 重映射
    remapped = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples]

    # 打乱并划分
    random.seed(seed)
    random.shuffle(remapped)
    n = len(remapped)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    train = remapped[:n_train]
    valid = remapped[n_train:n_train + n_valid]
    test = remapped[n_train + n_valid:]

    def _write(path: str, data: List[Tuple[int, int, int]]):
        with open(path, 'w', encoding='utf-8') as f:
            for h, r, t in data:
                f.write(f"{h} {r} {t}\n")

    # 写出两套命名，兼容加载器
    _write(os.path.join(out_dir, 'kg_train.txt'), train)
    _write(os.path.join(out_dir, 'kg_valid.txt'), valid)
    _write(os.path.join(out_dir, 'kg_test.txt'), test)

    # 保存映射以便复现
    with open(os.path.join(out_dir, 'entities_map.txt'), 'w', encoding='utf-8') as f:
        for e, i in ent2id.items():
            f.write(f"{i}\t{e}\n")
    with open(os.path.join(out_dir, 'relations_map.txt'), 'w', encoding='utf-8') as f:
        for r, i in rel2id.items():
            f.write(f"{i}\t{r}\n")

    return {
        'n_entities': len(entities),
        'n_relations': len(relations),
        'n_triples': len(remapped),
        'n_train': len(train),
        'n_valid': len(valid),
        'n_test': len(test),
    }


def main():
    parser = argparse.ArgumentParser(description='Extract Freebase movie subgraph and split dataset')
    parser.add_argument('--input_gz', type=str, default='freebase_douban.gz', help='Path to gzipped triples file')
    parser.add_argument('--output_dir', type=str, default=os.path.join('data', 'freebase'), help='Output directory')
    parser.add_argument('--min_ent_triples', type=int, default=10, help='Filter entities appearing in <X> triples at least')
    parser.add_argument('--min_rel_triples', type=int, default=10, help='Filter relations appearing in <X> triples at least')
    parser.add_argument('--target_entities', type=int, default=3000, help='Target number of entities in subgraph (>=3000)')
    parser.add_argument('--seed_entities', type=int, default=50, help='Number of top-degree seeds to start BFS')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for shuffle/sampling')
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = args.input_gz
    if not os.path.isabs(input_path):
        input_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input gz file not found: {input_path}")

    # 1) 统计
    print('[Pass-1] Counting entities/relations with Freebase prefix filtering ...')
    ent_cnt, rel_cnt, kept = build_stats(input_path, require_fb_prefix=True)
    print(f'Kept triples for stats: {kept:,}')

    # 2) 频次筛选
    eligible_ents = {e for e, c in ent_cnt.items() if c >= args.min_ent_triples}
    eligible_rels = {r for r, c in rel_cnt.items() if c >= args.min_rel_triples}
    print(f'Eligible entities: {len(eligible_ents):,}, relations: {len(eligible_rels):,}')

    # 3) 建邻接 & 收集候选三元组
    print('[Pass-2] Building adjacency and collecting filtered triples ...')
    adj, triples = build_adjacency_and_triples(input_path, eligible_ents, eligible_rels)
    print(f'Filtered triples collected: {len(triples):,}')

    if not adj:
        raise RuntimeError('Adjacency graph is empty after filtering. Try lowering thresholds.')

    # 4) BFS 选点（以度数前 seed_entities 为种子）
    deg = {u: len(neis) for u, neis in adj.items()}
    top_seeds = [u for u, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:args.seed_entities]]
    kept_entities = bfs_select_entities(adj, top_seeds, max(args.target_entities, 3000))
    print(f'BFS kept entities: {len(kept_entities):,}')

    # 5) 重映射与划分
    stats = remap_and_split(triples, kept_entities, args.output_dir)
    print('Done. Stats:')
    for k, v in stats.items():
        print(f'  - {k}: {v:,}')


if __name__ == '__main__':
    main()
