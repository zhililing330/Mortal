import prelude

import argparse
import numpy as np
import torch
import secrets
import os
from os import path
from model import Brain, DQN
from engine import MortalEngine
from libriichi.arena import OneVsThree
from common import load_torch_state
from config import config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=('eval', 'train-data'),
        help='eval writes to eval_log_dir; train-data writes to log_dir for training data generation',
    )
    parser.add_argument('--log-dir', help='override the selected log directory')
    parser.add_argument('--challenger-state-file', help='override [1v3.challenger].state_file')
    parser.add_argument('--games', type=int, help='override [1v3].games_per_iter')
    parser.add_argument('--iters', type=int, help='override [1v3].iters')
    return parser.parse_args()

def static_glob_root(pattern):
    abs_pattern = path.abspath(pattern)
    parts = []
    for part in abs_pattern.split(path.sep):
        if any(char in part for char in '*?['):
            break
        parts.append(part)
    if not parts:
        return path.abspath(path.sep)
    return path.normcase(path.normpath(path.sep.join(parts)))

def is_within(child, parent):
    child = path.normcase(path.abspath(child))
    parent = path.normcase(path.abspath(parent))
    try:
        return path.commonpath([child, parent]) == parent
    except ValueError:
        return False

def log_dir_overlaps_dataset(log_dir):
    log_dir = path.abspath(log_dir)
    return any(
        is_within(log_dir, static_glob_root(pattern))
        for pattern in config['dataset']['globs']
    )

def main():
    args = parse_args()
    cfg = config['1v3']
    mode = args.mode or cfg.get('mode', 'eval')
    if mode not in ('eval', 'train-data'):
        raise ValueError(f'Unexpected [1v3].mode {mode}')

    games_per_iter = args.games or cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = args.iters or cfg['iters']
    if args.log_dir:
        log_dir = args.log_dir
    elif mode == 'eval':
        log_dir = cfg.get('eval_log_dir', 'local_logs/1v3_eval')
    else:
        log_dir = cfg['log_dir']

    if games_per_iter % 4 != 0:
        raise ValueError('games_per_iter must be divisible by 4')
    if mode == 'eval' and log_dir_overlaps_dataset(log_dir):
        raise RuntimeError(
            f'eval log_dir {log_dir!r} overlaps dataset.globs; '
            'use a directory outside the training data glob or run with --mode train-data intentionally'
        )

    print(f'mode: {mode}')
    print(f'log_dir: {path.abspath(log_dir)}')
    use_akochan = cfg['akochan']['enabled']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = load_torch_state(cfg['champion']['state_file'], map_location=torch.device('cpu'))
        cham_cfg = state['config']
        version = cham_cfg['control'].get('version', 1)
        conv_channels = cham_cfg['resnet']['conv_channels']
        num_blocks = cham_cfg['resnet']['num_blocks']
        mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
        dqn = DQN(version=version, **cham_cfg.get('dqn', {})).eval()
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        if cfg['champion']['enable_compile']:
            mortal.compile()
            dqn.compile()
        engine_cham = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = version,
            device = torch.device(cfg['champion']['device']),
            enable_amp = cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard = cfg['champion']['enable_rule_based_agari_guard'],
            name = cfg['champion']['name'],
        )

    challenger_state_file = args.challenger_state_file or cfg['challenger']['state_file']
    state = load_torch_state(challenger_state_file, map_location=torch.device('cpu'))
    chal_cfg = state['config']
    version = chal_cfg['control'].get('version', 1)
    conv_channels = chal_cfg['resnet']['conv_channels']
    num_blocks = chal_cfg['resnet']['num_blocks']
    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version, **chal_cfg.get('dqn', {})).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    if cfg['challenger']['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine_chal = MortalEngine(
        mortal,
        dqn,
        is_oracle = False,
        version = version,
        device = torch.device(cfg['challenger']['device']),
        enable_amp = cfg['challenger']['enable_amp'],
        enable_rule_based_agari_guard = cfg['challenger']['enable_rule_based_agari_guard'],
        name = cfg['challenger']['name'],
    )

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            rankings = env.ako_vs_py(
                engine = engine_chal,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
