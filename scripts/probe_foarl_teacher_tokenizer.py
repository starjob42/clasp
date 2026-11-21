"""CPU-only synthetic teacher-tokenization/causal-alignment diagnostic.

Loads only a supplied local tokenizer, never model weights or task datasets.
Receipt contains identities and aggregate checks, not synthetic or real text.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'src'))
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja')
SOURCE_FILES = ('scripts/probe_foarl_teacher_tokenizer.py',
                'src/clasp/trainers/verified_teacher_grpo.py',
                'src/clasp/trainers/primal_dual_grpo.py', 'src/clasp/data/prompts.py',
                'src/clasp/rewards/co_tasks.py', 'src/clasp/rewards/verified_teacher_selection.py',
                'src/clasp/rewards/cvrp_verified_improvement.py',
                'src/clasp/rewards/verified_improvement.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'expected regular source/tokenizer file')
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def synthetic_cases(task):
    if task == 'mis':
        return [({'num_nodes': n, 'edges': []},
                 'Set: [' + ', '.join(map(str, range(n))) + '], Objective: 9999',
                 'Set: [' + ', '.join(map(str, range(n))) + f'], Objective: {n}')
                for n in (3, 5, 8)]
    return [
        ([[[0, 0], [1, 0], [2, 0]], [0, 1, 1], 2],
         'Routes: [[0, 1, 2, 0]], Objective: 9999',
         'Routes: [[0, 1, 2, 0]], Objective: 4.000000'),
        ([[[0, 0], [1, 0], [1, 1], [0, 1]], [0, 1, 1, 1], 3],
         'Routes: [[0, 1, 2, 3, 0]], Objective: 9999',
         'Routes: [[0, 1, 2, 3, 0]], Objective: 4.000000'),
        ([[[0, 0], [1, 0], [2, 0]], [0, 1, 1], 1],
         'Routes: [[0, 1, 0], [0, 2, 0]], Objective: 9999',
         'Routes: [[0, 1, 0], [0, 2, 0]], Objective: 6.000000'),
    ]


def make_trainer(tokenizer, torch, trainer_class):
    class CausalLogitSentinel(torch.nn.Module):
        """Deterministic position/current-token logits, without learned weights.

        Each position depends only on that position's input token and index.
        The diagnostic oracle computes the next-token probability analytically.
        """
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
            self.config = SimpleNamespace(vocab_size=len(tokenizer))
            self.calls = 0

        def forward(self, input_ids, attention_mask, logits_to_keep=None, use_cache=False):
            require(input_ids.device.type == 'cpu' and attention_mask.shape == input_ids.shape, 'unexpected sentinel inputs')
            require(use_cache is False, 'sentinel cannot use generation cache')
            self.calls += 1
            width = input_ids.shape[1]
            keep = width if logits_to_keep is None else logits_to_keep
            positions = torch.arange(width-keep, width, dtype=torch.float64)
            current = input_ids[:, -keep:]
            factors = .3 + positions.remainder(17)[None, :]/23 + current.remainder(29).double()/41
            vocabulary = torch.arange(self.config.vocab_size, dtype=torch.float64)/(self.config.vocab_size-1)
            logits = factors[:, :, None] * vocabulary[None, None, :] * self.anchor
            return SimpleNamespace(logits=logits)

    trainer = object.__new__(trainer_class)
    trainer.processing_class = tokenizer
    trainer.model = CausalLogitSentinel()
    trainer.args = SimpleNamespace(report_to=[], gradient_accumulation_steps=3,
                                  steps_per_generation=3, per_device_train_batch_size=3, num_iterations=1)
    trainer.accelerator = SimpleNamespace(num_processes=1, process_index=0, gather=lambda value: value)
    trainer._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    trainer.grpo_verified_improvement = True
    trainer.grpo_verified_improvement_weight = .05
    trainer.grpo_verified_improvement_max_tokens = 3000
    trainer.grpo_verified_improvement_max_passes = 2
    trainer.grpo_verified_teacher_selection = 'feasibility_incumbent'
    trainer.grpo_verified_teacher_normalization = 'optimizer_window'
    trainer.num_generations = 9
    trainer.num_iterations = 1
    trainer.current_gradient_accumulation_steps = 3
    trainer.temperature = .7
    trainer.model_kwarg_keys = ['logits_to_keep']
    trainer._step = 0
    trainer._buffered_inputs = None
    return trainer


def analytic_nll(prompt, target, vocabulary, temperature, *, shifted=False):
    """Independent geometric-series normalizer and causal label positions."""
    values = []
    combined = prompt + target
    for offset, label in enumerate(target):
        position = len(prompt) - 1 + offset
        if shifted:
            position += 1
        factor = (.3 + (position % 17)/23 + (combined[position] % 29)/41)/temperature
        spacing = factor/(vocabulary-1)
        log_normalizer = math.log(math.expm1(spacing*vocabulary)) - math.log(math.expm1(spacing))
        values.append(log_normalizer - spacing*label)
    return sum(values)/len(values)


def probe_task(task, tokenizer, torch, trainer_class, stock_trainer, template):
    trainer = make_trainer(tokenizer, torch, trainer_class)
    cases = synthetic_cases(task)
    eos = tokenizer.eos_token_id
    require(type(eos) is int and eos >= 0 and tokenizer.pad_token_id is not None, 'tokenizer lacks EOS or PAD')
    expected = [tokenizer.encode(target, add_special_tokens=False, truncation=False) + [eos]
                for _, _, target in cases]
    require(all(ids[-2] != eos for ids in expected), 'synthetic target unexpectedly already ends in EOS')
    prompts, inputs, raw, expected_rows = [], [], [], []
    for case_id, (instance, completion, _) in enumerate(cases):
        extra = ' Synthetic padding-length marker.' * (case_id * 7)
        prompt = template.format('Solve this synthetic ' + task + ' instance.', json.dumps(instance) + extra)
        require(prompt.endswith('### Response:'), 'Jiang prompt response boundary differs')
        for repeat in range(3):
            prompts.append(prompt)
            inputs.append({'task_type': task, 'instance': instance, 'prompt': prompt})
            # One infeasible source activates incumbent supervision while the
            # verified raw incumbent supplies an independently known target.
            source_text = completion if repeat < 2 else 'Synthetic missing solution.'
            raw.append(tokenizer.encode(source_text, add_special_tokens=False, truncation=False) + [eos])
            expected_rows.append(expected[case_id])
    pad = tokenizer.pad_token_id
    width = max(map(len, raw))
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True,
                        max_length=1024, return_tensors='pt')
    lengths = encoded['attention_mask'].sum(1)
    require(len(set(lengths.tolist())) == 3 and lengths.max() < 1024, 'uneven untruncated prompt fixture failed')
    for mask in encoded['attention_mask'].tolist():
        require(mask == [0]*mask.count(0)+[1]*mask.count(1), 'prompts are not left padded')
    output = {'prompt_ids': encoded['input_ids'], 'prompt_mask': encoded['attention_mask'],
              'completion_ids': torch.tensor([ids + [pad]*(width-len(ids)) for ids in raw]),
              'completion_mask': torch.tensor([[1]*len(ids)+[0]*(width-len(ids)) for ids in raw]),
              'advantages': torch.arange(9, dtype=torch.float64),
              'old_per_token_logps': torch.full((9, width), -.5, dtype=torch.float64),
              'synthetic_row_id': torch.arange(9)}
    original = {key: value.clone() for key, value in output.items()}
    backend_state = (tokenizer.backend_tokenizer.padding, tokenizer.backend_tokenizer.truncation)
    trainer._attach_verified_improvement_tensors(inputs, output, [task]*9)
    require(backend_state == (tokenizer.backend_tokenizer.padding, tokenizer.backend_tokenizer.truncation), 'teacher encoding mutated tokenizer caches')
    require(all(torch.equal(output[key], value) for key, value in original.items()), 'attachment changed original rollout tensors')
    prefix = 'grpo_verified_' + task
    ids, mask = output[prefix+'_completion_ids'], output[prefix+'_completion_mask']
    selected = [row for row in range(9) if mask[row].any()]
    require(len(selected) == 3 and len({row//3 for row in selected}) == 3, 'expected one teacher for each synthetic group')
    for row in selected:
        tokens = ids[row][mask[row].bool()].tolist()
        require(tokens == expected_rows[row], 'canonical target prefix/body/objective/EOS was altered')
        require(tokenizer.decode(tokens[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False) == cases[row//3][2], 'canonical target text did not round trip')
        require(tokens[-1] == eos and mask[row, len(tokens)-1] == 1, 'EOS is not supervised')
        require(trainer._tokenize_verified_target(cases[row//3][2], len(tokens)-1) == [], 'overlong target was truncated instead of rejected')
    require(bool((output[prefix+'_window_count'] == 3).all()), 'window count differs')
    require(backend_state == (tokenizer.backend_tokenizer.padding, tokenizer.backend_tokenizer.truncation), 'negative length check mutated tokenizer caches')
    # Stock TRL owns the actual shuffle and three-way split. Only generation is
    # replaced by our already attached synthetic generation buffer.
    trainer._generate_and_score_completions = lambda _: output
    torch.manual_seed(20260909)
    chunks = [stock_trainer._prepare_inputs(trainer, inputs) for _ in range(3)]
    observed_rows = []
    accumulated_loss = 0.
    observed_teacher_count = 0
    for chunk in chunks:
        row_ids = chunk['synthetic_row_id'].tolist()
        observed_rows.extend(row_ids)
        for local, original_row in enumerate(row_ids):
            for key, value in output.items():
                require(torch.equal(chunk[key][local], value[original_row]), 'TRL shuffle/split lost row alignment')
        observed_teacher_count += int((chunk[prefix+'_completion_mask'].sum(1) > 0).sum())
        loss, _ = trainer._compute_improvement_distillation_losses(
            trainer.model, chunk, chunk['prompt_ids'], chunk['prompt_mask'],
            chunk['completion_ids'], torch.zeros(1, dtype=torch.float64))
        require(loss is not None and torch.isfinite(loss), 'teacher loss is invalid')
        accumulated_loss += float(loss.detach())/3
    require(sorted(observed_rows) == list(range(9)) and observed_rows != list(range(9)), 'actual TRL shuffle was not exercised')
    require(observed_teacher_count == 3, 'split changed teacher count')
    expected_nll, wrong_nll = [], []
    for row in selected:
        padded_prompt = output['prompt_ids'][row].tolist()
        target = expected_rows[row]
        expected_nll.append(analytic_nll(padded_prompt, target, len(tokenizer), trainer.temperature))
        wrong_nll.append(analytic_nll(padded_prompt, target, len(tokenizer), trainer.temperature, shifted=True))
    oracle = .05*sum(expected_nll)/3
    require(math.isclose(accumulated_loss, oracle, rel_tol=1e-11, abs_tol=1e-11), 'causal next-token likelihood/masking differs from analytic oracle')
    require(abs(oracle - .05*sum(wrong_nll)/3) > 1e-6, 'fixture cannot detect an off-by-one causal shift')
    return {'task': task, 'synthetic_prompt_groups': 3, 'synthetic_rows': 9,
            'selected_complete_teachers': 3, 'trl_microbatches': 3,
            'prompt_token_length_min': int(lengths.min()), 'prompt_token_length_max': int(lengths.max()),
            'teacher_token_length_min': min(map(len, expected)), 'teacher_token_length_max': max(map(len, expected)),
            'causal_sentinel_forward_calls': trainer.model.calls,
            'checks': {'canonical_prefix_body_objective_and_eos_retained': True,
                       'eos_in_loss_mask': True, 'overlong_targets_rejected_whole': True,
                       'tokenizer_backend_caches_unchanged': True, 'original_rollout_tensors_unchanged': True,
                       'actual_trl_shuffle_and_split_alignment': True, 'uneven_left_padding_exercised': True,
                       'causal_next_token_analytic_oracle': True, 'off_by_one_negative_sensitive': True}}


def run(args):
    source_before = {name: identity(SOURCE/name) for name in SOURCE_FILES}
    import tokenizers
    import torch
    import transformers
    import trl
    from transformers import AutoTokenizer
    from trl import GRPOTrainer
    from trl.trainer.grpo_trainer import shuffle_sequence_dict, split_tensor_dict

    from clasp.data.prompts import JIANG_F638_INFERENCE_TEMPLATE
    from clasp.trainers.verified_teacher_grpo import VerifiedTeacherGRPOTrainer

    torch.set_num_threads(1)
    directory = Path(args.tokenizer).expanduser().resolve(strict=True)
    require(directory.is_dir(), 'tokenizer directory is missing')
    before = {name: identity(directory/name) for name in TOKENIZER_FILES if (directory/name).exists() or (directory/name).is_symlink()}
    require({'tokenizer.json', 'tokenizer_config.json'} <= set(before), 'fast-tokenizer files are required')
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False,
                                            use_fast=True, padding_side='left')
    require(tokenizer.is_fast, 'probe requires the actual fast tokenizer')
    tasks = ('cvrp', 'mis') if args.task == 'both' else (args.task,)
    cases = [probe_task(task, tokenizer, torch, VerifiedTeacherGRPOTrainer, GRPOTrainer,
                        JIANG_F638_INFERENCE_TEMPLATE) for task in tasks]
    after = {name: identity(directory/name) for name in TOKENIZER_FILES if (directory/name).exists() or (directory/name).is_symlink()}
    require(before == after, 'tokenizer files changed during diagnostic')
    require(Path(inspect.getfile(VerifiedTeacherGRPOTrainer)).resolve() ==
            SOURCE/'src/clasp/trainers/verified_teacher_grpo.py', 'trainer import escaped supplied source')
    source_after = {name: identity(SOURCE/name) for name in SOURCE_FILES}
    require(source_before == source_after, 'diagnostic source changed while running')
    sources = [*(SOURCE/name for name in SOURCE_FILES),
               Path(inspect.getfile(GRPOTrainer)), Path(inspect.getfile(shuffle_sequence_dict)),
               Path(inspect.getfile(split_tensor_dict))]
    result = {'status': 'passed_synthetic_foarl_teacher_tokenizer_probe',
              'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': 'local_tsp_tokenizer_fallback_not_task_sft_proof' if args.local_tokenizer_fallback else 'supplied_task_sft_tokenizer_synthetic_only',
              'local_tokenizer_fallback': args.local_tokenizer_fallback,
              'tokenizer': {'directory': str(directory), 'class': tokenizer.__class__.__name__,
                            'files': before, 'files_unchanged': True, 'padding_side': tokenizer.padding_side,
                            'vocabulary_size': len(tokenizer), 'pad_token_id': tokenizer.pad_token_id,
                            'eos_token_id': tokenizer.eos_token_id},
              'source_files': [identity(path) for path in dict.fromkeys(sources)],
              'source_files_unchanged': True,
              'source_manifest': identity(SOURCE/'SOURCE_MANIFEST.sha256')
              if (SOURCE/'SOURCE_MANIFEST.sha256').exists() else None,
              'jiang_inference_template_sha256': hashlib.sha256(JIANG_F638_INFERENCE_TEMPLATE.encode()).hexdigest(),
              'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__,
                          'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                          'trl': trl.__version__, 'cpu_threads': 1},
              'cases': cases, 'gpu_used': False, 'model_weights_loaded': False, 'real_data_loaded': False,
              'completion_or_prompt_text_exported': False, 'distributed_consensus_claim': False,
              'limitation': 'CPU synthetic tokenizer and causal-label check; does not certify native model/DDP gradients or training quality.'}
    result['receipt_sha256'] = canonical_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--task', choices=('cvrp', 'mis', 'both'), required=True)
    parser.add_argument('--local-tokenizer-fallback', choices=('tsp',))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
