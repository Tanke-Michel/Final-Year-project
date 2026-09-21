import 'package:flutter/material.dart';

import '../services/bench_logger.dart';
import '../services/llm_service.dart';

/// Day 13 benchmark runner.
///
/// Runs a fixed prompt set N times and emits one OHAI_BENCH line per run to
/// logcat, where `src/benchmark.py` collects them.
///
/// TWO RULES, both of which decide whether the numbers mean anything:
///
///  1. BUILD IN PROFILE MODE. `flutter run --profile`. Debug builds run
///     unoptimised Dart and the figures are meaningless.
///  2. LET IT COOL BETWEEN RUNS. Low-end handsets throttle thermally within
///     seconds. The cooldown is deliberate; do not reduce it to save time.
class BenchmarkScreen extends StatefulWidget {
  final LlmService llm;
  const BenchmarkScreen({super.key, required this.llm});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  // Fixed prompts. Same set every time, or runs are not comparable.
  static const _prompts = [
    'What is metformin used for?',
    'What are the side effects of amoxicillin?',
    'How should insulin be stored?',
    'Does paracetamol interact with alcohol?',
  ];

  int _runs = 5;
  int _cooldownSec = 20;
  bool _running = false;
  int _completed = 0;
  String _log = '';

  void _append(String line) => setState(() => _log = '$_log$line\n');

  Future<void> _run() async {
    setState(() {
      _running = true;
      _completed = 0;
      _log = '';
    });

    _append('model: ${widget.llm.modelId}');
    _append('runs: $_runs, cooldown: ${_cooldownSec}s\n');

    for (var i = 1; i <= _runs; i++) {
      if (!mounted) return;
      final prompt = _prompts[(i - 1) % _prompts.length];

      try {
        // The guard is deliberately bypassed here: this measures raw inference
        // cost, not the guarded turn. Guard overhead is pure string matching
        // and is negligible beside generation, but conflating them would make
        // the reported inference figures wrong.
        await widget.llm.generate(prompt);
        final m = widget.llm.lastMetrics;
        if (m == null) {
          _append('run $i: no metrics — backend is not instrumented');
          continue;
        }
        BenchLogger.emit(run: i, model: widget.llm.modelId, metrics: m);
        _append('run $i  ttft ${m.ttftMs}ms  '
            '${m.tokensPerSecond.toStringAsFixed(1)} tok/s'
            '${m.coldStartMs != null ? '  cold ${m.coldStartMs}ms' : ''}');
      } catch (e) {
        _append('run $i FAILED: $e');
      }

      setState(() => _completed = i);

      if (i < _runs) {
        _append('  cooling ${_cooldownSec}s…');
        await Future<void>.delayed(Duration(seconds: _cooldownSec));
      }
    }

    _append('\ndone — collect with:');
    _append('  python src/benchmark.py --gguf <path> --collect');
    setState(() => _running = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Benchmark')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Card(
            color: Theme.of(context).colorScheme.tertiaryContainer,
            child: const Padding(
              padding: EdgeInsets.all(12),
              child: Text(
                'Build in PROFILE mode (flutter run --profile). Debug builds run '
                'unoptimised Dart and the numbers are meaningless.',
                style: TextStyle(fontSize: 12.5),
              ),
            ),
          ),
          const SizedBox(height: 14),
          Row(children: [
            const Text('Runs'),
            Expanded(
              child: Slider(
                value: _runs.toDouble(),
                min: 3, max: 15, divisions: 12,
                label: '$_runs',
                onChanged: _running ? null : (v) => setState(() => _runs = v.round()),
              ),
            ),
            Text('$_runs'),
          ]),
          Row(children: [
            const Text('Cooldown'),
            Expanded(
              child: Slider(
                value: _cooldownSec.toDouble(),
                min: 0, max: 60, divisions: 12,
                label: '${_cooldownSec}s',
                onChanged: _running ? null : (v) => setState(() => _cooldownSec = v.round()),
              ),
            ),
            Text('${_cooldownSec}s'),
          ]),
          const SizedBox(height: 8),
          FilledButton.icon(
            onPressed: _running ? null : _run,
            icon: const Icon(Icons.speed),
            label: Text(_running ? 'Running $_completed/$_runs…' : 'Run benchmark'),
          ),
          const SizedBox(height: 14),
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(11),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
              ),
              child: SingleChildScrollView(
                child: SelectableText(
                  _log.isEmpty ? 'No runs yet.' : _log,
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 11.5),
                ),
              ),
            ),
          ),
        ]),
      ),
    );
  }
}
