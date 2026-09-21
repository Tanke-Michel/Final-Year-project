import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

import 'bench_logger.dart';

/// Generation settings, loaded from the generated asset.
///
/// These come from `configs/base.yaml` via `mobile/export_config.py`. The app
/// must generate under the same settings used by the evaluation harness — if it
/// does not, the reported quality figures describe a different system than the
/// one being demonstrated.
class ModelConfig {
  final String systemPrompt;
  final double temperature;
  final int maxNewTokens;
  final double topP;
  final int topK;
  final int contextLength;

  const ModelConfig({
    required this.systemPrompt,
    required this.temperature,
    required this.maxNewTokens,
    required this.topP,
    required this.topK,
    required this.contextLength,
  });

  static Future<ModelConfig> load(
      {String asset = 'assets/model_config.json'}) async {
    final j = jsonDecode(await rootBundle.loadString(asset)) as Map<String, dynamic>;
    final g = j['generation'] as Map<String, dynamic>;
    return ModelConfig(
      systemPrompt: j['system_prompt'] as String,
      temperature: (g['temperature'] as num).toDouble(),
      maxNewTokens: g['max_new_tokens'] as int,
      topP: (g['top_p'] as num).toDouble(),
      topK: g['top_k'] as int,
      contextLength: (j['context_length'] as num?)?.toInt() ?? 1024,
    );
  }
}

enum ModelStatus { notLoaded, loading, ready, missing, error }

/// Inference backend.
///
/// Deliberately an interface. The Flutter app can be built, installed and
/// demonstrated against [MockLlmService] before any native binding works, which
/// moves the riskiest part of Day 12 off the critical path: if the llama.cpp
/// bridge is not ready, the app still runs and the only thing missing is real
/// generation.
abstract class LlmService {
  ModelStatus get status;
  String get backendName;
  String? get lastError;

  /// Timings from the most recent [generate] call; null before the first.
  ///
  /// Held as state rather than returned, so that [generate] keeps the
  /// `Future<String> Function(String)` signature the guard callback requires.
  /// Instrumentation must not widen the interface the safety layer depends on.
  GenerationMetrics? get lastMetrics;

  /// Identifier of the loaded model, recorded in each benchmark row.
  String get modelId;

  Future<void> load();
  Future<String> generate(String userText);
  Future<void> dispose();

  /// Where the model file is expected. It is downloaded or side-loaded into
  /// app storage rather than bundled in the APK — a 400 MB APK is painful to
  /// build, transfer and reinstall a dozen times during development.
  static Future<File> modelFile(String filename) async {
    final dir = await getApplicationSupportDirectory();
    return File('${dir.path}/models/$filename');
  }
}

/// Deterministic stand-in. Answers from a small fixed table, and deliberately
/// emits some degraded output so the guard's output rules can be exercised on
/// device before a real model exists.
class MockLlmService implements LlmService {
  ModelStatus _status = ModelStatus.notLoaded;
  String? _err;
  final Random _rng;
  GenerationMetrics? _metrics;
  bool _first = true;

  MockLlmService({int seed = 42}) : _rng = Random(seed);

  @override
  ModelStatus get status => _status;
  @override
  String get backendName => 'mock (no model)';
  @override
  String? get lastError => _err;
  @override
  GenerationMetrics? get lastMetrics => _metrics;
  @override
  String get modelId => 'mock';

  static const _answers = <String, String>{
    'metformin':
        'Metformin is used to help control blood sugar in type 2 diabetes. It works '
        'mainly by reducing the amount of sugar released by the liver and helping the '
        'body respond better to its own insulin. Speak to a pharmacist or doctor about '
        'how it applies to your situation.',
    'amoxicillin':
        'Amoxicillin is an antibiotic used for bacterial infections. Commonly reported '
        'effects include nausea, diarrhoea and skin rash. Seek medical help promptly if '
        'you develop a spreading rash, facial swelling or difficulty breathing.',
    'paracetamol':
        'Paracetamol is used to relieve pain and reduce fever. Regularly drinking a lot '
        'of alcohol alongside it increases the risk of liver damage, because both are '
        'processed by the liver.',
    'omeprazole':
        'Omeprazole reduces the amount of acid the stomach produces, by blocking the '
        'pumps in the stomach lining that release acid. Less acid gives irritated tissue '
        'a chance to heal.',
  };

  @override
  Future<void> load() async {
    _status = ModelStatus.loading;
    await Future<void>.delayed(const Duration(milliseconds: 350));
    _status = ModelStatus.ready;
  }

  @override
  Future<String> generate(String userText) async {
    final sw = Stopwatch()..start();
    await Future<void>.delayed(const Duration(milliseconds: 500));
    final ttft = sw.elapsedMilliseconds;
    final t = userText.toLowerCase();

    for (final e in _answers.entries) {
      if (t.contains(e.key)) {
        _record(sw, ttft, e.value);
        // One response in eight is degraded, so the output rules (OUT-01
        // numeric dose, OUT-04 degenerate) are exercised on device rather than
        // only in the Python tests.
        if (_rng.nextInt(8) == 0) {
          return _rng.nextBool()
              ? 'Take 500 mg twice a day.'
              : 'ok';
        }
        return e.value;
      }
    }
    const fallback = 'I do not have information about that medicine. Please ask '
        'a pharmacist or another health professional.';
    _record(sw, ttft, fallback);
    return fallback;
  }

  void _record(Stopwatch sw, int ttft, String out) {
    _metrics = GenerationMetrics(
      coldStartMs: _first ? sw.elapsedMilliseconds + 350 : null,
      ttftMs: ttft,
      genMs: sw.elapsedMilliseconds,
      tokens: out.split(RegExp(r'\s+')).length,
    );
    _first = false;
  }

  @override
  Future<void> dispose() async => _status = ModelStatus.notLoaded;
}

/// Real backend. Wire this to whichever binding the Day 3 spike proved out.
///
/// The engine choice fixes the deployable quantization format, which fixes what
/// the quantization-aware training had to simulate. See docs/PIPELINE.md and
/// src/quantize.py — a mismatch there invalidates the headline result, so do not
/// change the model file format here without re-running the scheme check.
class LlamaCppService implements LlmService {
  final ModelConfig config;
  final String modelFilename;

  ModelStatus _status = ModelStatus.notLoaded;
  String? _err;
  GenerationMetrics? _metrics;

  LlamaCppService({required this.config, this.modelFilename = 'model-q4_0.gguf'});

  @override
  ModelStatus get status => _status;
  @override
  String get backendName => 'llama.cpp · $modelFilename';
  @override
  String? get lastError => _err;
  @override
  GenerationMetrics? get lastMetrics => _metrics;
  @override
  String get modelId => modelFilename;

  @override
  Future<void> load() async {
    _status = ModelStatus.loading;
    try {
      final f = await LlmService.modelFile(modelFilename);
      if (!await f.exists()) {
        _status = ModelStatus.missing;
        _err = 'Model not found at ${f.path}. Side-load it with:\n'
            'adb push model-q4_0.gguf <app support dir>/models/';
        return;
      }

      // TODO(day-12): initialise the native context here.
      //   - load the GGUF at f.path
      //   - set context length to config.contextLength (drives KV cache, which
      //     drives peak RAM — keep it small on a 3 GB device)
      //   - keep the context on a background isolate so the UI thread does not
      //     block during generation
      throw UnimplementedError(
          'Native binding not wired yet. Run the app with MockLlmService until '
          'the Day 3 toolchain spike is complete.');
    } catch (e) {
      _status = ModelStatus.error;
      _err = e.toString();
    }
  }

  @override
  Future<String> generate(String userText) async {
    if (_status != ModelStatus.ready) {
      throw StateError('Model not ready (status: $_status)');
    }
    // TODO(day-12): build the prompt from config.systemPrompt + userText using
    // the model's chat template, then sample with config.temperature, topP,
    // topK and maxNewTokens. These MUST match configs/base.yaml.
    //
    // Populate _metrics while sampling:
    //   - start a Stopwatch before the first decode call
    //   - record ttftMs when the FIRST token is emitted
    //   - record genMs and the token count when sampling ends
    // Time to first token and steady-state throughput behave differently on a
    // low-end device and must be reported separately, not averaged together.
    throw UnimplementedError();
  }

  @override
  Future<void> dispose() async {
    _status = ModelStatus.notLoaded;
  }
}
