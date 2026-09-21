import 'dart:convert';
import 'dart:developer' as developer;

/// Timings for a single generation.
///
/// [coldStartMs] is only set on the first generation after a model load; it
/// measures load plus first inference, which is what a user actually waits
/// through when opening the app.
class GenerationMetrics {
  final int? coldStartMs;
  final int ttftMs;
  final int genMs;
  final int tokens;

  const GenerationMetrics({
    this.coldStartMs,
    required this.ttftMs,
    required this.genMs,
    required this.tokens,
  });

  double get tokensPerSecond => genMs > 0 ? tokens * 1000 / genMs : 0;

  Map<String, dynamic> toJson() => {
        if (coldStartMs != null) 'cold_start_ms': coldStartMs,
        'ttft_ms': ttftMs,
        'gen_ms': genMs,
        'tokens': tokens,
        'tok_s': double.parse(tokensPerSecond.toStringAsFixed(2)),
      };
}

/// Emits benchmark records to logcat in a format `src/benchmark.py` parses.
///
/// THE CONTRACT. One line per run, prefixed with [marker], the rest valid JSON:
///
///   OHAI_BENCH {"run":1,"model":"model-q4_0.gguf","ttft_ms":412,...}
///
/// `src/benchmark.py` reads these with `adb logcat -d`. If you change the
/// marker or the field names here, change them there too — the Python side
/// validates the field set and will tell you which keys are missing rather
/// than silently reporting zeros.
class BenchLogger {
  static const marker = 'OHAI_BENCH';

  static void emit({
    required int run,
    required String model,
    required GenerationMetrics metrics,
    int? peakRssMb,
  }) {
    final record = <String, dynamic>{
      'run': run,
      'model': model,
      ...metrics.toJson(),
      if (peakRssMb != null) 'peak_rss_mb': peakRssMb,
    };
    // developer.log reaches logcat on Android in both debug and profile builds.
    // Benchmark in PROFILE mode, never debug: debug builds run unoptimised Dart
    // and the numbers are meaningless.
    developer.log('$marker ${jsonEncode(record)}', name: 'ohai');
    // ignore: avoid_print
    print('$marker ${jsonEncode(record)}');
  }
}
