import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

/// Deterministic medication-safety guard — Dart port.
///
/// This is a line-for-line port of `src/safety/guard.py`. It loads the SAME
/// `rules.json`, unchanged, bundled as a Flutter asset. Do not fork the rules:
/// one file, two runtimes.
///
/// WHY THE PARITY TEST MATTERS
/// The safety numbers in your results chapter are produced by the Python guard
/// on your laptop. If this port behaves differently, the app you demonstrate is
/// not the system you measured, and the safety results do not describe the
/// deliverable. `test/guard_test.dart` replays a golden corpus generated from
/// the Python implementation and fails on any divergence. Run it before Day 12
/// integration, and again before the defence.
///
/// Design constraints carried over unchanged:
///   - No model inference. Pure string and regex matching.
///   - Fails closed: on any internal error, block.
///   - Every intervention returns the rule id, for the intervention log.
class GuardResult {
  final bool blocked;
  final String? category;
  final String? ruleId;
  final String text;
  final List<String> matchedTerms;

  const GuardResult({
    this.blocked = false,
    this.category,
    this.ruleId,
    this.text = '',
    this.matchedTerms = const [],
  });

  /// Row for the intervention log. Deliberately omits the user's text — the log
  /// counts rule fires; it does not keep health questions on disk.
  Map<String, dynamic> toLogRecord() => {
        'blocked': blocked,
        'category': category,
        'rule_id': ruleId,
        'matched_terms': matchedTerms,
      };
}

class Guard {
  final Map<String, dynamic> _rules;
  final Map<String, dynamic> _tables;
  final Map<String, String> _responses;
  final String disclaimer;
  final List<Map<String, dynamic>> _inputRules;
  final List<Map<String, dynamic>> _outputRules;
  final Map<String, RegExp> _compiled;

  // Normalisation tables, read from rules.json so Python and Dart agree.
  final Map<String, String> _foldMap;
  final Map<String, String> _leetMap;
  final String _separatorChars;
  final int _rejoinMinRun;

  Guard._(this._rules)
      : _tables = (_rules['normalisation'] as Map<String, dynamic>? ?? {}),
        _responses = Map<String, String>.from(_rules['responses'] as Map),
        disclaimer = _rules['disclaimer'] as String,
        _inputRules = (_rules['input_rules'] as List)
            .cast<Map<String, dynamic>>()
            .toList()
          // Sort on (priority, id). Dart's List.sort is NOT guaranteed stable,
          // so rules sharing a priority could otherwise evaluate in a different
          // order than in Python. The id tie-break makes the key total.
          ..sort((a, b) {
            final p = ((a['priority'] ?? 99) as int)
                .compareTo((b['priority'] ?? 99) as int);
            return p != 0 ? p : (a['id'] as String).compareTo(b['id'] as String);
          }),
        _outputRules = (_rules['output_rules'] as List).cast<Map<String, dynamic>>(),
        _compiled = {
          for (final r in (_rules['output_rules'] as List).cast<Map<String, dynamic>>())
            if (r.containsKey('regex'))
              r['id'] as String: RegExp(r['regex'] as String, caseSensitive: false)
        },
        _foldMap = Map<String, String>.from(
            (_rules['normalisation']?['fold_map'] as Map?) ?? {}),
        _leetMap = Map<String, String>.from(
            (_rules['normalisation']?['leet_map'] as Map?) ?? {}),
        _separatorChars =
            (_rules['normalisation']?['separator_chars'] as String?) ?? '_-*.',
        _rejoinMinRun = (_rules['normalisation']?['rejoin_min_run'] as int?) ?? 3;

  /// Load from a bundled asset. Register in pubspec.yaml under assets.
  static Future<Guard> load({String asset = 'assets/rules.json'}) async {
    final raw = await rootBundle.loadString(asset);
    return Guard._(jsonDecode(raw) as Map<String, dynamic>);
  }

  /// Load from a JSON string — used by the parity test, which has no asset
  /// bundle available.
  static Guard fromJsonString(String raw) =>
      Guard._(jsonDecode(raw) as Map<String, dynamic>);

  // ------------------------------------------------------------ normalise

  /// Must match `normalise()` in guard.py step for step. The order is fixed by
  /// rules.json -> normalisation.order and changing it here alone will break
  /// parity.
  String normalise(String text) {
    var t = text.toLowerCase();

    // Accent folding. Dart has no core NFKD, which is exactly why the fold map
    // is data rather than code — francophone input ("problème", "à jeun") does
    // not match without it.
    if (_foldMap.isNotEmpty) {
      final buf = StringBuffer();
      for (final ch in t.split('')) {
        buf.write(_foldMap[ch] ?? ch);
      }
      t = buf.toString();
    }

    _leetMap.forEach((src, dst) => t = t.replaceAll(src, dst));

    final sepClass = RegExp('[${RegExp.escape(_separatorChars)}]+');
    t = t.replaceAll(sepClass, ' ');
    t = t.replaceAll(RegExp(r'\s+'), ' ');

    // Rejoin letter-spaced words: "l e t h a l" -> "lethal".
    // Note: Dart's \w is ASCII-only whereas Python's is Unicode-aware. That
    // would diverge on accented input — except folding runs first, so by this
    // point no accented characters remain. Do not reorder the steps.
    final rejoin = RegExp(r'\b(?:\w ){' '${_rejoinMinRun - 1}' r',}\w\b');
    t = t.replaceAllMapped(rejoin, (m) => m.group(0)!.replaceAll(' ', ''));

    return t.trim();
  }

  // ---------------------------------------------------------------- input

  GuardResult checkInput(String text) {
    try {
      final probe = normalise(text);

      for (final rule in _inputRules) {
        var matched = _matchAnyOf(probe, rule['any_of']);
        matched ??= _matchAllOf(probe, rule['all_of']);
        if (matched != null) {
          return GuardResult(
            blocked: true,
            category: rule['category'] as String?,
            ruleId: rule['id'] as String?,
            text: _responses[rule['response_key']] ?? _responses['unavailable']!,
            matchedTerms: matched,
          );
        }
      }
      return GuardResult(blocked: false, text: text);
    } catch (_) {
      // Fail closed. A crashed guard must never become an open gate.
      return GuardResult(
        blocked: true,
        category: 'guard_error',
        ruleId: 'ERR-00',
        text: _responses['unavailable']!,
      );
    }
  }

  List<String>? _matchAnyOf(String probe, dynamic terms) {
    if (terms == null) return null;
    for (final t in (terms as List).cast<String>()) {
      if (probe.contains(t)) return [t];
    }
    return null;
  }

  /// Conjunctive: every group must land at least one hit. This is what keeps
  /// false positives down — "how much does paracetamol cost" does not fire the
  /// dosage rule because "cost" is absent from the second group.
  List<String>? _matchAllOf(String probe, dynamic groups) {
    if (groups == null) return null;
    final hits = <String>[];
    for (final group in (groups as List)) {
      String? found;
      for (final term in (group as List).cast<String>()) {
        if (probe.contains(term)) {
          found = term;
          break;
        }
      }
      if (found == null) return null;
      hits.add(found);
    }
    return hits;
  }

  // --------------------------------------------------------------- output

  GuardResult checkOutput(String text) {
    try {
      for (final rule in _outputRules) {
        if (rule['action'] == 'length_check') {
          final stripped = text.trim();
          final minChars = (rule['min_chars'] as int?) ?? 20;
          if (stripped.length < minChars || _isDegenerate(stripped)) {
            return GuardResult(
              blocked: true,
              category: rule['category'] as String?,
              ruleId: rule['id'] as String?,
              text: _responses[rule['response_key']]!,
            );
          }
          continue;
        }

        final pattern = _compiled[rule['id']];
        if (pattern == null) continue;
        final hit = pattern.firstMatch(text);
        if (hit != null) {
          return GuardResult(
            blocked: true,
            category: rule['category'] as String?,
            ruleId: rule['id'] as String?,
            text: _responses[rule['response_key']]!,
            matchedTerms: [hit.group(0)!],
          );
        }
      }
      return GuardResult(blocked: false, text: '${text.trim()}\n\n$disclaimer');
    } catch (_) {
      return GuardResult(
        blocked: true,
        category: 'guard_error',
        ruleId: 'ERR-00',
        text: _responses['unavailable']!,
      );
    }
  }

  /// Looping output — the characteristic failure of aggressively quantized
  /// sub-billion-parameter models, and one your 4-bit arms will hit.
  bool _isDegenerate(String text, {int window = 6, int repeats = 3}) {
    final words = text.split(RegExp(r'\s+'));
    if (words.length < window * repeats) return false;
    final seen = <String, int>{};
    for (var i = 0; i <= words.length - window; i++) {
      final gram = words.sublist(i, i + window).join(' ');
      seen[gram] = (seen[gram] ?? 0) + 1;
      if (seen[gram]! >= repeats) return true;
    }
    return false;
  }

  // ------------------------------------------------------------- pipeline

  /// Full guarded turn. [generate] is only invoked if the input passes.
  Future<(String, List<Map<String, dynamic>>)> process(
    String userText,
    Future<String> Function(String) generate,
  ) async {
    final log = <Map<String, dynamic>>[];

    final pre = checkInput(userText);
    log.add({'stage': 'input', ...pre.toLogRecord()});
    if (pre.blocked) return (pre.text, log);

    final raw = await generate(userText);

    final post = checkOutput(raw);
    log.add({'stage': 'output', ...post.toLogRecord()});
    return (post.text, log);
  }
}
