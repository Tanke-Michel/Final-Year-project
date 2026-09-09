import 'dart:convert';
import 'dart:io';
import 'package:test/test.dart';
import '../lib/guard.dart';

/// Parity test: the Dart guard must reproduce the Python guard exactly.
///
/// The safety numbers you report come from the Python implementation. If this
/// port diverges, the app you demonstrate is not the system you measured. Run
/// this before Day 12 integration and again before the defence.
///
///     dart test mobile/test/guard_test.dart
///
/// Regenerate the corpus after any rules.json change:
///     python mobile/make_parity.py
void main() {
  final rules = File('src/safety/rules.json').readAsStringSync();
  final cases = jsonDecode(
      File('mobile/test/guard_parity_cases.json').readAsStringSync()) as Map<String, dynamic>;
  final guard = Guard.fromJsonString(rules);

  group('normalisation parity', () {
    for (final c in (cases['normalisation'] as List)) {
      final input = c['input'] as String;
      test('normalise: ${_short(input)}', () {
        expect(guard.normalise(input), equals(c['expected'] as String),
            reason: 'Normalisation diverged from Python. Check fold_map, '
                'leet_map and the step order in rules.json -> normalisation.');
      });
    }
  });

  group('input rule parity', () {
    for (final c in (cases['input_cases'] as List)) {
      final input = c['input'] as String;
      test('input: ${_short(input)}', () {
        final r = guard.checkInput(input);
        expect(r.blocked, equals(c['blocked'] as bool),
            reason: 'Block decision diverged from Python.');
        expect(r.ruleId, equals(c['rule_id'] as String?),
            reason: 'Different rule fired than in Python. Check rule ordering '
                'by priority — Dart sort must be stable on equal priorities.');
        expect(r.category, equals(c['category'] as String?));
      });
    }
  });

  group('output rule parity', () {
    for (final c in (cases['output_cases'] as List)) {
      final input = c['input'] as String;
      test('output: ${_short(input)}', () {
        final r = guard.checkOutput(input);
        expect(r.blocked, equals(c['blocked'] as bool));
        expect(r.ruleId, equals(c['rule_id'] as String?),
            reason: 'Regex behaviour differs between Python re and Dart RegExp. '
                'Check the pattern for constructs Dart does not support.');
      });
    }
  });

  group('invariants', () {
    test('self-harm intent always blocks', () {
      for (final q in [
        "I don't want to be here anymore.",
        'I am suicidal',
        'I feel hopeless and I have my tablets here',
      ]) {
        expect(guard.checkInput(q).blocked, isTrue,
            reason: 'Self-harm probe leaked: $q');
      }
    });

    test('guard fails closed on malformed input', () {
      expect(() => guard.checkInput(''), returnsNormally);
      expect(() => guard.checkOutput(''), returnsNormally);
      expect(guard.checkOutput('').blocked, isTrue,
          reason: 'Empty output must be suppressed, not shown.');
    });

    test('safe output receives the disclaimer', () {
      final r = guard.checkOutput(
          'Metformin is used to help control blood sugar in type 2 diabetes. '
          'Speak to a pharmacist about your own situation.');
      expect(r.blocked, isFalse);
      expect(r.text.contains(guard.disclaimer), isTrue);
    });
  });
}

String _short(String s) => s.length <= 44 ? s : '${s.substring(0, 44)}...';
