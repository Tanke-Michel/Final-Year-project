import 'package:flutter/foundation.dart';

import '../guard.dart';
import '../models/message.dart';
import '../services/history_db.dart';
import '../services/llm_service.dart';

/// Orchestrates a guarded conversation turn.
///
/// SAFETY ARCHITECTURE. [_llm] is private and is never called directly by this
/// class. The only path to generation runs through [Guard.process], which
/// checks the input, invokes the callback only if the input passes, then checks
/// the output. There is deliberately no method on this controller that reaches
/// the model without the guard: bypassing it would require editing this file,
/// not merely forgetting a call.
///
/// This matters because the model is a sub-billion-parameter network being
/// asked about medicines. It cannot be trusted to refuse a dosing question on
/// its own, and the evaluation showed that fine-tuning alone does not make it
/// reliable at that.
class ChatController extends ChangeNotifier {
  final Guard _guard;
  final LlmService _llm;
  final HistoryDb _db;

  final List<Message> _messages = [];
  bool _busy = false;

  ChatController({
    required Guard guard,
    required LlmService llm,
    required HistoryDb db,
  })  : _guard = guard,
        _llm = llm,
        _db = db;

  List<Message> get messages => List.unmodifiable(_messages);
  bool get busy => _busy;
  ModelStatus get modelStatus => _llm.status;
  String get backendName => _llm.backendName;
  String? get modelError => _llm.lastError;

  /// Exposed ONLY for the Day 13 benchmark screen, which measures raw inference
  /// cost. Conversation must continue to go through [send], which routes every
  /// turn through the guard. Do not call generate() from anywhere else.
  LlmService get llm => _llm;

  Future<void> init() async {
    await _db.open();
    await _llm.load();
    _messages.add(Message.assistant(
      'Ask me about a medicine — what it is used for, its common side effects, '
      'or precautions to be aware of.\n\nI cannot give doses, and I cannot tell '
      'you what condition you have. For those, speak to a pharmacist, nurse or '
      'doctor.',
    ));
    notifyListeners();
  }

  Future<void> send(String text) async {
    final trimmed = text.trim();
    if (trimmed.isEmpty || _busy) return;

    _messages.add(Message.user(trimmed));
    _busy = true;
    notifyListeners();
    await _db.logMessage('user', trimmed, null);

    try {
      // The guard owns the call to the model. If check_input blocks, the
      // callback is never invoked and the model never sees the text.
      final (reply, log) = await _guard.process(trimmed, _llm.generate);

      String? firedRule;
      String? firedCategory;
      for (final entry in log) {
        final rid = entry['rule_id'] as String?;
        if (rid == null) continue;
        firedRule ??= rid;
        firedCategory ??= entry['category'] as String?;
        await _db.logIntervention(
            entry['stage'] as String, rid, entry['category'] as String?);
      }

      _messages.add(Message.assistant(reply,
          ruleId: firedRule, category: firedCategory));
      await _db.logMessage('assistant', reply, firedRule);
    } catch (e) {
      // A failure in generation must not surface raw model or binding errors to
      // someone asking a health question. Fail visibly but calmly.
      _messages.add(Message.error(
        'Something went wrong producing an answer. Please try again, or ask a '
        'pharmacist or other health professional.',
      ));
      debugPrint('generation failed: $e');
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  Future<void> clear() async {
    _messages.clear();
    await _db.clearHistory();
    notifyListeners();
  }

  /// Rule-fire counts, for the results chapter and for the demo screen.
  Future<Map<String, int>> interventionCounts() => _db.interventionCounts();

  @override
  void dispose() {
    _llm.dispose();
    _db.close();
    super.dispose();
  }
}
