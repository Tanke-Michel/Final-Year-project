import 'package:path/path.dart' as p;
import 'package:sqflite/sqflite.dart';

/// Local storage for conversation history and safety interventions.
///
/// PRIVACY: the intervention log records the rule that fired, its category and
/// a timestamp. It does NOT record the question text. The log exists to count
/// rule fires for the results chapter, not to keep a record of what health
/// questions a user asked on a shared or borrowed phone. Conversation history
/// is stored separately and is user-deletable.
class HistoryDb {
  Database? _db;

  Future<void> open() async {
    final path = p.join(await getDatabasesPath(), 'ohai.db');
    _db = await openDatabase(
      path,
      version: 1,
      onCreate: (db, _) async {
        await db.execute('''
          CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            rule_id TEXT,
            created_at INTEGER NOT NULL
          )''');
        // Deliberately no text column here.
        await db.execute('''
          CREATE TABLE interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            category TEXT,
            created_at INTEGER NOT NULL
          )''');
      },
    );
  }

  Future<void> logMessage(String sender, String text, String? ruleId) async {
    await _db?.insert('messages', {
      'sender': sender,
      'text': text,
      'rule_id': ruleId,
      'created_at': DateTime.now().millisecondsSinceEpoch,
    });
  }

  Future<void> logIntervention(String stage, String ruleId, String? category) async {
    await _db?.insert('interventions', {
      'stage': stage,
      'rule_id': ruleId,
      'category': category,
      'created_at': DateTime.now().millisecondsSinceEpoch,
    });
  }

  /// Rule-fire counts for the results chapter.
  Future<Map<String, int>> interventionCounts() async {
    final rows = await _db?.rawQuery(
        'SELECT rule_id, COUNT(*) AS n FROM interventions GROUP BY rule_id ORDER BY n DESC');
    return {for (final r in rows ?? []) r['rule_id'] as String: r['n'] as int};
  }

  Future<void> clearHistory() async => _db?.delete('messages');

  Future<void> close() async => _db?.close();
}
