import 'package:flutter/material.dart';

import 'controllers/chat_controller.dart';
import 'guard.dart';
import 'screens/chat_screen.dart';
import 'services/history_db.dart';
import 'services/llm_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Both loaded from generated assets. rules.json is copied from
  // src/safety/rules.json and model_config.json is exported from
  // configs/base.yaml — run mobile/sync_rules.sh after changing either, or the
  // app will not behave like the system that was evaluated.
  final guard = await Guard.load();
  final config = await ModelConfig.load();

  // SWAP THIS LINE for the real backend once the Day 3 toolchain spike passes:
  //   final llm = LlamaCppService(config: config);
  // Keeping the mock as the default means the app always builds and runs, so a
  // native binding problem never blocks the rest of Day 12.
  final LlmService llm = MockLlmService();

  final controller = ChatController(guard: guard, llm: llm, db: HistoryDb());
  await controller.init();

  runApp(OhaiApp(controller: controller));
}

class OhaiApp extends StatelessWidget {
  final ChatController controller;
  const OhaiApp({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Offline Health AI-ASSIST',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF00696D)),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF00696D), brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: ChatScreen(controller: controller),
    );
  }
}
