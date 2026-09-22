import 'package:flutter/material.dart';
import '../models/message.dart';

/// A single message. Intercepted responses are visually distinct and carry the
/// rule identifier, so a refusal reads as a deliberate safety decision rather
/// than as the app malfunctioning. Showing the rule is also what makes the
/// system auditable to the person using it.
class MessageBubble extends StatelessWidget {
  final Message message;
  const MessageBubble({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final isUser = message.sender == Sender.user;
    final intercepted = message.wasIntercepted;

    final Color bg;
    final Color fg;
    if (isUser) {
      bg = scheme.primary;
      fg = scheme.onPrimary;
    } else if (message.isError) {
      bg = scheme.errorContainer;
      fg = scheme.onErrorContainer;
    } else if (intercepted) {
      bg = scheme.tertiaryContainer;
      fg = scheme.onTertiaryContainer;
    } else {
      bg = scheme.surfaceContainerHighest;
      fg = scheme.onSurface;
    }

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
            maxWidth: MediaQuery.of(context).size.width * 0.84),
        margin: const EdgeInsets.symmetric(vertical: 5, horizontal: 12),
        padding: const EdgeInsets.fromLTRB(14, 11, 14, 11),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 4),
            bottomRight: Radius.circular(isUser ? 4 : 16),
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (intercepted)
              Padding(
                padding: const EdgeInsets.only(bottom: 7),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Icon(Icons.shield_outlined, size: 15, color: fg),
                  const SizedBox(width: 6),
                  Text(
                    'Safety rule ${message.ruleId}',
                    style: TextStyle(
                      fontSize: 11.5,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.2,
                      color: fg,
                    ),
                  ),
                ]),
              ),
            SelectableText(
              message.text,
              style: TextStyle(color: fg, fontSize: 14.5, height: 1.42),
            ),
          ],
        ),
      ),
    );
  }
}
