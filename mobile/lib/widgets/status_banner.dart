import 'package:flutter/material.dart';
import '../services/llm_service.dart';

/// Shows what is actually running. Two things a user of an offline health tool
/// deserves to see at a glance: that nothing is leaving the device, and whether
/// a real model is loaded or the app is running on the stand-in.
class StatusBanner extends StatelessWidget {
  final ModelStatus status;
  final String backendName;
  final String? error;

  const StatusBanner({
    super.key,
    required this.status,
    required this.backendName,
    this.error,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;

    late final Color bg, fg;
    late final IconData icon;
    late final String label;

    switch (status) {
      case ModelStatus.ready:
        bg = scheme.secondaryContainer;
        fg = scheme.onSecondaryContainer;
        icon = Icons.wifi_off;
        label = 'Offline · $backendName';
      case ModelStatus.loading:
        bg = scheme.surfaceContainerHighest;
        fg = scheme.onSurfaceVariant;
        icon = Icons.hourglass_empty;
        label = 'Loading model…';
      case ModelStatus.missing:
        bg = scheme.errorContainer;
        fg = scheme.onErrorContainer;
        icon = Icons.folder_off_outlined;
        label = 'Model file not found on device';
      case ModelStatus.error:
        bg = scheme.errorContainer;
        fg = scheme.onErrorContainer;
        icon = Icons.error_outline;
        label = error ?? 'Model failed to load';
      case ModelStatus.notLoaded:
        bg = scheme.surfaceContainerHighest;
        fg = scheme.onSurfaceVariant;
        icon = Icons.circle_outlined;
        label = 'Not loaded';
    }

    return Container(
      width: double.infinity,
      color: bg,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
      child: Row(children: [
        Icon(icon, size: 15, color: fg),
        const SizedBox(width: 8),
        Expanded(
          child: Text(label,
              style: TextStyle(fontSize: 12, color: fg, fontWeight: FontWeight.w500),
              maxLines: 2, overflow: TextOverflow.ellipsis),
        ),
      ]),
    );
  }
}
