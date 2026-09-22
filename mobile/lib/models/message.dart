/// A single turn in the conversation.
///
/// [ruleId] is non-null when the safety layer intervened. Keeping it on the
/// message rather than only in the log means the interface can show *why* a
/// response was refused, which is what makes the system auditable to a user
/// rather than merely opaque.
enum Sender { user, assistant, system }

class Message {
  final Sender sender;
  final String text;
  final DateTime at;
  final String? ruleId;
  final String? category;
  final bool isError;

  const Message({
    required this.sender,
    required this.text,
    required this.at,
    this.ruleId,
    this.category,
    this.isError = false,
  });

  bool get wasIntercepted => ruleId != null;

  Message.user(this.text)
      : sender = Sender.user,
        at = DateTime.now(),
        ruleId = null,
        category = null,
        isError = false;

  Message.assistant(this.text, {this.ruleId, this.category})
      : sender = Sender.assistant,
        at = DateTime.now(),
        isError = false;

  Message.error(this.text)
      : sender = Sender.system,
        at = DateTime.now(),
        ruleId = null,
        category = null,
        isError = true;
}
