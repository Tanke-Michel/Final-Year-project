**Guard performance**

- Block rate: 44/44 (100.0%)
- False positives on legitimate questions: 49/49 (100.0%)
- Blocked by payload rule: 13 (29.5%); by framing rule: 31

| Adversarial type | Blocked | Rate (%) |
|---|---|---|
| dan | 5/5 | 100.0 |
| direct | 9/9 | 100.0 |
| harmful output | 5/5 | 100.0 |
| jailbreak roleplay | 5/5 | 100.0 |
| misinformation | 5/5 | 100.0 |
| obfuscation | 5/5 | 100.0 |
| prompt injection | 5/5 | 100.0 |
| prompt leak | 5/5 | 100.0 |

Report both the block rate and the false-positive rate. A guard that blocks everything scores 100% and is useless in a clinic.
