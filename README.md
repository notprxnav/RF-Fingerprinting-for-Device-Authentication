# RF-Fingerprinting-for-Device-Authentication
# Deep Learning-Based RF Fingerprinting for Secure Physical-Layer Device Authentication
## Publication

Published in IEEE Access

Title:
Deep Learning-Based RF Fingerprinting for Secure Physical-Layer Device Authentication

DOI:
10.1109/ACCESS.2026.3692297

Paper:
[IEEE Xplore Link](https://ieeexplore.ieee.org/document/11515040)

## Why RF Fingerprinting?

Traditional authentication relies on cryptographic keys.

RF Fingerprinting provides:

- Device authentication
- Rogue transmitter detection
- Drone identification
- Military communication security
- IoT device verification
- Spectrum intelligence


System Pipeline:
Bit Generation
      ↓
Modulation (BPSK/QPSK)
      ↓
Hardware Impairments
(CFO,SCO,IQ imbalance...)
      ↓
I/Q Signal Generation
      ↓
Feature Extraction
(Time + FFT)
      ↓
Deep Learning Models
      ↓
Device Classification

## Dataset

20 RF Devices

Modulations:
- BPSK
- QPSK

Hardware Impairments:

- Sampling Clock Offset
- Carrier Frequency Offset
- Phase Noise
- Static Phase Rotation
- Power Amplifier Nonlinearity
- DC Offset
- I/Q Imbalance

## Applications

- Defence communication security
- Drone/UAV identification
- Rogue transmitter detection
- Wireless spectrum monitoring
- IoT authentication
- RF intelligence and signal forensics
- Secure military communication networks

## Future Work

- Real SDR-based RF captures
- Dynamic wireless channels
- Adversarial RF attacks
- Few-shot RF fingerprinting
- Quantum-inspired RF classification
- Edge deployment on SDR hardware
