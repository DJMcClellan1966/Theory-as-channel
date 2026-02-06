"""
Example: Theory as a Channel

Shannon's communication theory says: a channel has a capacity C (bits per second).
If you send information at a rate below C, you can make errors arbitrarily rare
by using the right code (redundancy + decoding).

ML analogy:
  - "Sender"   = the true label or the teacher model's knowledge
  - "Channel"  = the noisy process (one model's predictions, or the student's limited capacity)
  - "Noise"    = model errors, approximation, or compression loss
  - "Receiver" = the final prediction we use (e.g. after error correction or distillation)

Two concrete views:

  1. ENSEMBLE AS CHANNEL: Send the same "message" (the true answer) through 3
     independent noisy channels (3 models). Each channel makes errors. By
     majority vote (decoding), we recover the message more reliably than any
     single channel. Redundancy (more models) trades bandwidth for accuracy.

  2. COMPRESSION AS CHANNEL: Teacher has "signal" (knowledge); student has
     limited capacity (fewer parameters). Channel capacity C = B * log2(1 + S/N)
     says: for a given SNR (how clean the teacher's signal is vs. noise), there's
     a maximum rate of information we can push through. So we can't arbitrarily
     compress a big teacher into a tiny student without loss—capacity bounds it.

This script demonstrates (1) with real numbers and (2) with the capacity formula.
"""

import numpy as np

from theory_as_channel import (
    ErrorCorrectingPredictions,
    channel_capacity,
)

def main():
    print("=" * 60)
    print("THEORY AS A CHANNEL - Example")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Part 1: Ensemble as a noisy channel — redundancy improves reliability
    # -------------------------------------------------------------------------
    print("\n1. ENSEMBLE AS CHANNEL (error correction via redundancy)")
    print("-" * 50)

    np.random.seed(42)
    n = 200
    true_labels = np.random.randint(0, 2, n)  # "Message" we want to receive

    # Three "channels" (models) that each make errors with probability ~0.2
    def noisy_channel(labels, error_prob=0.2):
        return np.where(np.random.rand(n) < (1 - error_prob), labels, 1 - labels)

    pred_A = noisy_channel(true_labels)
    pred_B = noisy_channel(true_labels)
    pred_C = noisy_channel(true_labels)

    acc_A = np.mean(pred_A == true_labels)
    acc_B = np.mean(pred_B == true_labels)
    acc_C = np.mean(pred_C == true_labels)

    print(f"   Single-model accuracy (each channel):  A={acc_A:.2%}, B={acc_B:.2%}, C={acc_C:.2%}")

    # Decode: majority vote across the 3 channels (like a repetition code)
    ec = ErrorCorrectingPredictions(redundancy_factor=3)
    predictions = np.column_stack([pred_A, pred_B, pred_C])
    decoded = ec.correct_predictions(predictions, method="majority_vote")
    acc_decoded = np.mean(decoded == true_labels)

    print(f"   After majority vote (decode):          {acc_decoded:.2%}")
    print("   -> Redundancy (3 channels) reduced effective error rate.")
    print("   -> Same idea as sending each bit 3 times and taking majority.")

    # -------------------------------------------------------------------------
    # Part 2: Channel capacity - how much "information" can the channel carry?
    # -------------------------------------------------------------------------
    print("\n2. CHANNEL CAPACITY (Shannon: C = B * log2(1 + S/N))")
    print("-" * 50)

    # Interpret: "signal" = correct predictions, "noise" = errors
    # Power ~ squared magnitude; for 0/1: signal power ~ mean(correct), noise ~ mean(wrong)
    signal_power = np.mean(true_labels.astype(float) ** 2) + 0.1
    noise_power = 0.2  # typical error variance
    C = channel_capacity(signal_power=signal_power, noise_power=noise_power, bandwidth=1.0)

    print(f"   Signal power (approx): {signal_power:.3f}")
    print(f"   Noise power (assumed): {noise_power:.3f}")
    print(f"   Capacity C = B * log2(1 + S/N) = {C:.3f} bits per use")
    print("   -> If we 'send' one bit per prediction, capacity tells us the")
    print("     maximum reliable rate we can achieve with the right 'code'.")
    print("     Here: 3 repetitions (ensemble) is a simple code that improves")
    print("     reliability.")

    # -------------------------------------------------------------------------
    # Part 3: Compression / distillation view
    # -------------------------------------------------------------------------
    print("\n3. COMPRESSION AS CHANNEL (teacher -> student)")
    print("-" * 50)

    # Suppose "teacher" has high SNR (accurate), "student" has limited capacity.
    teacher_snr_power = 10.0   # teacher is accurate → high signal
    noise_in_student = 2.0     # student approximates → some noise
    C_teacher_student = channel_capacity(
        signal_power=teacher_snr_power,
        noise_power=noise_in_student,
        bandwidth=1.0,
    )
    print(f"   Teacher signal power: {teacher_snr_power}, Student noise: {noise_in_student}")
    print(f"   Effective capacity (teacher->student): C = {C_teacher_student:.3f} bits")
    print("   -> You cannot transfer more than C bits of 'knowledge' per dimension")
    print("     without error. So compressing a huge teacher into a tiny student")
    print("     is limited by this capacity (and the student's parameter count).")

    print("\n" + "=" * 60)
    print("Summary: 'Theory as channel' = treat ML pipeline as communication;")
    print("use redundancy (ensemble) to correct errors, and capacity to bound compression.")
    print("=" * 60)


if __name__ == "__main__":
    main()
