"""
Diagnostic script to understand why losses are increasing in WGAN-GP training.
Run this after training to analyze the results.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def diagnose_training(G_losses, D_losses, GP_losses, D_real_scores, D_fake_scores, 
                     D_grad_norms, G_grad_norms):
    """
    Comprehensive diagnostic of WGAN-GP training health.
    """
    print("=" * 60)
    print("WGAN-GP TRAINING DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Check final values
    print("\n1. FINAL METRICS:")
    print(f"   Generator Loss: {G_losses[-1]:.4f}")
    print(f"   Discriminator Loss: {D_losses[-1]:.4f}")
    print(f"   Gradient Penalty: {GP_losses[-1]:.4f}")
    print(f"   D(real): {D_real_scores[-1]:.4f}")
    print(f"   D(fake): {D_fake_scores[-1]:.4f}")
    
    # 2. Check score gap
    gap = D_real_scores[-1] - D_fake_scores[-1]
    print(f"\n2. DISCRIMINATOR SCORE GAP: {gap:.4f}")
    if gap < 3.0:
        print("   ✅ GOOD: Small gap indicates balanced training")
    elif gap < 10.0:
        print("   ⚠️  MODERATE: Gap is reasonable but could be smaller")
    else:
        print("   ❌ BAD: Large gap - discriminator too strong!")
        print("   → Discriminator is overpowering generator")
        print("   → Generator cannot learn effectively")
    
    # 3. Check gradient penalty
    print(f"\n3. GRADIENT PENALTY: {GP_losses[-1]:.4f}")
    if GP_losses[-1] < 2.0:
        print("   ✅ EXCELLENT: GP is well-controlled")
    elif GP_losses[-1] < 5.0:
        print("   ⚠️  WARNING: GP is getting high but acceptable")
    else:
        print("   ❌ CRITICAL: GP too high - training unstable!")
        print("   → Gradients are exploding")
        print("   → Reduce LAMBDA_GP or learning rates")
    
    # 4. Check loss trends
    print("\n4. LOSS TRENDS:")
    
    # Generator loss trend
    if len(G_losses) >= 10:
        early_G = np.mean(G_losses[:5])
        late_G = np.mean(G_losses[-5:])
        G_trend = late_G - early_G
        print(f"   Generator Loss: {early_G:.4f} → {late_G:.4f} (change: {G_trend:+.4f})")
        if G_trend < -2.0:
            print("   ✅ GOOD: Generator loss decreasing (improving)")
        elif G_trend < 2.0:
            print("   ⚠️  MODERATE: Generator loss relatively stable")
        else:
            print("   ❌ BAD: Generator loss increasing (getting worse)")
            print("   → Generator is failing to learn")
    
    # Discriminator loss trend
    if len(D_losses) >= 10:
        early_D = np.mean(D_losses[:5])
        late_D = np.mean(D_losses[-5:])
        D_trend = late_D - early_D
        print(f"   Discriminator Loss: {early_D:.4f} → {late_D:.4f} (change: {D_trend:+.4f})")
        if D_trend < -5.0:
            print("   ⚠️  WARNING: D_loss becoming very negative (discriminator too strong)")
        elif abs(D_trend) < 5.0:
            print("   ✅ GOOD: Discriminator loss relatively stable")
        else:
            print("   ⚠️  MODERATE: Discriminator loss changing significantly")
    
    # 5. Check score convergence
    print("\n5. SCORE CONVERGENCE:")
    if len(D_real_scores) >= 10:
        early_gap = np.mean([D_real_scores[i] - D_fake_scores[i] for i in range(5)])
        late_gap = np.mean([D_real_scores[i] - D_fake_scores[i] for i in range(-5, 0)])
        gap_change = late_gap - early_gap
        print(f"   Score Gap: {early_gap:.4f} → {late_gap:.4f} (change: {gap_change:+.4f})")
        if gap_change < -2.0:
            print("   ✅ EXCELLENT: Gap decreasing - training is converging!")
        elif gap_change < 0:
            print("   ✅ GOOD: Gap slowly decreasing")
        elif gap_change < 2.0:
            print("   ⚠️  MODERATE: Gap relatively stable")
        else:
            print("   ❌ BAD: Gap increasing - discriminator getting stronger!")
    
    # 6. Check gradient norms
    print("\n6. GRADIENT NORMS:")
    print(f"   Discriminator: {D_grad_norms[-1]:.4f}")
    print(f"   Generator: {G_grad_norms[-1]:.4f}")
    if D_grad_norms[-1] > 0.4 or G_grad_norms[-1] > 0.4:
        print("   ⚠️  WARNING: Gradients are being heavily clipped")
        print("   → Consider reducing learning rates")
    else:
        print("   ✅ GOOD: Gradients are reasonable")
    
    # 7. Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT:")
    print("=" * 60)
    
    issues = []
    if gap > 10.0:
        issues.append("Discriminator too strong (gap > 10)")
    if GP_losses[-1] > 5.0:
        issues.append("Gradient penalty too high (> 5.0)")
    if len(G_losses) >= 10 and np.mean(G_losses[-5:]) > np.mean(G_losses[:5]) + 2.0:
        issues.append("Generator loss increasing")
    if D_losses[-1] < -50.0:
        issues.append("Discriminator loss extremely negative (< -50)")
    
    if not issues:
        print("✅ Training appears HEALTHY!")
        print("   - Losses are within acceptable ranges")
        print("   - Training is converging")
        print("   - Check generated images for visual quality")
    else:
        print("⚠️  Training has ISSUES:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n   RECOMMENDED FIXES:")
        if gap > 10.0:
            print("   → Reduce N_CRITIC (try 3 instead of 5)")
            print("   → Reduce discriminator learning rate further")
            print("   → Add more dropout to discriminator")
        if GP_losses[-1] > 5.0:
            print("   → Reduce LAMBDA_GP (try 0.5 instead of 1)")
            print("   → Reduce learning rates")
        if any("Generator loss increasing" in i for i in issues):
            print("   → Reduce generator learning rate")
            print("   → Increase N_CRITIC to give generator more training")
    
    print("=" * 60)

if __name__ == "__main__":
    # Example usage - load from checkpoint or use training variables
    print("To use this diagnostic:")
    print("1. After training, run: diagnose_training(G_losses, D_losses, GP_losses,")
    print("                                        D_real_scores, D_fake_scores,")
    print("                                        D_grad_norms, G_grad_norms)")
