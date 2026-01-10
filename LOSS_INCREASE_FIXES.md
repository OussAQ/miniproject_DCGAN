# Fixes for Increasing Losses in WGAN-GP

## Understanding the Problem

If you're seeing losses "increasing," it's likely one of these scenarios:

### Scenario A: Generator Loss Increasing (Positive) ❌
**Symptom**: G_loss going from -1 → 5 → 20
**Meaning**: Generator is failing to fool discriminator
**Cause**: Discriminator too strong, generator can't learn

### Scenario B: Discriminator Loss Too Negative ❌
**Symptom**: D_loss going from -1 → -20 → -50
**Meaning**: Discriminator is too powerful
**Cause**: Discriminator learning too fast, overpowering generator

---

## Fixes Applied

### 1. Reduced Discriminator Learning Rate ✅
**Change**: `LEARNING_RATE_D = 0.00001` (was 0.00002)
**Why**: Slows down discriminator learning to prevent overpowering
**Impact**: Discriminator learns slower, giving generator more time

### 2. Reduced N_CRITIC ✅
**Change**: `N_CRITIC = 3` (was 5)
**Why**: Fewer discriminator updates per generator update
**Impact**: Discriminator trained less frequently, less likely to overpower

### 3. Reduced Gradient Penalty ✅
**Change**: `LAMBDA_GP = 0.5` (was 1.0)
**Why**: Lower GP reduces gradient explosion risk
**Impact**: More stable gradients, less training instability

### 4. Increased Dropout ✅
**Change**: `DROPOUT_RATE = 0.3` (was 0.25)
**Why**: More regularization weakens discriminator
**Impact**: Discriminator less powerful, better balance

---

## What to Expect After These Fixes

### Immediate Effects:
- **Discriminator loss**: Should be less negative (closer to 0)
- **Generator loss**: Should decrease (become more negative)
- **Score gap**: Should decrease over time
- **Gradient penalty**: Should stay < 5.0

### Training Behavior:
- More balanced competition between G and D
- Smoother loss curves (less volatility)
- Better convergence

---

## Monitoring After Fixes

Watch for these improvements:

1. **D(real) - D(fake) gap decreasing**
   - Early: gap = 10-15
   - Later: gap = 3-5 ✅

2. **Generator loss trending negative**
   - Early: G_loss = 5-10
   - Later: G_loss = -5 to 0 ✅

3. **Gradient penalty stable**
   - Should stay between 0.5 - 2.0 ✅

4. **Losses stabilizing**
   - Less wild swings
   - Smooth convergence ✅

---

## If Losses Still Increase

### Try Even More Aggressive Settings:

```python
LEARNING_RATE_G = 0.00005  # Even lower
LEARNING_RATE_D = 0.000005  # Even lower (10:1 ratio maintained)
N_CRITIC = 1  # Train discriminator only once per generator update
LAMBDA_GP = 0.25  # Even lower gradient penalty
DROPOUT_RATE = 0.4  # More dropout
GRAD_CLIP = 0.25  # Tighter gradient clipping
```

### Alternative: Switch Optimizer

If Adam still unstable, try RMSprop:
```python
optimizerD = optim.RMSprop(netD.parameters(), lr=LEARNING_RATE_D)
optimizerG = optim.RMSprop(netG.parameters(), lr=LEARNING_RATE_G)
```

---

## Key Insight

**In WGAN, "increasing loss" doesn't always mean bad!**

- **Discriminator loss becoming more negative** = Discriminator learning (can be good early, bad if too extreme)
- **Generator loss becoming more positive** = Generator failing (always bad)
- **Generator loss becoming more negative** = Generator improving (always good)

**Focus on**:
1. Score gap (D(real) - D(fake)) - should decrease
2. Visual quality - most important!
3. Gradient penalty - should stay < 5.0

---

## Quick Diagnostic

After training, check:
```python
gap = D_real_scores[-1] - D_fake_scores[-1]
gp = GP_losses[-1]

if gap < 5.0 and gp < 5.0:
    print("✅ Training healthy!")
elif gap > 10.0:
    print("⚠️ Discriminator still too strong - reduce N_CRITIC further")
elif gp > 5.0:
    print("⚠️ Gradient penalty high - reduce LAMBDA_GP further")
```

---

**Remember**: Loss values in WGAN are different from traditional losses. Always check:
- Discriminator scores (real vs fake)
- Visual quality of generated images
- Score gap trends
