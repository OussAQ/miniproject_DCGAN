# Fixing Blurry Generated Images

## Problem Diagnosis

Your metrics show:
- ✅ Good loss values
- ✅ Small score gap (-0.0047)
- ❌ **But images are blurry!**

**Root Cause**: The score gap is TOO SMALL! D(real) ≈ D(fake) means:
- Discriminator is too weak (can't distinguish real from fake)
- Generator isn't getting useful feedback
- Generator learns to produce "safe" blurry images that fool weak discriminator

---

## Solutions

### 1. Strengthen Discriminator (Most Important!)

The discriminator was weakened too much to fix instability. Now we need to balance it.

**Changes:**
- Reduce dropout: 0.3 → 0.15 (less regularization)
- Increase discriminator capacity: ndf=32 → ndf=48
- Slightly increase discriminator learning rate

### 2. Increase Generator Capacity

Generator might not have enough capacity to learn detailed features.

**Changes:**
- Increase generator filters: ngf=64 → ngf=128

### 3. Adjust Learning Rates

Current rates might be too conservative.

**Changes:**
- Generator LR: 0.0001 → 0.00015
- Discriminator LR: 0.00001 → 0.000015
- Maintain 10:1 ratio

### 4. Train Longer

50 epochs might not be enough for detailed features.

**Changes:**
- Increase epochs: 50 → 100

### 5. Add Feature Matching (Optional)

Force generator to match real image features.

---

## Implementation Priority

1. **Immediate**: Reduce dropout, increase generator capacity
2. **Next**: Adjust learning rates
3. **If still blurry**: Train longer, add feature matching
