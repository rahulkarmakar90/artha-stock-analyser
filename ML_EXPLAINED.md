# Artha — ML Predictions & Backtest Metrics Explained

---

## 1. Up / Neutral / Down (Random Forest, Bayesian, Ensemble)

### The Target Definition

Every model is trained to predict **tomorrow's direction** based on today's technical indicators.
The calculation is:

```
next day return = (tomorrow's close − today's close) / today's close
```

| If tomorrow's return is... | Label |
|---|---|
| **> +0.5%** | **Up (+1)** |
| **Between −0.5% and +0.5%** | **Neutral (0)** |
| **< −0.5%** | **Down (−1)** |

The 0.5% threshold filters out noise — tiny moves that are essentially flat.

### Example: TCS at ₹3,500

| Scenario | Tomorrow's close | Return | Label |
|---|---|---|---|
| Strong buying | ₹3,535 | +1.0% | **Up** |
| Sideways | ₹3,510 | +0.3% | **Neutral** |
| Selling pressure | ₹3,462 | −1.1% | **Down** |

---

## 2. How Each Model Predicts Direction

### Random Forest
Trains 100 decision trees on 5 years of your stock's daily data using 11 features
(RSI, MACD, Bollinger Band position, SMA ratio, volume ratio, 1/5/20-day momentum, volatility).
Each tree independently votes Up / Neutral / Down. The final answer is the majority vote.
**Confidence** = percentage of trees that agreed on the winning label.

### Bayesian (Gaussian Naive Bayes)
Asks: *"On past Up days, what did RSI typically look like? What about MACD?"*

It builds a probability distribution for each indicator under each label (Up/Neutral/Down),
then uses today's actual indicator values to compute which label is most likely given the
observed pattern. It is the most interpretable model — it is essentially pattern-matching
today's indicators against historical Up, Neutral, and Down days using Bayes' theorem.

### Ensemble
Averages the raw probabilities from Random Forest + Gradient Boosting, then picks the
highest probability as the final direction.

**Example:**
```
Random Forest says:   Up = 60%,  Neutral = 25%,  Down = 15%
Gradient Boosting:    Up = 50%,  Neutral = 30%,  Down = 20%

Ensemble average:     Up = 55%,  Neutral = 27.5%, Down = 17.5%
Final call → Up with 55% confidence
```

It is more stable than any single model because errors in one model are partially
cancelled by the other.

### What it means practically

If the **Ensemble says Down with 65% confidence** for RELIANCE, it means:
based on 5 years of how RELIANCE behaved when its RSI, MACD, momentum, and volume
looked exactly like they do today — on 65% of those past occasions, the stock closed
more than 0.5% lower the next day.

It is **not** a guarantee. It is a historical pattern match. The backtest tells you
how accurate these predictions have actually been for that specific stock.

---

## 3. Linear Regression Target

Unlike RF / Bayesian / Ensemble which classify into Up / Neutral / Down buckets,
Linear Regression predicts a **continuous number** — specifically tomorrow's return
percentage, which is then converted to a price.

### How it works

```
Step 1: Learn from 5 years of history
        "When RSI was X, MACD was Y, momentum was Z...
         what was the actual % return the next day?"

Step 2: Apply that relationship to today's indicators
        → predicted return % (e.g. −0.42%)

Step 3: Convert to price
        predicted price = today's close × (1 + predicted return)
        e.g. ₹3,500 × (1 − 0.0042) = ₹3,485
```

### Example: TCS at ₹3,500

| | |
|---|---|
| Today's close | ₹3,500 |
| Predicted return | −0.42% |
| **LR Target** | **₹3,485** |

The card is saying: *"Based on how today's indicators compare to 5 years of history,
the model's best continuous estimate of tomorrow's close is ₹3,485."*

### Comparison: LR vs the classifiers

| Model | Output type | What it answers |
|---|---|---|
| Random Forest | Up / Neutral / Down | Which direction is most likely? |
| Bayesian | Up / Neutral / Down | Which direction fits the pattern? |
| Ensemble | Up / Neutral / Down | Combined vote |
| **Linear Regression** | **A price (₹)** | **Exactly where might it close?** |

The LR number looks precise but is typically less reliable than the classifiers for
direction — stock returns are noisy and do not fit a straight line well. The **R²**
score shown alongside it tells you how well the model fit the training data
(closer to 1.0 = better fit; even 0.05–0.15 is normal for daily stock returns).

---

## 4. Backtest Metrics

### How the backtest works

Every single day in the 5-year history, the model:
1. **Trains** on the previous 252 trading days (~1 year)
2. **Predicts** direction for the next day
3. **Compares** that prediction to what actually happened
4. **Moves forward** one day and repeats

This is called **walk-forward validation** — it never peeks at future data.
It simulates what would have happened if you had run the model in real-time.

---

### Direction Accuracy %
> "How often did the model correctly call Up / Neutral / Down?"

Out of 1,000 test days, if it was right 520 times → **52% direction accuracy**.

Random guessing across 3 classes gives ~33%, so anything above 40–45% is meaningful.
Do not expect 80% — no model achieves that consistently on real stock data.

---

### Range Accuracy %
> "How often did the actual next-day close fall inside the predicted price band?"

The band is: `today's close ± (1.96 × 20-day volatility)`

Statistically, if returns were normally distributed this should capture ~95% of moves.
If your range accuracy is 85%, it means the stock had more extreme moves than the
model expected — telling you the volatility estimate is conservative.

**Example:** INFY at ₹1,500, 20-day volatility = 1.2%
```
Band = ₹1,500 ± (1.96 × 1.2%) = ₹1,470 – ₹1,530

Next day closes at ₹1,512  → inside the band  ✓
Next day closes at ₹1,440  → outside the band ✗  (surprise earnings miss)
```

---

### Cumulative Strategy Return %
> "If you had followed every Up signal over 5 years, what total return would you have got?"

The strategy is simple: **invest only on days the model predicted Up, sit in cash otherwise.**

If strategy return = **+38%**, that means ₹1 lakh became ₹1.38 lakh over 5 years
by following the model's signals.

---

### Cumulative Buy-and-Hold Return %
> "What if you had simply bought on day 1 and held for 5 years?"

If buy-and-hold = **+61%**, just holding the stock for 5 years without any ML would
have made more money than following the model's signals.

---

### Alpha %
> "Strategy return minus buy-and-hold return. Did the model actually add value?"

```
Alpha = Strategy Return − Buy-and-Hold Return

Example A:  38% − 61% = −23%   ← model destroyed value vs just holding
Example B:  38% − 20% = +18%   ← model outperformed passive holding
```

A **positive alpha** means the ML signals beat passive holding.
A **negative alpha** (common) means you would have been better off just buying and
holding — which is true for most large-cap stocks most of the time.
This is the most honest metric of whether the model is actually useful.

---

### N Test Days / N Trades

| Field | Meaning |
|---|---|
| **N Test Days** | Total days evaluated — typically 750–950 for a 5-year history |
| **N Trades** | Days where the model predicted Up and "invested" |

If N Trades = 280 out of 800 days, the model was selective — it only went long 35%
of the time. A low N Trades with a high strategy return means the model was picky
and right when it did act.

---

### Quick reference

| Metric | Healthy range | Be cautious if... |
|---|---|---|
| Direction Accuracy | > 45% | < 38% — barely better than random |
| Range Accuracy | > 88% | < 80% — volatility is underestimated |
| Alpha | Positive | Negative — just hold the stock instead |
| N Trades | 20–60% of days | < 5% — too few signals to trust |

---

## 5. Feature Importance Chart

Shown alongside the Random Forest prediction. It answers:
**"Which indicators mattered most in making today's prediction?"**

The longer the bar, the more that indicator influenced the model's Up/Neutral/Down call.
Common top features are momentum (return_1d, return_5d) and RSI — indicators that
capture recent price behaviour rather than absolute levels.

---

*Artha uses these models as research aids. Nothing here is financial advice.*
