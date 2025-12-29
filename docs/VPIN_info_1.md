Based on the video transcript and the methodology of Marco López de Prado (referenced extensively in the video), here is a detailed breakdown of how to calculate **VPIN** (Volume-Synchronized Probability of Informed Trading) using **Binance Kline (candlestick) data**.

### 1. Data Acquisition & Pre-processing
Standard VPIN requires tick-by-tick data to determine if a trade was initiated by a buyer or seller. However, downloading and processing billions of ticks is computationally expensive. The method described uses **1-minute Klines** as an efficient proxy.

**Binance API Data Required:**
When querying the Binance API for Klines (e.g., `GET /api/v3/klines`), you get a list of parameters. You need specifically:
*   **Index 5:** `Volume` (Total Volume of the candle, let's call it $V_{total}$)
*   **Index 9:** `Taker buy base asset volume` (Volume bought via market orders, let's call it $V_{buy}$)

**Deriving Sell Volume:**
Binance does not explicitly provide "Sell Volume". You must calculate it:
$$V_{sell} = V_{total} - V_{buy}$$

*   **Assumption:** This method assumes that "Taker Buy" represents aggressive buying (informed or demand pressure) and the remainder is aggressive selling (supply pressure).

---

### 2. Defining the Volume Bucket Size ($V$)
VPIN operates on **Volume Clock** time, not Chronological time. This means we sample the market every time a specific amount of volume ($V$) changes hands, rather than every minute.

**The Heuristic for $V$:**
The video recommends aiming for approximately **50 Volume Bars (Buckets) per day** to ensure statistical robustness.

1.  **Calculate Average Daily Volume (ADV):** Take a reference period (e.g., the last 3 months) and calculate the average total volume traded per day.
2.  **Calculate Bucket Size:**
    $$V = \frac{\text{Average Daily Volume}}{50}$$

*Example:* If Bitcoin trades 50,000 BTC per day on average, your bucket size $V$ is 1,000 BTC.

---

### 3. The Resampling Process (Time to Volume)
This is the most critical algorithmic step. You must convert your sequence of 1-minute Klines into a sequence of Volume Buckets of size $V$.

**The Logic:**
You iterate through the 1-minute Klines chronologically and "pour" their volume into the current bucket.

**The Fractional Split Algorithm:**
Since a 1-minute candle usually doesn't align perfectly with the bucket capacity, you must split candles fractionally.

Let:
*   $V_{rem}$: Remaining capacity in the current bucket.
*   $v_t$: Total volume of the current 1-minute candle.
*   $vb_t$: Buy volume of the current 1-minute candle.
*   $vs_t$: Sell volume of the current 1-minute candle.

**Scenario A: Candle fits in the bucket ($v_t \leq V_{rem}$)**
1.  Add $vb_t$ and $vs_t$ to the current bucket's accumulators.
2.  Decrease $V_{rem}$ by $v_t$.
3.  Move to the next candle.

**Scenario B: Candle overflows the bucket ($v_t > V_{rem}$)**
1.  Calculate the split fraction: $\phi = \frac{V_{rem}}{v_t}$
2.  **Fill the current bucket:** Add $\phi \times vb_t$ to Buy Accumulator and $\phi \times vs_t$ to Sell Accumulator.
3.  **Close the bucket:** The bucket is full. Record its Buy ($B$) and Sell ($S$) totals. Reset accumulators for a new bucket.
4.  **Carry over remainder:** The remaining volume $(1 - \phi) \times v_t$ goes into the *new* bucket.
    *   Add $(1 - \phi) \times vb_t$ to the new Buy Accumulator.
    *   Add $(1 - \phi) \times vs_t$ to the new Sell Accumulator.
    *   Update $V_{rem}$ for the new bucket.

---

### 4. Calculating Order Flow Imbalance (OI)
Once you have a list of Volume Buckets, you calculate the absolute order flow imbalance for each bucket $i$.

$$OI_i = |B_i - S_i|$$

Where:
*   $B_i$ = Total Buy volume aggregated in Bucket $i$.
*   $S_i$ = Total Sell volume aggregated in Bucket $i$.
*   Note that $B_i + S_i = V$ (The fixed bucket size).

---

### 5. Calculating VPIN
VPIN is essentially a rolling average of the imbalance ratio. You define a window size $n$ (The video suggests **$n=50$**, matching the daily bar count heuristic).

For the current bucket $t$:

$$VPIN_t = \frac{\sum_{j=t-n+1}^{t} OI_j}{n \times V}$$

*   **Numerator:** Sum of absolute imbalances over the last $n$ buckets.
*   **Denominator:** Total volume traded in those $n$ buckets (which is simply $n \times V$).

The result is a value between 0 and 1.

---

### 6. Normalization via CDF (The "Toxic" Signal)
A raw VPIN value (e.g., 0.25) is ambiguous. Does 0.25 mean high risk for BTC? Maybe not. For a low liquidity altcoin? Maybe yes. To make this actionable, you use an **Empirical Cumulative Distribution Function (CDF)**.

1.  **Reference History:** Collect VPIN values calculated over a significant past period (e.g., 3 to 6 months prior to the event).
2.  **Build Distribution:** Create a sorted list or histogram of these historical VPIN values.
3.  **Rank Current VPIN:** Compare the current VPIN value against the history to find its percentile.
    $$CDF(VPIN_t) = P(X \leq VPIN_t)$$
    *(i.e., "What percentage of historical VPINs were lower than my current VPIN?")*

**Interpretation Thresholds (from the video):**
*   **CDF $\approx$ 0.50:** Normal market noise. Healthy liquidity.
*   **CDF > 0.90:** Elevated toxicity.
*   **CDF > 0.95:** High toxicity (Warning zone).
*   **CDF > 0.99:** Extreme toxicity. This indicates that informed traders (or a massive liquidation cascade) are dominating the order flow. **This is the signal used to predict the Flash Crash.**

### Summary Algorithm
1.  **Fetch** 1m Klines (Volume, TakerBuyVolume).
2.  **Compute** Sell Volume ($Vol - BuyVol$).
3.  **Determine** Bucket Size $V$ ($AvgDailyVol / 50$).
4.  **Resample** 1m data into Volume Buckets of size $V$ (splitting candles where necessary).
5.  **Compute** Imbalance $|Buy - Sell|$ for every bucket.
6.  **Compute** Rolling VPIN (Sum of imbalances / Total Volume).
7.  **Apply CDF** using historical data to get a probability (0-100%).
8.  **Trigger** risk controls if CDF > 0.95 or 0.99.